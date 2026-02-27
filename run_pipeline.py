#!/usr/bin/env python
"""
DUP-OT Pipeline: from raw data to evaluation.

Usage:
    python run_pipeline.py --config config.yaml --stages all
    python run_pipeline.py --stages gen_data,split,preprocess
    python run_pipeline.py --stages train,eval
"""

import argparse
import os
import json
import math
from collections import defaultdict

import yaml
import torch

from src.utils import get_device, ensure_dir, load_json, save_json, load_pickle, save_pickle
from src.data import filter_dataset, gen_data, time_split, flatten_nested, judge_domain_time_order
from src.preprocess import process_data, train_autoencoder
from src.model.gmm import fit_gmm_to_items
from src.model.components import build_user_pos_items
from src.transport import compute_cost_matrix, compute_transport_plan


ALL_STAGES = [
    "gen_data", "split", "preprocess", "train", "eval",
    "train_ablation", "eval_ablation",
]


def _resolve_paths(cfg):
    """Build all derived file paths from config."""
    root = cfg["dataset_root"]
    src, tgt = cfg["source_domain"], cfg["target_domain"]
    pair = f"{src}_{tgt}"
    return {
        "src_raw": f"{root}/raw/{src}.json.gz",
        "tgt_raw": f"{root}/raw/{tgt}.json.gz",
        "src_filtered": f"{root}/filtered/{src}_filtered.json",
        "tgt_filtered": f"{root}/filtered/{tgt}_filtered.json",
        "src_interactions": f"{root}/processed/{src}_interactions.json",
        "tgt_interactions": f"{root}/processed/{tgt}_interactions.json",
        "pair_name": pair,
        "stored": f"stored/{pair}",
    }


def _time_decay(records, ref_time, cfg_td):
    """Apply exponential time decay to records in-place."""
    MONTH = 30 * 86400
    half_life = cfg_td.get("half_life_months", 6.0)
    max_delta = cfg_td.get("max_delta_months", 36.0)
    lambda_ = math.log(2) / half_life
    for r in records:
        delta_m = min((ref_time - r['time']) / MONTH, max_delta)
        w = math.exp(-lambda_ * delta_m)
        r['time'] = w / (1.0 + w)  # range: (0, 0.5]


def _to_interaction_list(records):
    return [(r['user_id'], r['item_id'], r['rating'], r['time']) for r in records]


def main():
    parser = argparse.ArgumentParser(description="DUP-OT Pipeline")
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument(
    '--dataset_pair',
    type=str,
    default=None,
    help='Override dataset pair in config, format: source,target (e.g., Books,Electronics)'
)
    parser.add_argument('--stages', type=str, default='all',
                        help='Comma-separated stages or "all"')
    parser.add_argument('--mode', type=str, default='rating', help='rating | ranking')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    if args.dataset_pair is not None:
        source_domain, target_domain = args.dataset_pair.split(',')
        source_domain = source_domain.strip()
        target_domain = target_domain.strip()

        cfg['source_domain'] = source_domain
        cfg['target_domain'] = target_domain

    stages = ALL_STAGES if args.stages == "all" else [s.strip() for s in args.stages.split(",")]
    
    if args.mode is not None:
        cfg['mode'] = args.mode
    
    device = get_device(cfg.get("device", "auto"))
    paths = _resolve_paths(cfg)
    pair_name = paths["pair_name"]
    seeds = tuple(cfg.get("seeds", [7, 11, 17, 33, 2025]))
    mode = cfg.get("mode", "rating")

    print(f"Device: {device}")
    print(f"Pair: {cfg['source_domain']} → {cfg['target_domain']}")
    print(f"Mode: {mode}")
    print(f"Stages: {stages}")
    print(f"GMM trainable: {cfg['gmm'].get('trainable', True)}")

    # ================================================================
    # Stage: gen_data (filter + build interaction dict)
    # ================================================================
    if "gen_data" in stages:
        print("\n" + "=" * 60)
        print("Stage: gen_data")
        print("=" * 60)
        for domain, raw, filt, inter in [
            (cfg["source_domain"], paths["src_raw"], paths["src_filtered"], paths["src_interactions"]),
            (cfg["target_domain"], paths["tgt_raw"], paths["tgt_filtered"], paths["tgt_interactions"]),
        ]:
            print(f"\n--- {domain} ---")
            if not os.path.exists(filt):
                filter_dataset(raw, filt, cfg["filter"])
            else:
                print(f"  Filtered file exists: {filt}")
            if not os.path.exists(inter):
                gen_data(filt, inter, cfg["gen_data"])
            else:
                print(f"  Interaction file exists: {inter}")

    # ================================================================
    # Stage: split (time-based split + feasibility check)
    # ================================================================
    source_data = load_json(paths["src_interactions"])
    target_data = load_json(paths["tgt_interactions"])

    if "split" in stages:
        print("\n" + "=" * 60)
        print("Stage: split")
        print("=" * 60)

    # Feasibility assessment
    assessment = judge_domain_time_order(source_data, target_data)
    print("Source-Target Assessment:", assessment["recommendation"])
    if not assessment['A_as_source']['feasible']:
        raise ValueError("Source-target pair not feasible based on time order.")
    ratio_train = assessment['A_as_source']['chosen_keep_ratio']

    # Split
    by_user = cfg.get("split", {}).get("by_user", False)
    tgt_ratios = tuple(cfg.get("split", {}).get("target_ratios", [0.8, 0.1, 0.1]))

    train_source, _ = time_split(source_data, (ratio_train, 1 - ratio_train), by_user)
    train_target, val_target, test_target = time_split(target_data, tgt_ratios, by_user)

    # Leakage check
    train_end_src = max(r['time'] for r in flatten_nested(train_source))
    val_start_tgt = min(r['time'] for r in flatten_nested(val_target))
    test_start_tgt = min(r['time'] for r in flatten_nested(test_target))
    if train_end_src >= val_start_tgt or train_end_src >= test_start_tgt:
        raise ValueError("Data leakage detected!")
    print("No data leakage detected.")

    # ================================================================
    # Stage: preprocess (embeddings + autoencoder + GMM + OT)
    # ================================================================
    if "preprocess" in stages:
        print("\n" + "=" * 60)
        print("Stage: preprocess")
        print("=" * 60)

    # Review-based embeddings
    u_src, i_src, u_tgt, i_tgt = process_data(
        train_source, train_target, pair_name, cfg["review_encoder"], device)

    # AutoEncoder
    all_features = torch.stack(
        list(u_src.values()) + list(i_src.values()) +
        list(u_tgt.values()) + list(i_tgt.values())
    ).to(device)
    ae = train_autoencoder(all_features, pair_name, cfg["autoencoder"], device)

    # Dim-reduced embeddings
    reduced_path = f"{paths['stored']}/dim_reduced_embeddings.json"
    ensure_dir(paths["stored"])
    if os.path.exists(reduced_path):
        raw = load_json(reduced_path)
        all_reduced = {int(k): torch.tensor(v) for k, v in raw.items()}
    else:
        print("Computing dim-reduced embeddings...")
        with torch.no_grad():
            all_reduced = {}
            bs = 256
            for idx in range(0, all_features.size(0), bs):
                batch = all_features[idx:idx + bs]
                _, z = ae(batch)
                for j in range(z.size(0)):
                    all_reduced[idx + j] = z[j].cpu()
        save_json({k: v.tolist() for k, v in all_reduced.items()}, reduced_path)

    # Build ID → embedding dicts
    src_uids = list(u_src.keys())
    src_iids = list(i_src.keys())
    tgt_uids = list(u_tgt.keys())
    tgt_iids = list(i_tgt.keys())

    src_id2idx = {id_: idx for idx, id_ in enumerate(src_uids + src_iids)}
    offset = len(src_uids) + len(src_iids)
    tgt_id2idx = {id_: idx + offset for idx, id_ in enumerate(tgt_uids + tgt_iids)}

    u_src_r = {uid: all_reduced[src_id2idx[uid]] for uid in src_uids}
    i_src_r = {iid: all_reduced[src_id2idx[iid]] for iid in src_iids}
    u_tgt_r = {uid: all_reduced[tgt_id2idx[uid]] for uid in tgt_uids}
    i_tgt_r = {iid: all_reduced[tgt_id2idx[iid]] for iid in tgt_iids}

    # Reduced embedding tensors for GMM
    i_src_tensor = torch.stack([i_src_r[iid] for iid in src_iids]).to(device)
    i_tgt_tensor = torch.stack([i_tgt_r[iid] for iid in tgt_iids]).to(device)

    # GMM fitting
    src_gmm_path = f"{paths['stored']}/source_gmm.pkl"
    tgt_gmm_path = f"{paths['stored']}/target_gmm.pkl"
    cfg_gmm = cfg["gmm"]

    if os.path.exists(src_gmm_path) and os.path.exists(tgt_gmm_path):
        print("Loading fitted GMMs...")
        gmm_src = load_pickle(src_gmm_path)
        gmm_tgt = load_pickle(tgt_gmm_path)
    else:
        print("Fitting GMMs...")
        gmm_src = fit_gmm_to_items(i_src_tensor, cfg_gmm)
        gmm_tgt = fit_gmm_to_items(i_tgt_tensor, cfg_gmm)
        save_pickle(gmm_src, src_gmm_path)
        save_pickle(gmm_tgt, tgt_gmm_path)
    print(f"Source GMM: K={gmm_src.n_components}, Target GMM: K={gmm_tgt.n_components}")

    # OT plan
    cost = compute_cost_matrix(gmm_src, gmm_tgt,
                               return_squared=cfg.get("transport", {}).get("return_squared", False))
    T = compute_transport_plan(cost)
    print(f"Transport plan shape: {T.shape}")

    # ================================================================
    # Prepare training data (time decay + interaction format)
    # ================================================================
    cfg_td = cfg.get("time_decay", {})
    src_flat = flatten_nested(train_source)
    tgt_flat = flatten_nested(train_target)
    val_flat = flatten_nested(val_target)
    test_flat = flatten_nested(test_target)

    ref_src = max(r['time'] for r in src_flat)
    ref_tgt = max(r['time'] for r in tgt_flat)

    _time_decay(src_flat, ref_src, cfg_td)
    _time_decay(tgt_flat, ref_tgt, cfg_td)
    _time_decay(val_flat, ref_tgt, cfg_td)
    _time_decay(test_flat, ref_tgt, cfg_td)

    # Save reference times
    save_json({"source_ref": ref_src, "target_ref": ref_tgt},
              f"{paths['stored']}/reference_times.json")

    # For both rating and ranking modes, use standard interaction lists.
    # BPR negative sampling is now handled dynamically inside BPRDataset.
    src_inter = _to_interaction_list(src_flat)
    tgt_inter = _to_interaction_list(tgt_flat)
    val_inter = _to_interaction_list(val_flat)
    test_inter = _to_interaction_list(test_flat)

    # Build train user-positive-items for evaluation masking (target domain)
    tgt_train_user_pos = build_user_pos_items(tgt_inter)

    cfg_train = cfg["training"]

    # ================================================================
    # Stage: train
    # ================================================================
    if "train" in stages:
        print("\n" + "=" * 60)
        print(f"Stage: train ({mode})")
        print("=" * 60)

        if mode == "rating":
            from src.model.rating_model import train_rating_model
            src_states, tgt_states = train_rating_model(
                src_inter, tgt_inter, val_inter,
                u_src_r, i_src_r, u_tgt_r, i_tgt_r,
                gmm_src, gmm_tgt, T, pair_name,
                cfg_train, cfg_gmm, seeds, device)
        else:
            from src.model.ranking_model import train_bpr_model
            src_states, tgt_states = train_bpr_model(
                src_inter, tgt_inter, val_inter,
                u_src_r, i_src_r, u_tgt_r, i_tgt_r,
                gmm_src, gmm_tgt, T, pair_name,
                cfg_train, cfg_gmm, seeds, device)

    # ================================================================
    # Stage: eval
    # ================================================================
    if "eval" in stages:
        print("\n" + "=" * 60)
        print(f"Stage: eval ({mode})")
        print("=" * 60)

        # Ensure models are loaded
        if "train" not in stages:
            if mode == "rating":
                from src.model.rating_model import train_rating_model
                src_states, tgt_states = train_rating_model(
                    src_inter, tgt_inter, val_inter,
                    u_src_r, i_src_r, u_tgt_r, i_tgt_r,
                    gmm_src, gmm_tgt, T, pair_name,
                    cfg_train, cfg_gmm, seeds, device)
            else:
                from src.model.ranking_model import train_bpr_model
                src_states, tgt_states = train_bpr_model(
                    src_inter, tgt_inter, val_inter,
                    u_src_r, i_src_r, u_tgt_r, i_tgt_r,
                    gmm_src, gmm_tgt, T, pair_name,
                    cfg_train, cfg_gmm, seeds, device)

        eval_bs = cfg.get("eval", {}).get("batch_size", 256)
        topk = cfg.get("eval", {}).get("topk", 10)

        if mode == "rating":
            from src.eval.metrics import eval_rating_model
            eval_rating_model(src_states, tgt_states, u_src_r, u_tgt_r, i_tgt_r,
                              gmm_src, gmm_tgt, T, test_inter, pair_name,
                              cfg_gmm, seeds, device, eval_bs)
        else:
            from src.eval.metrics import eval_bpr_model
            eval_bpr_model(src_states, tgt_states, u_src_r, u_tgt_r, i_tgt_r,
                           gmm_src, gmm_tgt, T, test_inter, pair_name,
                           cfg_gmm, seeds, device, topk,
                           train_user_pos=tgt_train_user_pos)

    # ================================================================
    # Stage: train_ablation (target-only)
    # ================================================================
    if "train_ablation" in stages:
        print("\n" + "=" * 60)
        print(f"Stage: train_ablation ({mode}, target-only)")
        print("=" * 60)

        if mode == "rating":
            from src.model.rating_model import train_target_only_rating
            tgt_only_states = train_target_only_rating(
                tgt_inter, val_inter, u_tgt_r, i_tgt_r, gmm_tgt,
                pair_name, cfg_train, cfg_gmm, seeds, device)
        else:
            from src.model.ranking_model import train_bpr_tgtonly
            tgt_only_states = train_bpr_tgtonly(
                tgt_inter, val_inter, u_src_r, u_tgt_r, i_tgt_r,
                gmm_tgt, pair_name, cfg_train, cfg_gmm, seeds, device)

    # ================================================================
    # Stage: eval_ablation
    # ================================================================
    if "eval_ablation" in stages:
        print("\n" + "=" * 60)
        print(f"Stage: eval_ablation ({mode}, target-only)")
        print("=" * 60)

        if "train_ablation" not in stages:
            if mode == "rating":
                from src.model.rating_model import train_target_only_rating
                tgt_only_states = train_target_only_rating(
                    tgt_inter, val_inter, u_tgt_r, i_tgt_r, gmm_tgt,
                    pair_name, cfg_train, cfg_gmm, seeds, device)
            else:
                from src.model.ranking_model import train_bpr_tgtonly
                tgt_only_states = train_bpr_tgtonly(
                    tgt_inter, val_inter, u_src_r, u_tgt_r, i_tgt_r,
                    gmm_tgt, pair_name, cfg_train, cfg_gmm, seeds, device)

        eval_bs = cfg.get("eval", {}).get("batch_size", 256)
        topk = cfg.get("eval", {}).get("topk", 10)

        if mode == "rating":
            from src.eval.metrics import eval_rating_tgtonly
            eval_rating_tgtonly(tgt_only_states, u_tgt_r, i_tgt_r, gmm_tgt,
                                test_inter, pair_name, cfg_gmm, seeds, device, eval_bs)
        else:
            from src.eval.metrics import eval_bpr_tgtonly
            eval_bpr_tgtonly(tgt_only_states, u_tgt_r, i_tgt_r, gmm_tgt,
                             test_inter, pair_name, cfg_gmm, seeds, device, topk,
                             train_user_pos=tgt_train_user_pos)

    print("\n" + "=" * 60)
    print("Pipeline complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
