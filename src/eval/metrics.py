"""Evaluation metrics and test routines for rating and ranking models."""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error

from src.model.components import (
    compute_weighted_neg_mahalanobis,
    CrossDomainValidDataset, SingleDomainValidDataset,
    estimate_scale, rand_embed_for_id, to_device,
)
from src.model.rating_model import RatingPredictionModel
from src.model.ranking_model import BPRRankingModel
from src.utils import ensure_dir


# ============================================================
# Rating evaluation
# ============================================================

def _test_rating_fusion(m_src, m_tgt, test_ds, T, device, batch_size=256):
    """Test cross-domain rating model (fusion path)."""
    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    m_src.eval(); m_tgt.eval()
    T_dev = T.to(device) if torch.is_tensor(T) else torch.tensor(T, dtype=torch.float32).to(device)
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            b = to_device(batch, device)
            ws = m_src.weight_learner(b["u_src"])
            wt = m_tgt.weight_learner(b["u_tgt"])
            alpha = (b["m_src"] / (b["m_src"] + b["m_tgt"] + 1e-8)).unsqueeze(1)
            comb = torch.softmax(alpha * torch.matmul(ws, T_dev) + (1 - alpha) * wt, dim=-1)
            scores = compute_weighted_neg_mahalanobis(m_tgt.gmm, comb, b["i_tgt"])
            pred = m_tgt.rating_predictor(scores, b["time"].unsqueeze(-1))
            all_preds.extend(pred.cpu().numpy().tolist())
            all_labels.extend(b["rating"].cpu().numpy().tolist())
    mse = mean_squared_error(all_labels, all_preds)
    mae = mean_absolute_error(all_labels, all_preds)
    return mse ** 0.5, mae


def eval_rating_model(src_states, tgt_states, u_emb_src, u_emb_tgt, i_emb_tgt,
                      gmm_src, gmm_tgt, T, test_interactions, pair_name,
                      cfg_gmm, seeds, device, batch_size=256):
    """Evaluate cross-domain rating over multiple seeds."""
    trainable = cfg_gmm.get("trainable", True)
    test_ds = CrossDomainValidDataset(test_interactions, u_emb_tgt, i_emb_tgt, u_emb_src)
    d_s = list(u_emb_src.values())[0].shape[0]
    d_t = list(u_emb_tgt.values())[0].shape[0]

    rmse_list, mae_list = [], []
    for i, (ss, ts) in enumerate(zip(src_states, tgt_states)):
        ms = RatingPredictionModel(d_s, gmm_src, trainable).to(device)
        mt = RatingPredictionModel(d_t, gmm_tgt, trainable).to(device)
        ms.load_state_dict(ss); mt.load_state_dict(ts)
        rmse, mae = _test_rating_fusion(ms, mt, test_ds, T, device, batch_size)
        print(f"  Seed {seeds[i]}: RMSE={rmse:.4f}, MAE={mae:.4f}")
        rmse_list.append(rmse); mae_list.append(mae)

    avg_rmse, avg_mae = np.mean(rmse_list), np.mean(mae_list)
    var_rmse = np.var(rmse_list)
    var_mae = np.var(mae_list)
    print(f"Average RMSE: {avg_rmse:.4f} (var: {var_rmse:.6f}), MAE: {avg_mae:.4f} (var: {var_mae:.6f})")

    stored = f"stored/{pair_name}"
    ensure_dir(stored)
    with open(f"{stored}/test_results.txt", "w") as f:
        f.write(f"Average RMSE: {avg_rmse:.4f} (var: {var_rmse:.6f}), MAE: {avg_mae:.4f} (var: {var_mae:.6f})\n")
        for idx, (r, m) in enumerate(zip(rmse_list, mae_list)):
            f.write(f"Run {idx+1}: RMSE={r:.4f}, MAE={m:.4f}\n")


def eval_rating_tgtonly(tgt_states, u_emb_tgt, i_emb_tgt, gmm_tgt,
                        test_interactions, pair_name, cfg_gmm, seeds, device,
                        batch_size=256):
    """Evaluate target-only rating over multiple seeds."""
    trainable = cfg_gmm.get("trainable", True)
    test_ds = SingleDomainValidDataset(test_interactions, u_emb_tgt, i_emb_tgt)
    d_t = list(u_emb_tgt.values())[0].shape[0]

    rmse_list, mae_list = [], []
    for i, ts in enumerate(tgt_states):
        mt = RatingPredictionModel(d_t, gmm_tgt, trainable).to(device)
        mt.load_state_dict(ts); mt.eval()
        loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
        preds, labels = [], []
        with torch.no_grad():
            for batch in loader:
                b = to_device(batch, device)
                p = mt(b["u"], b["i"], b["time"].unsqueeze(-1))
                preds.extend(p.cpu().numpy().tolist())
                labels.extend(b["rating"].cpu().numpy().tolist())
        rmse = mean_squared_error(labels, preds) ** 0.5
        mae = mean_absolute_error(labels, preds)
        print(f"  Seed {seeds[i]} (tgt-only): RMSE={rmse:.4f}, MAE={mae:.4f}")
        rmse_list.append(rmse); mae_list.append(mae)

    avg_rmse, avg_mae = np.mean(rmse_list), np.mean(mae_list)
    print(f"TgtOnly Average RMSE: {avg_rmse:.4f}, MAE: {avg_mae:.4f}")

    stored = f"stored/{pair_name}"
    ensure_dir(stored)
    with open(f"{stored}/test_tgtonly_results.txt", "w") as f:
        f.write(f"Average RMSE: {avg_rmse:.4f}, MAE: {avg_mae:.4f}\n")
        for idx, (r, m) in enumerate(zip(rmse_list, mae_list)):
            f.write(f"Run {idx+1}: RMSE={r:.4f}, MAE={m:.4f}\n")


# ============================================================
# Ranking evaluation (HR@K, NDCG@K)
# ============================================================

def _build_test_data(interactions):
    """Leave-one-out: last interaction per user is test."""
    user_pos, all_items = {}, set()
    for (u, i, _, t) in interactions:
        all_items.add(i)
        user_pos.setdefault(u, []).append((i, t))
    test_data = {}
    for u, its in user_pos.items():
        its.sort(key=lambda x: x[1])
        test_data[u] = its[-1]
    return test_data, all_items


def _build_item_matrix(all_items, item_dict, device):
    item_np = {k: np.asarray(v, dtype=np.float32) for k, v in item_dict.items()}
    scale = estimate_scale(np.stack(list(item_np.values()), axis=0) if item_np else np.zeros((0, 32)))
    ids, embs, cache = list(all_items), [], {}
    for i in ids:
        if i in item_np:
            embs.append(torch.tensor(item_np[i], device=device))
        else:
            s = str(i)
            if s not in cache:
                cache[s] = rand_embed_for_id(f"I_TGT::{s}", scale["d"], scale["std"], True)
            embs.append(torch.tensor(cache[s], device=device))
    return ids, torch.stack(embs, dim=0)


def eval_bpr_model(src_states, tgt_states, u_emb_src, u_emb_tgt, i_emb_tgt,
                   gmm_src, gmm_tgt, T, test_interactions, pair_name,
                   cfg_gmm, seeds, device, topk=10):
    """Evaluate cross-domain BPR over multiple seeds."""
    trainable = cfg_gmm.get("trainable", True)
    test_data, all_items = _build_test_data(test_interactions)
    d_s = list(u_emb_src.values())[0].shape[0]
    d_t = list(u_emb_tgt.values())[0].shape[0]

    item_ids, item_embs = _build_item_matrix(all_items, i_emb_tgt, device)
    num_items = item_embs.size(0)
    T_dev = (torch.tensor(T, dtype=torch.float32) if not torch.is_tensor(T) else T).to(device)

    # Precompute scales
    u_tgt_np = {k: np.asarray(v, dtype=np.float32) for k, v in u_emb_tgt.items()}
    u_src_np = {k: np.asarray(v, dtype=np.float32) for k, v in u_emb_src.items()}
    u_tgt_mat = np.stack(list(u_tgt_np.values()), axis=0) if u_tgt_np else np.zeros((0, 32))
    u_src_mat = np.stack(list(u_src_np.values()), axis=0) if u_src_np else np.zeros((0, u_tgt_mat.shape[1] if u_tgt_mat.size else 32))
    sc_tgt = estimate_scale(u_tgt_mat)
    sc_src = estimate_scale(u_src_mat)

    HR_all, NDCG_all = [], []
    for idx, (ss, ts) in enumerate(zip(src_states, tgt_states)):
        ms = BPRRankingModel(d_s, gmm_src, trainable).to(device)
        mt = BPRRankingModel(d_t, gmm_tgt, trainable).to(device)
        ms.load_state_dict(ss); mt.load_state_dict(ts)
        ms.eval(); mt.eval()

        HRs, NDCGs = [], []
        u_tgt_cache, u_src_cache = {}, {}
        with torch.no_grad():
            for u, (pos_i, pos_t) in test_data.items():
                # User embeddings with fallback
                if u in u_tgt_np:
                    ue_tgt = torch.tensor(u_tgt_np[u], device=device).unsqueeze(0); m_tgt = 1.0
                else:
                    s = str(u)
                    if s not in u_tgt_cache:
                        u_tgt_cache[s] = rand_embed_for_id(f"U_TGT::{s}", sc_tgt["d"], sc_tgt["std"], True)
                    ue_tgt = torch.tensor(u_tgt_cache[s], device=device).unsqueeze(0); m_tgt = 0.0
                if u in u_src_np:
                    ue_src = torch.tensor(u_src_np[u], device=device).unsqueeze(0); m_src = 1.0
                else:
                    s = str(u)
                    if s not in u_src_cache:
                        u_src_cache[s] = rand_embed_for_id(f"U_SRC::{s}", sc_src["d"], sc_src["std"], True)
                    ue_src = torch.tensor(u_src_cache[s], device=device).unsqueeze(0); m_src = 0.0

                ws = ms.weight_learner(ue_src); wt = mt.weight_learner(ue_tgt)
                alpha = m_src / (m_src + m_tgt + 1e-8)
                comb = torch.softmax(alpha * torch.matmul(ws, T_dev) + (1 - alpha) * wt, dim=-1)

                scores = compute_weighted_neg_mahalanobis(mt.gmm, comb.repeat(num_items, 1), item_embs)
                t_rep = torch.full((num_items, 1), pos_t, device=device)
                ratings = mt.rating_predictor(scores, t_rep)

                k = min(topk, num_items)
                _, indices = torch.topk(ratings, k=k)
                recs = [item_ids[j] for j in indices.cpu().numpy().tolist()]
                HRs.append(1.0 if pos_i in recs else 0.0)
                if pos_i in recs:
                    NDCGs.append(1.0 / np.log2(recs.index(pos_i) + 2))
                else:
                    NDCGs.append(0.0)

        hr, ndcg = np.mean(HRs), np.mean(NDCGs)
        print(f"  Seed {seeds[idx]}: HR@{topk}={hr:.4f}, NDCG@{topk}={ndcg:.4f}")
        HR_all.append(hr); NDCG_all.append(ndcg)

    print(f"Average HR@{topk}: {np.mean(HR_all):.4f}, NDCG@{topk}: {np.mean(NDCG_all):.4f}")
    stored = f"stored/{pair_name}"
    ensure_dir(stored)
    with open(f"{stored}/test_bpr_results.txt", "w") as f:
        f.write(f"Average HR@{topk}: {np.mean(HR_all):.4f}, NDCG@{topk}: {np.mean(NDCG_all):.4f}\n")
        for i, (h, n) in enumerate(zip(HR_all, NDCG_all)):
            f.write(f"Run {i+1}: HR={h:.4f}, NDCG={n:.4f}\n")


def eval_bpr_tgtonly(tgt_states, u_emb_tgt, i_emb_tgt, gmm_tgt,
                     test_interactions, pair_name, cfg_gmm, seeds, device, topk=10):
    """Evaluate target-only BPR over multiple seeds."""
    trainable = cfg_gmm.get("trainable", True)
    test_data, all_items = _build_test_data(test_interactions)
    d_t = list(u_emb_tgt.values())[0].shape[0]
    item_ids, item_embs = _build_item_matrix(all_items, i_emb_tgt, device)
    num_items = item_embs.size(0)

    u_tgt_np = {k: np.asarray(v, dtype=np.float32) for k, v in u_emb_tgt.items()}
    sc_tgt = estimate_scale(np.stack(list(u_tgt_np.values()), axis=0) if u_tgt_np else np.zeros((0, 32)))

    HR_all, NDCG_all = [], []
    for idx, ts in enumerate(tgt_states):
        mt = BPRRankingModel(d_t, gmm_tgt, trainable).to(device)
        mt.load_state_dict(ts); mt.eval()

        HRs, NDCGs, u_cache = [], [], {}
        with torch.no_grad():
            for u, (pos_i, pos_t) in test_data.items():
                if u in u_tgt_np:
                    ue = torch.tensor(u_tgt_np[u], device=device).unsqueeze(0)
                else:
                    s = str(u)
                    if s not in u_cache:
                        u_cache[s] = rand_embed_for_id(f"U_TGT::{s}", sc_tgt["d"], sc_tgt["std"], True)
                    ue = torch.tensor(u_cache[s], device=device).unsqueeze(0)

                w = torch.softmax(mt.weight_learner(ue), dim=-1)
                scores = compute_weighted_neg_mahalanobis(mt.gmm, w.repeat(num_items, 1), item_embs)
                t_rep = torch.full((num_items, 1), pos_t, device=device)
                ratings = mt.rating_predictor(scores, t_rep)

                k = min(topk, num_items)
                _, indices = torch.topk(ratings, k=k)
                recs = [item_ids[j] for j in indices.cpu().numpy().tolist()]
                HRs.append(1.0 if pos_i in recs else 0.0)
                NDCGs.append(1.0 / np.log2(recs.index(pos_i) + 2) if pos_i in recs else 0.0)

        hr, ndcg = np.mean(HRs), np.mean(NDCGs)
        print(f"  Seed {seeds[idx]} (tgt-only): HR@{topk}={hr:.4f}, NDCG@{topk}={ndcg:.4f}")
        HR_all.append(hr); NDCG_all.append(ndcg)

    print(f"TgtOnly Average HR@{topk}: {np.mean(HR_all):.4f}, NDCG@{topk}: {np.mean(NDCG_all):.4f}")
    stored = f"stored/{pair_name}"
    ensure_dir(stored)
    with open(f"{stored}/test_bpr_tgtonly_results.txt", "w") as f:
        f.write(f"Average HR@{topk}: {np.mean(HR_all):.4f}, NDCG@{topk}: {np.mean(NDCG_all):.4f}\n")
        for i, (h, n) in enumerate(zip(HR_all, NDCG_all)):
            f.write(f"Run {i+1}: HR={h:.4f}, NDCG={n:.4f}\n")
