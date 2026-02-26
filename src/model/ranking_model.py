"""BPR-based ranking models: cross-domain + target-only ablation."""

import os
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.model.gmm import GMMWrapper
from src.model.components import (
    weightLearner, ratingPredictor, compute_weighted_neg_mahalanobis,
    BPRDataset, CrossDomainBPRValidDataset,
    build_user_pos_items, collect_all_items,
    to_device,
)
from src.transport.ot_plan import compute_mmd
from src.utils import set_seed, ensure_dir


class BPRLoss(nn.Module):
    def forward(self, pos_scores, neg_scores):
        return -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8))


class BPRRankingModel(nn.Module):
    """GMM-based BPR ranking model."""

    def __init__(self, d: int, sklearn_gmm, trainable_gmm: bool = True):
        super().__init__()
        K = sklearn_gmm.n_components
        self.gmm = GMMWrapper(sklearn_gmm, trainable=trainable_gmm)
        self.weight_learner = weightLearner(d, K)
        self.rating_predictor = ratingPredictor(K)

    def forward(self, user_emb, pos_item_emb, neg_item_emb, pos_t, neg_t):
        w = self.weight_learner(user_emb)
        pos_scores = compute_weighted_neg_mahalanobis(self.gmm, w, pos_item_emb)
        neg_scores = compute_weighted_neg_mahalanobis(self.gmm, w, neg_item_emb)
        return self.rating_predictor(pos_scores, pos_t), self.rating_predictor(neg_scores, neg_t)


def _make_optimizer(model, lr, gmm_lr_scale, weight_decay):
    if model.gmm.trainable:
        gmm_p = list(model.gmm.parameters())
        other_p = [p for n, p in model.named_parameters() if not n.startswith('gmm.')]
        return torch.optim.Adam([
            {'params': other_p, 'lr': lr},
            {'params': gmm_p, 'lr': lr * gmm_lr_scale},
        ], weight_decay=weight_decay)
    else:
        trainable = [p for p in model.parameters() if p.requires_grad]
        return torch.optim.Adam(trainable, lr=lr, weight_decay=weight_decay)


def _train_bpr_one_run(m_src, m_tgt, src_ds, tgt_ds, val_ds, T, cfg_train, device):
    bs = cfg_train["batch_size"]
    lr = cfg_train["learning_rate"]
    wd = cfg_train.get("weight_decay", 1e-4)
    gmm_lr_scale = cfg_train.get("gmm_lr_scale", 0.1)
    lambda_align = cfg_train.get("lambda_align", 0.1)
    num_epochs = cfg_train["num_epochs"]
    patience = cfg_train.get("patience", 7)

    # NOTE: shuffle=True ensures different negative samples each epoch
    # because BPRDataset samples negatives dynamically in __getitem__
    src_loader = DataLoader(src_ds, batch_size=bs, shuffle=True, num_workers=0, pin_memory=True)
    tgt_loader = DataLoader(tgt_ds, batch_size=bs, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=0, pin_memory=True)

    bpr = BPRLoss().to(device)
    opt_src = _make_optimizer(m_src, lr, gmm_lr_scale, wd)
    opt_tgt = _make_optimizer(m_tgt, lr, gmm_lr_scale, wd)
    opt_align = torch.optim.Adam(
        list(m_src.weight_learner.parameters()) + list(m_tgt.weight_learner.parameters()),
        lr=lr * cfg_train.get("align_lr_scale", 0.1), weight_decay=wd)

    T_dev = (torch.tensor(T, dtype=torch.float32) if not torch.is_tensor(T) else T).to(device)

    best_val, es = float('inf'), 0
    best_ss = copy.deepcopy(m_src.state_dict())
    best_ts = copy.deepcopy(m_tgt.state_dict())

    for epoch in range(num_epochs):
        # Source
        m_src.train(); loss_s = 0.0
        for batch in src_loader:
            u, pi, ni, pt, nt = to_device(batch, device)
            opt_src.zero_grad()
            pp, np_ = m_src(u, pi, ni, pt.unsqueeze(-1), nt.unsqueeze(-1))
            loss = bpr(pp, np_); loss.backward(); opt_src.step()
            loss_s += loss.item()
        print(f"  Epoch [{epoch+1}/{num_epochs}] Src BPR: {loss_s/max(len(src_loader),1):.4f}")

        # Target
        m_tgt.train(); loss_t = 0.0
        for batch in tgt_loader:
            u, pi, ni, pt, nt = to_device(batch, device)
            opt_tgt.zero_grad()
            pp, np_ = m_tgt(u, pi, ni, pt.unsqueeze(-1), nt.unsqueeze(-1))
            loss = bpr(pp, np_); loss.backward(); opt_tgt.step()
            loss_t += loss.item()
        print(f"  Epoch [{epoch+1}/{num_epochs}] Tgt BPR: {loss_t/max(len(tgt_loader),1):.4f}")

        # Alignment
        if lambda_align > 0:
            m_src.train(); m_tgt.train()
            al_total, n_steps = 0.0, 0
            for sb, tb in zip(
                iter(DataLoader(src_ds, batch_size=bs, shuffle=True, num_workers=0)),
                iter(DataLoader(tgt_ds, batch_size=bs, shuffle=True, num_workers=0))):
                su, tu = sb[0].to(device), tb[0].to(device)
                opt_align.zero_grad()
                ws = m_src.weight_learner(su); wt = m_tgt.weight_learner(tu)
                transported = torch.softmax(torch.matmul(ws, T_dev), dim=-1)
                al = lambda_align * compute_mmd(transported, wt.detach())
                al.backward(); opt_align.step()
                al_total += al.item(); n_steps += 1
            if n_steps:
                print(f"  Epoch [{epoch+1}/{num_epochs}] Align: {al_total/n_steps:.6f}")

        # Validation (fusion)
        m_src.eval(); m_tgt.eval()
        vl = 0.0
        with torch.no_grad():
            for batch in val_loader:
                b = to_device(batch, device)
                ws = m_src.weight_learner(b["u_src"])
                wt = m_tgt.weight_learner(b["u_tgt"])
                alpha = (b["m_src"] / (b["m_src"] + b["m_tgt"] + 1e-8)).unsqueeze(1)
                comb = torch.softmax(alpha * torch.matmul(ws, T_dev) + (1 - alpha) * wt, dim=-1)
                pos_s = compute_weighted_neg_mahalanobis(m_tgt.gmm, comb, b["pos_i_tgt"])
                neg_s = compute_weighted_neg_mahalanobis(m_tgt.gmm, comb, b["neg_i_tgt"])
                pp = m_tgt.rating_predictor(pos_s, b["pos_time"].unsqueeze(-1))
                np_ = m_tgt.rating_predictor(neg_s, b["neg_time"].unsqueeze(-1))
                vl += bpr(pp, np_).item()
        cv = vl / max(len(val_loader), 1)
        print(f"  Epoch [{epoch+1}/{num_epochs}] Val BPR: {cv:.4f}")

        if cv < best_val:
            best_val = cv; es = 0
            best_ss = copy.deepcopy(m_src.state_dict())
            best_ts = copy.deepcopy(m_tgt.state_dict())
        else:
            es += 1
            if es >= patience:
                print(f"  Early stopping. Best: {best_val:.4f}"); break

    return best_ss, best_ts


def train_bpr_model(src_inter, tgt_inter, val_inter,
                    u_emb_src, i_emb_src, u_emb_tgt, i_emb_tgt,
                    gmm_src, gmm_tgt, T, pair_name,
                    cfg_train, cfg_gmm, seeds, device):
    """Multi-seed cross-domain BPR training.

    Parameters
    ----------
    src_inter, tgt_inter, val_inter : list of (user, item, rating, time)
        Standard interaction lists (NOT pre-built BPR pairs).
    """
    trainable = cfg_gmm.get("trainable", True)
    cfg_t = {**cfg_train, "gmm_lr_scale": cfg_gmm.get("gmm_lr_scale", 0.1)}

    # Build negative-sampling helpers
    src_all_items = collect_all_items(src_inter)
    tgt_all_items = collect_all_items(tgt_inter)
    src_user_pos = build_user_pos_items(src_inter)
    tgt_user_pos = build_user_pos_items(tgt_inter)

    # For val negative sampling: use union of train + val positives
    val_user_pos = build_user_pos_items(val_inter)
    val_neg_user_pos = {}
    for u in set(list(tgt_user_pos.keys()) + list(val_user_pos.keys())):
        val_neg_user_pos[u] = tgt_user_pos.get(u, set()) | val_user_pos.get(u, set())

    src_ds = BPRDataset(src_inter, {**u_emb_src, **i_emb_src},
                        src_all_items, src_user_pos)
    tgt_ds = BPRDataset(tgt_inter, {**u_emb_tgt, **i_emb_tgt},
                        tgt_all_items, tgt_user_pos)
    val_ds = CrossDomainBPRValidDataset(val_inter, u_emb_tgt, i_emb_tgt, u_emb_src,
                                        tgt_all_items, val_neg_user_pos)

    stored = f"stored/{pair_name}"
    ensure_dir(stored)
    sp = [f"{stored}/bpr_src_seed{sd}.pth" for sd in seeds]
    tp = [f"{stored}/bpr_tgt_seed{sd}.pth" for sd in seeds]

    if all(os.path.exists(p) for p in sp + tp):
        print("Loading pre-trained BPR models...")
        return ([torch.load(p, map_location=device) for p in sp],
                [torch.load(p, map_location=device) for p in tp])

    print("Training cross-domain BPR...")
    d_s = list(u_emb_src.values())[0].shape[0]
    d_t = list(u_emb_tgt.values())[0].shape[0]
    src_states, tgt_states = [], []
    for sd in seeds:
        print(f"\n===== Seed {sd} =====")
        set_seed(sd)
        ms = BPRRankingModel(d_s, gmm_src, trainable).to(device)
        mt = BPRRankingModel(d_t, gmm_tgt, trainable).to(device)
        ss, ts = _train_bpr_one_run(ms, mt, src_ds, tgt_ds, val_ds, T, cfg_t, device)
        src_states.append(ss); tgt_states.append(ts)

    for s, p in zip(src_states, sp): torch.save(s, p)
    for s, p in zip(tgt_states, tp): torch.save(s, p)
    return src_states, tgt_states


# ============================================================
# Target-only BPR ablation
# ============================================================

def _train_bpr_tgtonly_one(model, tgt_ds, val_ds, cfg_train, cfg_gmm, device):
    bs = cfg_train["batch_size"]
    lr = cfg_train["learning_rate"]
    patience = cfg_train.get("patience_ablation", 5)
    bpr = BPRLoss().to(device)
    opt = _make_optimizer(model, lr, cfg_gmm.get("gmm_lr_scale", 0.1),
                          cfg_train.get("weight_decay", 1e-4))

    loader = DataLoader(tgt_ds, batch_size=bs, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=0, pin_memory=True)

    best_val, es, best_state = float('inf'), 0, copy.deepcopy(model.state_dict())
    for epoch in range(cfg_train["num_epochs"]):
        model.train(); tl = 0.0
        for batch in loader:
            u, pi, ni, pt, nt = to_device(batch, device)
            opt.zero_grad()
            pp, np_ = model(u, pi, ni, pt.unsqueeze(-1), nt.unsqueeze(-1))
            loss = bpr(pp, np_); loss.backward(); opt.step()
            tl += loss.item()

        model.eval(); vl = 0.0
        with torch.no_grad():
            for batch in val_loader:
                b = to_device(batch, device)
                w = model.weight_learner(b["u_tgt"])
                comb = torch.softmax(w, dim=-1)
                ps = compute_weighted_neg_mahalanobis(model.gmm, comb, b["pos_i_tgt"])
                ns = compute_weighted_neg_mahalanobis(model.gmm, comb, b["neg_i_tgt"])
                pp = model.rating_predictor(ps, b["pos_time"].unsqueeze(-1))
                np_ = model.rating_predictor(ns, b["neg_time"].unsqueeze(-1))
                vl += bpr(pp, np_).item()
        cv = vl / len(val_loader)
        if cv < best_val:
            best_val = cv; best_state = copy.deepcopy(model.state_dict()); es = 0
        else:
            es += 1
            if es >= patience: break
    return best_state


def train_bpr_tgtonly(tgt_inter, val_inter, u_emb_src, u_emb_tgt, i_emb_tgt,
                      gmm_tgt, pair_name, cfg_train, cfg_gmm, seeds, device):
    """Target-only BPR training with dynamic negative sampling.

    Parameters
    ----------
    tgt_inter, val_inter : list of (user, item, rating, time)
        Standard interaction lists.
    """
    trainable = cfg_gmm.get("trainable", True)

    tgt_all_items = collect_all_items(tgt_inter)
    tgt_user_pos = build_user_pos_items(tgt_inter)
    val_user_pos = build_user_pos_items(val_inter)
    val_neg_user_pos = {}
    for u in set(list(tgt_user_pos.keys()) + list(val_user_pos.keys())):
        val_neg_user_pos[u] = tgt_user_pos.get(u, set()) | val_user_pos.get(u, set())

    tgt_ds = BPRDataset(tgt_inter, {**u_emb_tgt, **i_emb_tgt},
                        tgt_all_items, tgt_user_pos)
    val_ds = CrossDomainBPRValidDataset(val_inter, u_emb_tgt, i_emb_tgt, u_emb_src,
                                        tgt_all_items, val_neg_user_pos)

    stored = f"stored/{pair_name}"
    ensure_dir(stored)
    paths = [f"{stored}/bpr_tgtonly_seed{sd}.pth" for sd in seeds]
    if all(os.path.exists(p) for p in paths):
        return [torch.load(p, map_location=device) for p in paths]

    d_t = list(u_emb_tgt.values())[0].shape[0]
    states = []
    for sd in seeds:
        print(f"\n===== Seed {sd} (tgt-only BPR) =====")
        set_seed(sd)
        m = BPRRankingModel(d_t, gmm_tgt, trainable).to(device)
        s = _train_bpr_tgtonly_one(m, tgt_ds, val_ds, cfg_train, cfg_gmm, device)
        states.append(s)
    for s, p in zip(states, paths): torch.save(s, p)
    return states
