"""Rating prediction model: cross-domain training with OT alignment + target-only ablation."""

import os
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.model.gmm import GMMWrapper
from src.model.components import (
    weightLearner, ratingPredictor, compute_weighted_neg_mahalanobis,
    InteractionDataset, CrossDomainValidDataset, SingleDomainValidDataset,
    to_device,
)
from src.transport.ot_plan import compute_mmd
from src.utils import set_seed, ensure_dir


# ============================================================
# Combined model
# ============================================================

class RatingPredictionModel(nn.Module):
    """GMM-based rating predictor: weightLearner + ratingPredictor + GMMWrapper."""

    def __init__(self, d: int, sklearn_gmm, trainable_gmm: bool = True):
        super().__init__()
        K = sklearn_gmm.n_components
        self.gmm = GMMWrapper(sklearn_gmm, trainable=trainable_gmm)
        self.weight_learner = weightLearner(d, K)
        self.rating_predictor = ratingPredictor(K)

    def forward(self, user_emb, item_emb, t):
        w = self.weight_learner(user_emb)
        scores = compute_weighted_neg_mahalanobis(self.gmm, w, item_emb)
        return self.rating_predictor(scores, t)


# ============================================================
# Optimizer factory (respects GMM trainable/frozen setting)
# ============================================================

def _make_optimizer(model, lr, gmm_lr_scale, weight_decay):
    """Create optimizer with optional lower LR for GMM params."""
    if model.gmm.trainable:
        gmm_params = list(model.gmm.parameters())
        other_params = [p for n, p in model.named_parameters()
                        if not n.startswith('gmm.')]
        return torch.optim.Adam([
            {'params': other_params, 'lr': lr},
            {'params': gmm_params, 'lr': lr * gmm_lr_scale},
        ], weight_decay=weight_decay)
    else:
        # GMM is frozen (buffers) — only train other params
        trainable = [p for p in model.parameters() if p.requires_grad]
        return torch.optim.Adam(trainable, lr=lr, weight_decay=weight_decay)


# ============================================================
# Single-run cross-domain training
# ============================================================

def _train_one_run(model_src, model_tgt, src_train_ds, tgt_train_ds, tgt_valid_ds,
                   T, cfg_train, device):
    bs = cfg_train["batch_size"]
    lr = cfg_train["learning_rate"]
    wd = cfg_train.get("weight_decay", 1e-4)
    gmm_lr_scale = cfg_train.get("gmm_lr_scale", 0.1)
    lambda_align = cfg_train.get("lambda_align", 0.1)
    num_epochs = cfg_train["num_epochs"]
    patience = cfg_train.get("patience", 7)

    src_loader = DataLoader(src_train_ds, batch_size=bs, shuffle=True, num_workers=0, pin_memory=True)
    tgt_loader = DataLoader(tgt_train_ds, batch_size=bs, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(tgt_valid_ds, batch_size=bs, shuffle=False, num_workers=0, pin_memory=True)

    if cfg_train.get("loss_type", "huber") == "mse":
        criterion = nn.MSELoss()
    else:
        criterion = nn.HuberLoss(delta=cfg_train.get("huber_delta", 1.0))
    opt_src = _make_optimizer(model_src, lr, gmm_lr_scale, wd)
    opt_tgt = _make_optimizer(model_tgt, lr, gmm_lr_scale, wd)
    opt_align = torch.optim.Adam(
        list(model_src.weight_learner.parameters()) +
        list(model_tgt.weight_learner.parameters()),
        lr=lr * cfg_train.get("align_lr_scale", 0.1), weight_decay=wd)

    sched_src = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_src, 'min', 0.5, 2, min_lr=1e-5)
    sched_tgt = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_tgt, 'min', 0.5, 2, min_lr=1e-5)

    T_dev = torch.tensor(T, dtype=torch.float32).to(device) if not torch.is_tensor(T) else T.to(device)

    best_val = float('inf')
    es_counter = 0
    best_src_state = copy.deepcopy(model_src.state_dict())
    best_tgt_state = copy.deepcopy(model_tgt.state_dict())

    for epoch in range(num_epochs):
        # Phase 1: Source
        model_src.train()
        loss_s = 0.0
        for batch in src_loader:
            u, i, r, t = to_device(batch, device)
            opt_src.zero_grad()
            pred = model_src(u, i, t.unsqueeze(-1))
            loss = criterion(pred, r)
            loss.backward(); opt_src.step()
            loss_s += loss.item()
        print(f"  Epoch [{epoch+1}/{num_epochs}] Src Loss: {loss_s/max(len(src_loader),1):.4f}")

        # Phase 2: Target
        model_tgt.train()
        loss_t = 0.0
        for batch in tgt_loader:
            u, i, r, t = to_device(batch, device)
            opt_tgt.zero_grad()
            pred = model_tgt(u, i, t.unsqueeze(-1))
            loss = criterion(pred, r)
            loss.backward(); opt_tgt.step()
            loss_t += loss.item()
        print(f"  Epoch [{epoch+1}/{num_epochs}] Tgt Loss: {loss_t/max(len(tgt_loader),1):.4f}")

        # Phase 3: OT alignment
        if lambda_align > 0:
            model_src.train(); model_tgt.train()
            align_total, n_steps = 0.0, 0
            s_iter = iter(DataLoader(src_train_ds, batch_size=bs, shuffle=True, num_workers=0))
            t_iter = iter(DataLoader(tgt_train_ds, batch_size=bs, shuffle=True, num_workers=0))
            for sb, tb in zip(s_iter, t_iter):
                su = sb[0].to(device); tu = tb[0].to(device)
                opt_align.zero_grad()
                ws = model_src.weight_learner(su)
                wt = model_tgt.weight_learner(tu)
                transported = torch.softmax(torch.matmul(ws, T_dev), dim=-1)
                al = lambda_align * compute_mmd(transported, wt.detach())
                al.backward(); opt_align.step()
                align_total += al.item(); n_steps += 1
            if n_steps > 0:
                print(f"  Epoch [{epoch+1}/{num_epochs}] Align: {align_total/n_steps:.6f}")

        # Validation (fusion path)
        model_src.eval(); model_tgt.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                b = to_device(batch, device)
                ws = model_src.weight_learner(b["u_src"])
                wt = model_tgt.weight_learner(b["u_tgt"])
                alpha = (b["m_src"] / (b["m_src"] + b["m_tgt"] + 1e-8)).unsqueeze(1)
                combined = torch.softmax(alpha * torch.matmul(ws, T_dev) + (1 - alpha) * wt, dim=-1)
                scores = compute_weighted_neg_mahalanobis(model_tgt.gmm, combined, b["i_tgt"])
                pred = model_tgt.rating_predictor(scores, b["time"].unsqueeze(-1))
                val_loss += criterion(pred, b["rating"]).item()
        cur_val = val_loss / max(len(val_loader), 1)
        print(f"  Epoch [{epoch+1}/{num_epochs}] Val Loss: {cur_val:.4f}")

        sched_src.step(cur_val); sched_tgt.step(cur_val)
        if cur_val < best_val:
            best_val = cur_val
            best_src_state = copy.deepcopy(model_src.state_dict())
            best_tgt_state = copy.deepcopy(model_tgt.state_dict())
            es_counter = 0
        else:
            es_counter += 1
            if es_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}. Best: {best_val:.4f}")
                break

    return best_src_state, best_tgt_state


# ============================================================
# Multi-seed cross-domain training
# ============================================================

def train_rating_model(src_interactions, tgt_interactions, val_interactions,
                       u_emb_src, i_emb_src, u_emb_tgt, i_emb_tgt,
                       gmm_src, gmm_tgt, T, pair_name,
                       cfg_train, cfg_gmm, seeds, device):
    """Train cross-domain rating models across multiple seeds."""
    trainable_gmm = cfg_gmm.get("trainable", True)
    cfg_train = {**cfg_train, "gmm_lr_scale": cfg_gmm.get("gmm_lr_scale", 0.1)}

    src_ds = InteractionDataset(src_interactions, {**u_emb_src, **i_emb_src})
    tgt_ds = InteractionDataset(tgt_interactions, {**u_emb_tgt, **i_emb_tgt})
    val_ds = CrossDomainValidDataset(val_interactions, u_emb_tgt, i_emb_tgt, u_emb_src)

    stored = f"stored/{pair_name}"
    ensure_dir(stored)
    src_paths = [f"{stored}/model_source_seed{sd}.pth" for sd in seeds]
    tgt_paths = [f"{stored}/model_target_seed{sd}.pth" for sd in seeds]

    if all(os.path.exists(p) for p in src_paths + tgt_paths):
        print("Loading pre-trained rating models...")
        src_states = [torch.load(p, map_location=device) for p in src_paths]
        tgt_states = [torch.load(p, map_location=device) for p in tgt_paths]
        return src_states, tgt_states

    print("Training cross-domain rating models...")
    src_states, tgt_states = [], []
    d_src = list(u_emb_src.values())[0].shape[0]
    d_tgt = list(u_emb_tgt.values())[0].shape[0]

    for sd in seeds:
        print(f"\n===== Seed {sd} =====")
        set_seed(sd)
        m_src = RatingPredictionModel(d_src, gmm_src, trainable_gmm).to(device)
        m_tgt = RatingPredictionModel(d_tgt, gmm_tgt, trainable_gmm).to(device)
        ss, ts = _train_one_run(m_src, m_tgt, src_ds, tgt_ds, val_ds, T, cfg_train, device)
        src_states.append(ss); tgt_states.append(ts)

    for ss, sp in zip(src_states, src_paths):
        torch.save(ss, sp)
    for ts, tp in zip(tgt_states, tgt_paths):
        torch.save(ts, tp)
    return src_states, tgt_states


# ============================================================
# Target-only training (ablation)
# ============================================================

def _train_target_only_one_run(model_tgt, tgt_ds, val_ds, cfg_train, cfg_gmm, device):
    bs = cfg_train["batch_size"]
    lr = cfg_train["learning_rate"]
    wd = cfg_train.get("weight_decay", 1e-4)
    patience = cfg_train.get("patience_ablation", 5)

    loader = DataLoader(tgt_ds, batch_size=bs, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=0, pin_memory=True)
    if cfg_train.get("loss_type", "huber") == "mse":
        criterion = nn.MSELoss()
    else:
        criterion = nn.HuberLoss(delta=cfg_train.get("huber_delta", 1.0))
    opt = _make_optimizer(model_tgt, lr, cfg_gmm.get("gmm_lr_scale", 0.1), wd)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', 0.5, 2, min_lr=1e-5)

    best_val, es, best_state = float('inf'), 0, copy.deepcopy(model_tgt.state_dict())
    for epoch in range(cfg_train["num_epochs"]):
        model_tgt.train()
        tl = 0.0
        for batch in loader:
            u, i, r, t = to_device(batch, device)
            opt.zero_grad()
            pred = model_tgt(u, i, t.unsqueeze(-1))
            loss = criterion(pred, r); loss.backward(); opt.step()
            tl += loss.item()
        print(f"  Epoch [{epoch+1}] TgtOnly Train: {tl/len(loader):.4f}")

        model_tgt.eval()
        vl = 0.0
        with torch.no_grad():
            for batch in val_loader:
                b = to_device(batch, device)
                pred = model_tgt(b["u"], b["i"], b["time"].unsqueeze(-1))
                vl += criterion(pred, b["rating"]).item()
        cv = vl / len(val_loader)
        print(f"  Epoch [{epoch+1}] TgtOnly Val: {cv:.4f}")
        sched.step(cv)
        if cv < best_val:
            best_val = cv; best_state = copy.deepcopy(model_tgt.state_dict()); es = 0
        else:
            es += 1
            if es >= patience:
                print(f"  Early stopping. Best: {best_val:.4f}"); break
    return best_state


def train_target_only_rating(tgt_interactions, val_interactions,
                             u_emb_tgt, i_emb_tgt, gmm_tgt, pair_name,
                             cfg_train, cfg_gmm, seeds, device):
    """Multi-seed target-only training for ablation."""
    trainable = cfg_gmm.get("trainable", True)
    tgt_ds = InteractionDataset(tgt_interactions, {**u_emb_tgt, **i_emb_tgt})
    val_ds = SingleDomainValidDataset(val_interactions, u_emb_tgt, i_emb_tgt)

    stored = f"stored/{pair_name}"
    ensure_dir(stored)
    paths = [f"{stored}/model_tgtonly_seed{sd}.pth" for sd in seeds]

    if all(os.path.exists(p) for p in paths):
        print("Loading pre-trained target-only models...")
        return [torch.load(p, map_location=device) for p in paths]

    print("Training target-only rating models...")
    d_tgt = list(u_emb_tgt.values())[0].shape[0]
    states = []
    for sd in seeds:
        print(f"\n===== Seed {sd} (target-only) =====")
        set_seed(sd)
        m = RatingPredictionModel(d_tgt, gmm_tgt, trainable).to(device)
        s = _train_target_only_one_run(m, tgt_ds, val_ds, cfg_train, cfg_gmm, device)
        states.append(s)
    for s, p in zip(states, paths):
        torch.save(s, p)
    return states
