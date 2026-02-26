"""Shared model components: weightLearner, ratingPredictor, datasets, utilities."""

import hashlib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


# ============================================================
# Core NN components
# ============================================================

def compute_weighted_neg_mahalanobis(gmm, weights, item_embeddings):
    """
    Weighted negative Mahalanobis distance squared.

    Args:
        gmm: GMMWrapper with .means_t [K,D] and .prec_t [K,D].
        weights: [B, K] user GMM weights.
        item_embeddings: [B, D].
    Returns:
        [B, K] weighted negative Mahalanobis distance squared.
    """
    x = item_embeddings.unsqueeze(1)     # (B, 1, D)
    mu = gmm.means_t.unsqueeze(0)        # (1, K, D)
    p = gmm.prec_t.unsqueeze(0)          # (1, K, D)
    md2 = torch.sum((x - mu) ** 2 * p, dim=-1)  # (B, K)
    return weights * (-md2)              # (B, K)


class weightLearner(nn.Module):
    """MLP: user embedding → GMM component weights (softmax-normalized)."""
    def __init__(self, d: int, k: int, dropout: float = 0.2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(d, d // 2), nn.ReLU(), nn.Dropout(p=dropout),
            nn.Linear(d // 2, d // 4), nn.ReLU(), nn.Dropout(p=dropout),
            nn.Linear(d // 4, k),
        )

    def forward(self, x):
        return torch.softmax(self.model(x), dim=-1)


class ratingPredictor(nn.Module):
    """MLP: weighted Mahalanobis scores + time → predicted rating [1, 5]."""
    def __init__(self, K: int, dropout: float = 0.2):
        super().__init__()
        self.model = nn.Sequential(
            nn.BatchNorm1d(K + 1),
            nn.Linear(K + 1, (K + 1) // 2), nn.ReLU(), nn.Dropout(p=dropout),
            nn.Linear((K + 1) // 2, 1),
        )

    def forward(self, x, t):
        out = self.model(torch.cat([x, t], dim=-1))
        return (1 + 4 * torch.sigmoid(out)).squeeze(-1)


# ============================================================
# Embedding fallback for unseen IDs
# ============================================================

def estimate_scale(train_vecs: np.ndarray) -> dict:
    norms = np.linalg.norm(train_vecs, axis=1)
    avg_norm = float(np.mean(norms)) if norms.size else 1.0
    d = train_vecs.shape[1] if train_vecs.ndim == 2 else 32
    std = avg_norm / max(np.sqrt(d), 1.0)
    return {"std": std, "avg_norm": avg_norm, "d": d}


def rand_embed_for_id(key: str, d: int, std: float,
                      unit_norm: bool = False) -> np.ndarray:
    """Deterministic random vector for a given ID (reproducible)."""
    seed = int(hashlib.md5(key.encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.RandomState(seed)
    v = rng.normal(loc=0.0, scale=std, size=(d,)).astype(np.float32)
    if unit_norm:
        n = np.linalg.norm(v)
        if n > 0:
            v = v / n
    return v


# ============================================================
# Helper: build user-positive-items dict
# ============================================================

def build_user_pos_items(interaction_list):
    """Build {user_id: set(item_ids)} from interaction list.

    interaction_list: [(user_id, item_id, rating, time), ...]
    """
    user_pos = {}
    for u, i, *_ in interaction_list:
        user_pos.setdefault(u, set()).add(i)
    return user_pos


def collect_all_items(interaction_list):
    """Collect the set of all item IDs from an interaction list."""
    return list({i for _, i, *_ in interaction_list})


# ============================================================
# Datasets
# ============================================================

class InteractionDataset(Dataset):
    """Training dataset: (user_id, item_id, rating, time) tuples."""
    def __init__(self, tuple_list, embedding_dict):
        self.tuple_list = tuple_list
        self.embedding_dict = embedding_dict

    def __len__(self):
        return len(self.tuple_list)

    def __getitem__(self, idx):
        u, i, r, t = self.tuple_list[idx]
        return (torch.tensor(self.embedding_dict[u], dtype=torch.float32),
                torch.tensor(self.embedding_dict[i], dtype=torch.float32),
                torch.tensor(r, dtype=torch.float32),
                torch.tensor(t, dtype=torch.float32))


class BPRDataset(Dataset):
    """Standard BPR training dataset with per-epoch dynamic negative sampling.

    For each positive interaction (user, item), a random uninteracted item
    is sampled as the negative at access time, so negatives differ every epoch.

    Parameters
    ----------
    interaction_list : list of (user_id, item_id, rating, time)
        Positive interactions (rating is unused for ranking but kept for compat).
    embedding_dict : dict
        {id: np.ndarray} for both users and items.
    all_items : list
        List of all candidate item IDs.
    user_pos_items : dict
        {user_id: set(item_ids)} — items each user has interacted with.
    """
    def __init__(self, interaction_list, embedding_dict, all_items, user_pos_items):
        self.interactions = interaction_list
        self.embedding_dict = embedding_dict
        self.all_items = list(all_items)
        self.user_pos_items = user_pos_items

    def __len__(self):
        return len(self.interactions)

    def __getitem__(self, idx):
        uid, pos_iid, _rating, t = self.interactions[idx]

        # Dynamic negative sampling: uniformly sample an uninteracted item
        pos_set = self.user_pos_items.get(uid, set())
        neg_iid = self.all_items[np.random.randint(len(self.all_items))]
        while neg_iid in pos_set:
            neg_iid = self.all_items[np.random.randint(len(self.all_items))]

        return (torch.tensor(self.embedding_dict[uid], dtype=torch.float32),
                torch.tensor(self.embedding_dict[pos_iid], dtype=torch.float32),
                torch.tensor(self.embedding_dict[neg_iid], dtype=torch.float32),
                torch.tensor(t, dtype=torch.float32),
                torch.tensor(t, dtype=torch.float32))  # neg reuses pos time


class CrossDomainValidDataset(Dataset):
    """Validation/test dataset with source+target user embeddings and fallback."""
    def __init__(self, tuple_list, user_tgt_dict, item_tgt_dict, user_src_dict,
                 unit_norm_fallback=False):
        self.tuple_list = tuple_list
        self.user_tgt = {k: np.asarray(v, dtype=np.float32) for k, v in user_tgt_dict.items()}
        self.item_tgt = {k: np.asarray(v, dtype=np.float32) for k, v in item_tgt_dict.items()}
        self.user_src = {k: np.asarray(v, dtype=np.float32) for k, v in user_src_dict.items()}

        u_tgt_mat = np.stack(list(self.user_tgt.values()), axis=0) if self.user_tgt else np.zeros((0, 32), dtype=np.float32)
        d_base = u_tgt_mat.shape[1] if u_tgt_mat.size else 32
        u_src_mat = np.stack(list(self.user_src.values()), axis=0) if self.user_src else np.zeros((0, d_base), dtype=np.float32)
        i_tgt_mat = np.stack(list(self.item_tgt.values()), axis=0) if self.item_tgt else np.zeros((0, d_base), dtype=np.float32)

        self.scale_u_tgt = estimate_scale(u_tgt_mat)
        self.scale_u_src = estimate_scale(u_src_mat)
        self.scale_i_tgt = estimate_scale(i_tgt_mat)
        self.unit_norm = unit_norm_fallback
        self._cache_u_tgt, self._cache_u_src, self._cache_i_tgt = {}, {}, {}

    def _get(self, id_, pool, cache, prefix, scale):
        if id_ in pool:
            return pool[id_], 1.0
        s = str(id_)
        if s not in cache:
            cache[s] = rand_embed_for_id(f"{prefix}::{s}", scale["d"],
                                         scale["std"], self.unit_norm)
        return cache[s], 0.0

    def __len__(self):
        return len(self.tuple_list)

    def __getitem__(self, idx):
        u, i, r, ts = self.tuple_list[idx]
        u_tgt, m_tgt = self._get(u, self.user_tgt, self._cache_u_tgt, "U_TGT", self.scale_u_tgt)
        u_src, m_src = self._get(u, self.user_src, self._cache_u_src, "U_SRC", self.scale_u_src)
        v_tgt, _ = self._get(i, self.item_tgt, self._cache_i_tgt, "I_TGT", self.scale_i_tgt)
        return {
            "u_src": torch.from_numpy(u_src), "u_tgt": torch.from_numpy(u_tgt),
            "m_src": torch.tensor(m_src, dtype=torch.float32),
            "m_tgt": torch.tensor(m_tgt, dtype=torch.float32),
            "i_tgt": torch.from_numpy(v_tgt),
            "rating": torch.tensor(r, dtype=torch.float32),
            "time": torch.tensor(ts, dtype=torch.float32),
        }


class CrossDomainBPRValidDataset(Dataset):
    """Validation for BPR ranking with dynamic negative sampling.

    Parameters
    ----------
    interaction_list : list of (user_id, item_id, rating, time)
        Positive validation interactions.
    user_tgt_dict, item_tgt_dict, user_src_dict : dict
        Embedding dicts.
    all_items : list
        All candidate item IDs.
    user_pos_items : dict
        {user_id: set(item_ids)} — union of train+val positives for proper
        negative sampling (ensures negatives are truly uninteracted).
    """
    def __init__(self, interaction_list, user_tgt_dict, item_tgt_dict, user_src_dict,
                 all_items, user_pos_items, unit_norm_fallback=False):
        self.interactions = interaction_list
        self.all_items = list(all_items)
        self.user_pos_items = user_pos_items

        self.user_tgt = {k: np.asarray(v, dtype=np.float32) for k, v in user_tgt_dict.items()}
        self.item_tgt = {k: np.asarray(v, dtype=np.float32) for k, v in item_tgt_dict.items()}
        self.user_src = {k: np.asarray(v, dtype=np.float32) for k, v in user_src_dict.items()}

        u_tgt_mat = np.stack(list(self.user_tgt.values()), axis=0) if self.user_tgt else np.zeros((0, 32), dtype=np.float32)
        d_base = u_tgt_mat.shape[1] if u_tgt_mat.size else 32
        u_src_mat = np.stack(list(self.user_src.values()), axis=0) if self.user_src else np.zeros((0, d_base), dtype=np.float32)
        i_tgt_mat = np.stack(list(self.item_tgt.values()), axis=0) if self.item_tgt else np.zeros((0, d_base), dtype=np.float32)

        self.scale_u_tgt = estimate_scale(u_tgt_mat)
        self.scale_u_src = estimate_scale(u_src_mat)
        self.scale_i_tgt = estimate_scale(i_tgt_mat)
        self.unit_norm = unit_norm_fallback
        self._cache_u_tgt, self._cache_u_src, self._cache_i_tgt = {}, {}, {}

    def _get(self, id_, pool, cache, prefix, scale):
        if id_ in pool:
            return pool[id_], 1.0
        s = str(id_)
        if s not in cache:
            cache[s] = rand_embed_for_id(f"{prefix}::{s}", scale["d"],
                                         scale["std"], self.unit_norm)
        return cache[s], 0.0

    def _get_item(self, id_):
        v, _ = self._get(id_, self.item_tgt, self._cache_i_tgt, "I_TGT", self.scale_i_tgt)
        return v

    def __len__(self):
        return len(self.interactions)

    def __getitem__(self, idx):
        uid, pos_iid, _rating, t = self.interactions[idx]

        # Dynamic negative sampling
        pos_set = self.user_pos_items.get(uid, set())
        neg_iid = self.all_items[np.random.randint(len(self.all_items))]
        while neg_iid in pos_set:
            neg_iid = self.all_items[np.random.randint(len(self.all_items))]

        u_tgt, m_tgt = self._get(uid, self.user_tgt, self._cache_u_tgt, "U_TGT", self.scale_u_tgt)
        u_src, m_src = self._get(uid, self.user_src, self._cache_u_src, "U_SRC", self.scale_u_src)
        return {
            "u_src": torch.from_numpy(u_src), "u_tgt": torch.from_numpy(u_tgt),
            "m_src": torch.tensor(m_src, dtype=torch.float32),
            "m_tgt": torch.tensor(m_tgt, dtype=torch.float32),
            "pos_i_tgt": torch.from_numpy(self._get_item(pos_iid)),
            "neg_i_tgt": torch.from_numpy(self._get_item(neg_iid)),
            "pos_time": torch.tensor(t, dtype=torch.float32),
            "neg_time": torch.tensor(t, dtype=torch.float32),
        }


class SingleDomainValidDataset(Dataset):
    """Validation/test for target-only models."""
    def __init__(self, tuple_list, user_dict, item_dict):
        self.tuple_list = tuple_list
        self.user = {k: np.asarray(v, dtype=np.float32) for k, v in user_dict.items()}
        self.item = {k: np.asarray(v, dtype=np.float32) for k, v in item_dict.items()}

        u_mat = np.stack(list(self.user.values()), axis=0) if self.user else np.zeros((0, 32), dtype=np.float32)
        i_mat = np.stack(list(self.item.values()), axis=0) if self.item else np.zeros((0, 32), dtype=np.float32)
        self.u_scale = estimate_scale(u_mat)
        self.i_scale = estimate_scale(i_mat)
        self._u_cache, self._i_cache = {}, {}

    def __len__(self):
        return len(self.tuple_list)

    def __getitem__(self, idx):
        u, i, r, ts = self.tuple_list[idx]

        def _get(id_, pool, cache, prefix, scale):
            if id_ in pool:
                return pool[id_]
            if id_ not in cache:
                cache[id_] = rand_embed_for_id(f"{prefix}::{id_}", scale["d"], scale["std"])
            return cache[id_]

        return {
            "u": torch.from_numpy(_get(u, self.user, self._u_cache, "U", self.u_scale)),
            "i": torch.from_numpy(_get(i, self.item, self._i_cache, "I", self.i_scale)),
            "rating": torch.tensor(r, dtype=torch.float32),
            "time": torch.tensor(ts, dtype=torch.float32),
        }


# ============================================================
# Helper: batch to device
# ============================================================

@torch.no_grad()
def to_device(batch, device):
    if isinstance(batch, dict):
        return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        return tuple(x.to(device) if torch.is_tensor(x) else x for x in batch)
    return batch
