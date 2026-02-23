"""Generate user/item embeddings from review text via sentence encoder + time-weighted pooling."""

import os
import json
import math
from typing import Dict, List, Tuple
from collections import defaultdict

import torch
from tqdm import tqdm
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer

from src.utils import ensure_dir


def _safe_sentences(text: str) -> List[str]:
    sents = sent_tokenize(text)
    return sents if sents else [text]


@torch.no_grad()
def _encode_texts(encoder, texts: List[str], device: torch.device,
                  sent_batch: int = 64) -> torch.Tensor:
    if not texts:
        return torch.empty(0, 0, device=device)
    all_embs = []
    for i in range(0, len(texts), sent_batch):
        chunk = texts[i:i + sent_batch]
        embs = encoder.encode(chunk, convert_to_tensor=True, device=device)
        all_embs.append(embs)
    return torch.cat(all_embs, dim=0)


def extract_user_item_reviews(nested_dict: dict
                              ) -> Tuple[Dict[str, List], Dict[str, List]]:
    """Convert nested interaction dict to user_reviews and item_reviews."""
    user_reviews: Dict[str, List] = {}
    item_reviews: Dict[str, List] = {}
    for user, items in nested_dict.items():
        u_revs = []
        for item, payload in items.items():
            if ("review" in payload and isinstance(payload["review"], str)
                    and "time" in payload):
                u_revs.append((payload["review"], float(payload["time"])))
                item_reviews.setdefault(item, []).append(
                    (payload["review"], float(payload["time"])))
        user_reviews[user] = sorted(u_revs, key=lambda x: x[1])
    for item in item_reviews:
        item_reviews[item].sort(key=lambda x: x[1])
    return user_reviews, item_reviews


def export_embeddings_batched(
    encoder, reviews: Dict[str, List[Tuple[str, float]]],
    sent_batch: int = 64, time_pooling: bool = True,
    device: torch.device = torch.device("cpu")
) -> Dict[str, torch.Tensor]:
    """Encode reviews and aggregate to per-entity embeddings."""

    # Flatten all sentences
    global_sents = []
    for id_, revs in reviews.items():
        for r_idx, (text, ts) in enumerate(revs):
            for s in _safe_sentences(text):
                global_sents.append((id_, r_idx, ts, s))
    if not global_sents:
        return {}

    # Encode all sentences
    texts = [x[3] for x in global_sents]
    all_embs = []
    for i in tqdm(range(0, len(texts), sent_batch), desc="Encoding sentences"):
        be = _encode_texts(encoder, texts[i:i + sent_batch],
                           device=device, sent_batch=sent_batch)
        if be.ndim == 1:
            be = be.unsqueeze(0)
        all_embs.append(be.cpu())
    all_embs = torch.cat(all_embs, dim=0)

    # Regroup per (ID, review)
    tmp = defaultdict(lambda: defaultdict(list))
    for (id_, r_idx, ts, _), e in zip(global_sents, all_embs):
        tmp[id_][r_idx].append((ts, e))

    # Review pooling + time pooling
    final = {}
    MONTH = 30 * 86400
    lambda_ = math.log(2) / 6.0

    for id_, rev_map in tqdm(tmp.items(), desc="Aggregating reviews"):
        review_embs, review_ts = [], []
        for r_idx, entries in rev_map.items():
            emb_stack = torch.stack([e for (_, e) in entries], dim=0)
            review_embs.append(emb_stack.mean(dim=0))
            review_ts.append(entries[0][0])

        review_embs = torch.stack(review_embs, dim=0)

        if time_pooling and review_embs.size(0) > 1:
            ts = torch.tensor(review_ts, dtype=torch.float32)
            t_ref = ts.max()
            delta = ((t_ref - ts) / MONTH).clamp(max=36.0)
            w = torch.exp(-lambda_ * delta)
            w = w / (w.sum() + 1e-9)
            agg = (w.unsqueeze(-1) * review_embs).sum(dim=0)
        else:
            agg = review_embs.mean(dim=0)

        final[id_] = agg.cpu()
    return final


def process_data(source_train, target_train, pair_name: str,
                 cfg_encoder: dict, device: torch.device):
    """
    Generate or load review-based user/item embeddings for both domains.

    Returns: (user_emb_source, item_emb_source, user_emb_target, item_emb_target)
    """
    stored = f"stored/{pair_name}"
    source_path = f"{stored}/review_based_embeddings_source.json"
    target_path = f"{stored}/review_based_embeddings_target.json"

    model_name = cfg_encoder.get("model_name", "all-MiniLM-L6-v2")
    sent_batch = cfg_encoder.get("sent_batch", 64)
    tp = cfg_encoder.get("time_pooling", True)

    encoder = SentenceTransformer(model_name, device=device)

    def _load_or_build(path, train_data, label):
        if os.path.exists(path):
            print(f"Loading {label} embeddings from {path}")
            raw = json.load(open(path, "r"))
            u_emb = {k: torch.tensor(v) for k, v in raw["user_embeddings"].items()}
            i_emb = {k: torch.tensor(v) for k, v in raw["item_embeddings"].items()}
            return u_emb, i_emb

        print(f"Generating {label} embeddings...")
        user_reviews, item_reviews = extract_user_item_reviews(train_data)

        u_emb = export_embeddings_batched(
            encoder, user_reviews, sent_batch=sent_batch,
            time_pooling=tp, device=device)
        i_emb = export_embeddings_batched(
            encoder, item_reviews, sent_batch=sent_batch,
            time_pooling=False, device=device)

        ensure_dir(stored)
        json_data = {
            "user_embeddings": {k: v.tolist() for k, v in u_emb.items()},
            "item_embeddings": {k: v.tolist() for k, v in i_emb.items()},
        }
        with open(path, "w") as f:
            json.dump(json_data, f)
        return u_emb, i_emb

    u_src, i_src = _load_or_build(source_path, source_train, "source")
    u_tgt, i_tgt = _load_or_build(target_path, target_train, "target")
    return u_src, i_src, u_tgt, i_tgt
