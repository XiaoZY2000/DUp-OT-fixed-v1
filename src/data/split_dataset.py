"""Time-based dataset splitting (global or per-user chronological)."""

from typing import Dict, Tuple, List
from math import floor
from collections import defaultdict

Record = Dict[str, object]
Nested = Dict[str, Dict[str, Dict[str, object]]]


def flatten_nested(nested: Nested) -> List[Record]:
    """Flatten nested dict to list of records with user_id / item_id."""
    out = []
    for u, items in nested.items():
        for i, payload in items.items():
            r = dict(payload)
            r["user_id"] = u
            r["item_id"] = i
            out.append(r)
    return out


def _to_nested(records: List[Record]) -> Nested:
    """Reconstruct nested dict from list of records."""
    nested: Nested = {}
    for r in records:
        u, i = r["user_id"], r["item_id"]
        payload = {k: v for k, v in r.items() if k not in ("user_id", "item_id")}
        nested.setdefault(u, {})[i] = payload
    return nested


def _chronological_split_global(records: List[Record],
                                ratios: Tuple[float, ...]) -> List[List[Record]]:
    """Global chronological split by unix timestamp."""
    recs = sorted(records, key=lambda x: x["time"])
    n = len(recs)
    if n == 0:
        return [[] for _ in ratios]
    cuts, acc = [], 0.0
    for j, r in enumerate(ratios):
        if j == len(ratios) - 1:
            cuts.append(n)
        else:
            acc += r
            cuts.append(floor(acc * n))
    splits, start = [], 0
    for c in cuts:
        splits.append(recs[start:c])
        start = c
    # Min-sample protection
    for idx in range(len(splits)):
        if len(splits[idx]) == 0 and idx > 0 and len(splits[idx - 1]) > 1:
            splits[idx].append(splits[idx - 1].pop())
    return splits


def _chronological_split_per_user(records: List[Record],
                                  ratios: Tuple[float, ...]) -> List[List[Record]]:
    """Per-user chronological split, then merge."""
    by_user = defaultdict(list)
    for r in records:
        by_user[r["user_id"]].append(r)
    K = len(ratios)
    buckets: List[List[Record]] = [[] for _ in range(K)]
    for u, recs in by_user.items():
        recs = sorted(recs, key=lambda x: x["time"])
        n = len(recs)
        if n == 0:
            continue
        cuts, acc = [], 0.0
        for j, r in enumerate(ratios):
            if j == K - 1:
                cuts.append(n)
            else:
                acc += r
                cuts.append(floor(acc * n) + 1)
        start = 0
        temp = []
        for c in cuts:
            temp.append(recs[start:c])
            start = c
        for idx in range(K):
            if len(temp[idx]) == 0 and idx > 0 and len(temp[idx - 1]) > 1:
                temp[idx].append(temp[idx - 1].pop())
        for idx in range(K):
            buckets[idx].extend(temp[idx])
    for idx in range(K):
        buckets[idx].sort(key=lambda x: x["time"])
    return buckets


def time_split(data: Nested, ratios: tuple, by_user: bool = False) -> list:
    """
    Split data by time. Returns list of nested dicts.

    Parameters
    ----------
    data : Nested
        {user: {item: {review, rating, time}}}.
    ratios : tuple
        E.g. (0.8, 0.1, 0.1) for train/val/test.
    by_user : bool
        If True, split per-user then merge. If False, global chronological.
    """
    recs = flatten_nested(data)
    if by_user:
        splits = _chronological_split_per_user(recs, ratios)
    else:
        splits = _chronological_split_global(recs, ratios)
    return [_to_nested(s) for s in splits]
