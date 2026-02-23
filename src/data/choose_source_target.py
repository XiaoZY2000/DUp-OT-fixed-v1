"""Determine feasible source-target pairing based on time order."""

import numpy as np


def judge_domain_time_order(domain_a: dict, domain_b: dict,
                            source_grid=None) -> dict:
    """
    Determine feasible source-target pairing based on time order.

    - Target uses last 20% as test: test_start_target = times_tgt[floor(0.8*n)]
    - Source keeps a prefix (keep_ratio in (0,1)), searching from large to small
    - Tries both A→B and B→A directions

    Returns dict with A_as_source, B_as_source, and recommendation.
    """

    def collect_times(d):
        arr = []
        for u in d.values():
            for it in u.values():
                t = it.get("time")
                if t is not None:
                    arr.append(float(t))
        return np.sort(np.asarray(arr, dtype=float)) if arr else np.array([], dtype=float)

    def stats(times):
        return {
            "n": int(len(times)),
            "min": float(times[0]) if len(times) else None,
            "max": float(times[-1]) if len(times) else None,
        }

    def target_test_start_last20(times):
        n = len(times)
        if n < 2:
            return None
        k = max(1, min(int(np.floor(0.8 * n)), n - 1))
        return float(times[k])

    def source_train_end_keep_prefix(times, keep_ratio):
        n = len(times)
        if n < 2:
            return None
        k = max(1, min(int(np.floor(keep_ratio * n)), n - 1))
        return float(times[k - 1])

    if source_grid is None:
        source_grid = [i / 100 for i in range(5, 100, 5)]
    source_grid = sorted([r for r in source_grid if 0.0 < r < 1.0], reverse=True)

    A, B = collect_times(domain_a), collect_times(domain_b)
    A_stats, B_stats = stats(A), stats(B)

    def solve(times_src, times_tgt):
        if len(times_src) < 2 or len(times_tgt) < 2:
            return {"feasible": False, "chosen_keep_ratio": None,
                    "train_end_source": None, "test_start_target": None, "gap": None}
        te_tgt = target_test_start_last20(times_tgt)
        if te_tgt is None:
            return {"feasible": False, "chosen_keep_ratio": None,
                    "train_end_source": None, "test_start_target": None, "gap": None}
        for r in source_grid:
            te_src = source_train_end_keep_prefix(times_src, r)
            if te_src is not None and te_tgt > te_src:
                return {"feasible": True, "chosen_keep_ratio": float(r),
                        "train_end_source": float(te_src),
                        "test_start_target": float(te_tgt),
                        "gap": float(te_tgt - te_src)}
        return {"feasible": False, "chosen_keep_ratio": None,
                "train_end_source": None, "test_start_target": float(te_tgt), "gap": None}

    A_as_source = solve(A, B)
    A_as_source.update({"A_stats": A_stats, "B_stats": B_stats})
    B_as_source = solve(B, A)
    B_as_source.update({"A_stats": A_stats, "B_stats": B_stats})

    # Pick recommendation
    if not A_as_source["feasible"] and not B_as_source["feasible"]:
        rec = "none"
    elif A_as_source["feasible"] and not B_as_source["feasible"]:
        rec = "A_as_source"
    elif B_as_source["feasible"] and not A_as_source["feasible"]:
        rec = "B_as_source"
    else:
        k1 = (A_as_source["gap"], A_as_source["chosen_keep_ratio"])
        k2 = (B_as_source["gap"], B_as_source["chosen_keep_ratio"])
        rec = "A_as_source" if k1 >= k2 else "B_as_source"

    return {"A_as_source": A_as_source, "B_as_source": B_as_source,
            "recommendation": rec}
