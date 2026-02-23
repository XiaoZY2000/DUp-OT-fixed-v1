"""K-core filtering for Amazon review datasets."""

import os
import sys
import pandas as pd


def _human_int(n: int) -> str:
    return f"{n:,}"


def _print_stats(df: pd.DataFrame, user_col: str, item_col: str, title: str):
    print(f"[{title}]")
    print(f"  Interactions: {_human_int(len(df))}")
    print(f"  Users:        {_human_int(df[user_col].nunique())}")
    print(f"  Items:        {_human_int(df[item_col].nunique())}")


def k_core_filter(df: pd.DataFrame, user_col: str, item_col: str,
                  k_user: int, k_item: int) -> pd.DataFrame:
    """Iteratively enforce user/item degree >= k until convergence."""
    prev_len = -1
    it = 0
    while True:
        it += 1
        before_len = len(df)
        user_counts = df[user_col].value_counts()
        item_counts = df[item_col].value_counts()
        if k_user > 0:
            df = df[df[user_col].isin(user_counts[user_counts >= k_user].index)]
        if k_item > 0:
            df = df[df[item_col].isin(item_counts[item_counts >= k_item].index)]
        after_len = len(df)
        print(f"  - Iter {it}: {_human_int(before_len)} -> {_human_int(after_len)} interactions")
        if after_len == before_len or after_len == prev_len:
            break
        prev_len = after_len
    return df


def filter_dataset(input_path: str, output_path: str, cfg_filter: dict):
    """
    Load raw JSONL, apply k-core filter, save filtered JSONL.

    Parameters
    ----------
    input_path : str
        Path to input .jsonl or .json.gz.
    output_path : str
        Path to output .jsonl or .json.gz.
    cfg_filter : dict
        Keys: user_col, item_col, k_user, k_item, n_rows.
    """
    user_col = cfg_filter["user_col"]
    item_col = cfg_filter["item_col"]
    k_user = cfg_filter["k_user"]
    k_item = cfg_filter["k_item"]
    n_rows = cfg_filter.get("n_rows")

    try:
        print(f"[Loading] {input_path}")
        df = pd.read_json(input_path, lines=True, nrows=n_rows)
    except ValueError:
        print("Read failed: ensure the file is JSON Lines.", file=sys.stderr)
        raise

    for col in (user_col, item_col):
        if col not in df.columns:
            raise KeyError(f"Missing column: {col}. Available: {list(df.columns)}")

    _print_stats(df, user_col, item_col, "Before Filter")
    print(f"[Filtering] k_user={k_user}, k_item={k_item}")
    df_f = k_core_filter(df, user_col, item_col, k_user, k_item)
    _print_stats(df_f, user_col, item_col, "After  Filter")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    df_f.to_json(output_path, orient="records", lines=True,
                 force_ascii=False, compression="infer")
    print(f"[Saved] {output_path}  ({_human_int(len(df_f))} lines)")
