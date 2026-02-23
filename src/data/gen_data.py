"""Convert filtered JSONL to nested interaction dict {user: {item: {review, rating, time}}}."""

import os
import re
import html
import json
import collections
import pandas as pd
from tqdm import tqdm


def clean_text(text: str) -> str:
    text = html.unescape(text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\|', '', text)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r"['\"]", '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def gen_data(input_path: str, output_path: str, cfg_gen: dict):
    """
    Read filtered JSONL, build interaction dict, save as JSON.

    Parameters
    ----------
    input_path : str
        Filtered JSONL path.
    output_path : str
        Output JSON path.
    cfg_gen : dict
        Keys: user_col, item_col, rating_col, review_col, time_col.
    """
    user_col = cfg_gen["user_col"]
    item_col = cfg_gen["item_col"]
    rating_col = cfg_gen["rating_col"]
    review_col = cfg_gen["review_col"]
    time_col = cfg_gen["time_col"]

    df = pd.read_json(input_path, lines=True)
    print(f"Loaded {len(df)} records from {input_path}")

    grouped = (
        df.groupby(user_col)
        .apply(lambda g: g.drop(columns=user_col).to_dict(orient='records'))
        .to_dict()
    )

    interaction_dict = collections.defaultdict(dict)
    for user_id, reviews in tqdm(grouped.items(), desc="Building interactions"):
        for review in reviews:
            iid = review.get(item_col)
            if iid is None:
                continue
            r = review.get(rating_col)
            rv = review.get(review_col)
            t = review.get(time_col)
            if not (r and rv and t and isinstance(rv, str)):
                continue
            interaction_dict[user_id][iid] = {
                'review': clean_text(rv),
                'rating': float(r),
                'time': float(t),
            }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(dict(interaction_dict), f)
    print(f"Saved {len(interaction_dict)} users to {output_path}")
