import os
import json
import pickle
from typing import Any


def ensure_dir(path: str):
    """Create directory (and parents) if it does not exist."""
    os.makedirs(path, exist_ok=True)


def load_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def save_json(obj: Any, path: str):
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    with open(path, "w") as f:
        json.dump(obj, f)


def load_pickle(path: str) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pickle(obj: Any, path: str):
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    with open(path, "wb") as f:
        pickle.dump(obj, f)
