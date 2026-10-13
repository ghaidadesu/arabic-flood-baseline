"""
Load, clean and split the Arabic Flood corpus (Alharbi & Lee, 2019).

The corpus is NOT stored in this repository. It is downloaded from the
authors' public repository (or read from a local copy you give with --csv).

Run `python data_utils.py` to check the numbers and write the ID lists to splits/.
"""

import argparse
import ast
import os
import re
import urllib.request

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_URL = ("https://raw.githubusercontent.com/alaa-a-a/"
            "Arabic-Twitter-corpus-for-flood-detection/master/ArabicFloods.csv")
LOCAL_COPY = os.path.join("data", "ArabicFloods.csv")   # data/ is in .gitignore

SPLIT_SEED = 42
EVAL_SIZE = 805      # evaluation set; everything else (3,223) is development


def load_raw(csv_path=None):
    """Read the original CSV (4,037 rows). Downloads it if no local copy exists."""
    path = csv_path or LOCAL_COPY
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        try:
            urllib.request.urlretrieve(DATA_URL, path)
        except Exception as error:
            raise RuntimeError(
                f"Could not download the dataset ({error}). Download ArabicFloods.csv "
                f"from {DATA_URL} and pass it with --csv, or save it as {LOCAL_COPY}."
            ) from error
    return pd.read_csv(path, encoding="utf-8-sig", dtype=str)


def clean_tweet_text(raw):
    """The 'Tweet Text' column is a Python list printed as text, e.g. ['the tweet\\nline 2'].
    Python's own parser turns it back into the real tweet (new lines, emoji, quotes).
    A few rows are malformed; for those we fall back to stripping the wrapper by hand."""
    text = str(raw).strip()
    candidates = [text]
    if text.endswith(']"'):          # some rows have one stray quote after the closing bracket
        candidates.append(text[:-1])
    for candidate in candidates:
        try:
            value = ast.literal_eval(candidate)
            if isinstance(value, (list, tuple)) and len(value) == 1:
                value = value[0]
            if isinstance(value, str):
                return value.strip()
        except (ValueError, SyntaxError):
            pass
    text = re.sub(r"^\[\s*['\"]?", "", text)
    text = re.sub(r"['\"]?\s*\]\s*['\"]?$", "", text)
    return text.replace("\\n", "\n").strip()


def clean_dataset(df):
    """Drop IDs whose duplicate rows have conflicting labels, keep one row per ID,
    and add a clean 'text' column. 4,037 rows -> 4,028 rows."""
    conflicting = (
        df.groupby("ID")[["Relevance", "Information Type"]]
        .nunique().gt(1).any(axis=1)
    )
    conflicting_ids = conflicting[conflicting].index
    clean = df[~df["ID"].isin(conflicting_ids)]
    clean = clean.drop_duplicates(subset=["ID"], keep="first").reset_index(drop=True)
    clean["text"] = clean["Tweet Text"].map(clean_tweet_text)
    clean["joint_label"] = clean["Relevance"] + " | " + clean["Information Type"]
    return clean


def split_dataset(clean, eval_size=EVAL_SIZE, seed=SPLIT_SEED):
    """Disjoint, stratified split. Returns (dev_df, eval_df) = (3,223 / 805 rows)."""
    dev_df, eval_df = train_test_split(
        clean, test_size=eval_size, random_state=seed, stratify=clean["joint_label"]
    )
    return dev_df.copy(), eval_df.copy()


def load_splits(csv_path=None):
    clean = clean_dataset(load_raw(csv_path))
    return split_dataset(clean)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", help="path to a local ArabicFloods.csv")
    args = parser.parse_args()

    raw = load_raw(args.csv)
    clean = clean_dataset(raw)
    dev, ev = split_dataset(clean)
    print("Original tweets      :", len(raw))
    print("After cleaning       :", len(clean))
    print("Development          :", len(dev))
    print("Evaluation           :", len(ev))
    print("Overlap (must be 0)  :", len(set(dev["ID"]) & set(ev["ID"])))

    os.makedirs("splits", exist_ok=True)
    dev[["ID"]].to_csv("splits/dev_ids.csv", index=False)
    ev[["ID"]].to_csv("splits/eval_ids.csv", index=False)
    print("Wrote splits/dev_ids.csv and splits/eval_ids.csv (IDs only, no tweet text).")
