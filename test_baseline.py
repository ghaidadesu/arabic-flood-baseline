"""
Baseline Testing: run the single-agent baseline on a few Development tweets,
compare with the ground truth, and record results, time and token usage.

    python test_baseline.py            # 5 tweets
    python test_baseline.py --n 10
"""

import argparse
import json
import os

import pandas as pd

import baseline
from data_utils import SPLIT_SEED, load_splits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5, help="number of Development tweets")
    parser.add_argument("--csv", help="path to a local ArabicFloods.csv")
    parser.add_argument("--out", default="results", help="folder for the results")
    args = parser.parse_args()

    dev_df, _ = load_splits(args.csv)
    sample = dev_df.sample(n=args.n, random_state=SPLIT_SEED)   # same tweets every run

    rows = []
    for number, (_, tweet) in enumerate(sample.iterrows(), start=1):
        result = baseline.classify_tweet(tweet["text"])
        correct = (result["valid"]
                   and result["relevance"] == tweet["Relevance"]
                   and result["information_type"] == tweet["Information Type"])
        rows.append({
            "ID": tweet["ID"], "Event": tweet["Event"],
            "gold_relevance": tweet["Relevance"],
            "gold_information_type": tweet["Information Type"],
            "pred_relevance": result["relevance"],
            "pred_information_type": result["information_type"],
            "joint_correct": bool(correct), "valid": result["valid"], "error": result["error"],
            "first_output_violation": result["first_output_violation"],
            "calls": result["calls"], "repairs": result["repairs"],
            "api_retries": result["api_retries"],
            "input_tokens": result["input_tokens"], "output_tokens": result["output_tokens"],
            "reasoning_tokens": result["reasoning_tokens"], "total_tokens": result["total_tokens"],
            "cost_usd": result["cost_usd"], "latency_s": round(result["latency_s"], 2),
        })
        print(f"\n[{number}/{args.n}] {tweet['text'][:90]!r}")
        print(f"   gold: {tweet['Relevance']} | {tweet['Information Type']}")
        print(f"   pred: {result['relevance']} | {result['information_type']}"
              f"   -> {'CORRECT' if correct else 'WRONG'}")
        if result["error"]:
            print("   error:", result["error"])
        print(f"   calls={result['calls']} tokens={result['total_tokens']} "
              f"time={result['latency_s']:.1f}s")

    results = pd.DataFrame(rows)
    summary = {
        "model": baseline.MODEL_NAME, "temperature": baseline.TEMPERATURE,
        "seed": baseline.SEED, "n_tweets": len(results),
        "joint_exact_match": float(results["joint_correct"].mean()),
        "invalid_outputs": int((~results["valid"]).sum()),
        "avg_latency_s": round(float(results["latency_s"].mean()), 2),
        "avg_total_tokens": round(float(results["total_tokens"].mean()), 1),
        "total_input_tokens": int(results["input_tokens"].sum()),
        "total_output_tokens": int(results["output_tokens"].sum()),
        "total_reasoning_tokens": int(results["reasoning_tokens"].sum()),
    }
    os.makedirs(args.out, exist_ok=True)
    results.to_csv(os.path.join(args.out, "dev_sample_results.csv"), index=False)
    with open(os.path.join(args.out, "dev_sample_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== SUMMARY ===")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"\nSaved to {args.out}/ (IDs and labels only, no tweet text).")


if __name__ == "__main__":
    main()
