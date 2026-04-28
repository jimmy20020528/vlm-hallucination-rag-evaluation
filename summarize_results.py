import argparse
import os

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-csv", required=True)
    parser.add_argument("--output-csv", default="outputs/summary.csv")
    return parser.parse_args()


def safe_div(a, b):
    return (a / b) if b else 0.0


def main():
    args = parse_args()
    df = pd.read_csv(args.results_csv)
    output_parent = os.path.dirname(args.output_csv)
    if output_parent:
        os.makedirs(output_parent, exist_ok=True)

    summaries = []
    for method, g in df.groupby("method"):
        total = len(g)
        example_with_h = int(g["has_hallucination"].sum())
        avg_h_count = float(g["hallucinated_count"].mean())

        total_mentioned = int(g["mentioned_count"].sum())
        total_hallucinated = int(g["hallucinated_count"].sum())
        in_gt_mentions = total_mentioned - total_hallucinated

        object_precision = (in_gt_mentions / total_mentioned) if total_mentioned else None

        summaries.append(
            {
                "method": method,
                "num_examples": total,
                "hallucination_rate": round(safe_div(example_with_h, total), 4),
                "avg_hallucinated_mentions": round(avg_h_count, 4),
                "object_precision": (round(object_precision, 4) if object_precision is not None else "NA"),
            }
        )

    summary_df = pd.DataFrame(summaries).sort_values("method")
    summary_df.to_csv(args.output_csv, index=False)

    print("Summary metrics:")
    print(summary_df.to_string(index=False))
    print(f"\nSaved summary to: {args.output_csv}")


if __name__ == "__main__":
    main()
