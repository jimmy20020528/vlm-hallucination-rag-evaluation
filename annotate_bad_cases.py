import argparse
import os

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-csv", required=True, help="Detailed results CSV")
    parser.add_argument("--output-csv", required=True, help="Annotated output CSV")
    parser.add_argument(
        "--bad-threshold",
        type=int,
        default=1,
        help="hallucinated_count >= threshold will be tagged bad_case",
    )
    return parser.parse_args()


def tag_case(row, bad_threshold):
    if int(row.get("hallucinated_count", 0)) >= bad_threshold:
        return "bad_case"
    if int(row.get("mentioned_count", 0)) > 0 and int(row.get("hallucinated_count", 0)) == 0:
        return "good_case"
    return "neutral_case"


def main():
    args = parse_args()
    df = pd.read_csv(args.results_csv)

    if "experiment_name" not in df.columns:
        # Backward compatible label for old result files.
        df["experiment_name"] = "legacy_run"

    df["case_tag"] = df.apply(lambda r: tag_case(r, args.bad_threshold), axis=1)
    df["case_id"] = df["experiment_name"].astype(str) + "__" + df["method"].astype(str) + "__" + df["image_id"].astype(str)

    out_parent = os.path.dirname(args.output_csv)
    if out_parent:
        os.makedirs(out_parent, exist_ok=True)
    df.to_csv(args.output_csv, index=False)

    bad_df = df[df["case_tag"] == "bad_case"].copy()
    bad_csv = args.output_csv.replace(".csv", "_bad_only.csv")
    bad_df.to_csv(bad_csv, index=False)

    print(f"Annotated file: {args.output_csv}")
    print(f"Bad-case-only file: {bad_csv}")
    print("\nCounts by method and case_tag:")
    print(df.groupby(["method", "case_tag"]).size().to_string())


if __name__ == "__main__":
    main()
