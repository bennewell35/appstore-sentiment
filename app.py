"""CLI to fetch Apple App Store reviews, analyze sentiment, and export CSV.

Usage examples:

# Fetch and analyze (HF default)
python app.py --app_id 284882215 --country us --pages 2 --engine hf --out data/reviews.csv

# Analyze existing CSV using OpenAI
python app.py --csv_in data/reviews_raw.csv --engine openai --out data/reviews_analyzed.csv
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from utils.fetch import fetch_reviews
from utils.analyze import attach_sentiment


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="App Store Sentiment CLI")
    p.add_argument("--app_id", type=str, help="Apple App Store app ID")
    p.add_argument("--country", type=str, default="us", help="App Store country code")
    p.add_argument("--pages", type=int, default=1, help="Number of pages to fetch")
    p.add_argument(
        "--engine",
        type=str,
        choices=["hf", "openai"],
        default="hf",
        help="Sentiment engine",
    )
    p.add_argument("--out", type=str, default="reviews.csv", help="Output CSV path")
    p.add_argument(
        "--csv_in",
        type=str,
        default=None,
        help="Use existing CSV (skip fetching). Expects review columns.",
    )
    return p.parse_args()


def main() -> None:
    setup_logging()
    load_dotenv()

    args = parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.csv_in:
        logging.info("Loading existing CSV: %s", args.csv_in)
        df = pd.read_csv(args.csv_in)
        # Required columns check
        if "body" not in df.columns:
            raise ValueError("CSV must contain a 'body' column for sentiment analysis.")
    else:
        logging.info(
            "Fetching reviews for app_id=%s country=%s pages=%s",
            args.app_id,
            args.country,
            args.pages,
        )
        reviews = fetch_reviews(
            app_id=args.app_id,
            country=args.country,
            pages=args.pages,
            api_key=os.getenv("SERPAPI_API_KEY"),
        )
        df = pd.DataFrame(reviews)
        if df.empty:
            logging.warning("No reviews fetched.")

    logging.info("Analyzing sentiment using engine=%s", args.engine)
    df = attach_sentiment(df, text_col="body", engine=args.engine)

    logging.info("Writing CSV: %s", out_path)
    df.to_csv(out_path, index=False)
    logging.info("Done. Rows: %s", len(df))


if __name__ == "__main__":
    main()
