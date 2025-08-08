# App Store Sentiment

This repo fetches Apple App Store reviews via SerpAPI, analyzes sentiment (HuggingFace by default, optional OpenAI), exports to CSV, and includes a Streamlit dashboard for exploration.

## Features
- Fetch reviews from Apple App Store using SerpAPI
- Sentiment analysis using HuggingFace Transformers (default)
- Optional sentiment via OpenAI
- Robust CLI with retries, timeouts, simple logging
- CSV output; can analyze an existing CSV
- Streamlit dashboard for filtering and visualization

## Quickstart

1) Create and activate a virtual environment
```
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

2) Install dependencies
```
pip install -r requirements.txt
```

3) Configure environment
- Copy .env.example to .env and set values
```
cp .env.example .env
```

Required:
- SERPAPI_API_KEY

Optional (only if using OpenAI engine):
- OPENAI_API_KEY
- OPENAI_MODEL (default: gpt-4o-mini, fallback: gpt-3.5-turbo)

4) Fetch, analyze, and export CSV via CLI
```
python app.py \
  --app_id 284882215 \
  --country us \
  --pages 2 \
  --engine hf \
  --out data/reviews_analyzed.csv
```

Or analyze an existing CSV (skip fetching):
```
python app.py --csv_in data/reviews_raw.csv --engine openai --out data/reviews_analyzed.csv
```

5) Launch the Streamlit dashboard
```
streamlit run dashboard.py
```
Then select your CSV from the sidebar or paste its path.

## CLI Reference
```
usage: app.py [-h] [--app_id APP_ID] [--country COUNTRY] [--pages PAGES]
              [--engine {hf,openai}] [--out OUT] [--csv_in CSV_IN]

Options:
  --app_id        Apple App Store app ID (e.g., 284882215 for Facebook)
  --country       App Store country code (default: us)
  --pages         Number of pages to fetch (default: 1)
  --engine        Sentiment engine: hf (default) or openai
  --out           Output CSV path (default: reviews.csv)
  --csv_in        Existing CSV path to analyze (skips fetching)
```

## Data Columns
- review_id, user_name, rating, title, body, date, country, app_id, source, language
- sentiment, confidence

## Notes
- HuggingFace model: distilbert-base-uncased-finetuned-sst-2-english
- Neutral detection is inferred by confidence threshold; adjust in utils/analyze.py
- SerpAPI endpoint: engine=apple_app_store_reviews

## Troubleshooting
- HTTP 429/Rate limit: increase delay or reduce pages
- Large model downloads: first run may take time; ensure stable internet
- OpenAI: set OPENAI_API_KEY and possibly OPENAI_MODEL

## License
MIT
