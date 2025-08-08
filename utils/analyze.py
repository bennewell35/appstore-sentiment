"""Sentiment analysis utilities (HuggingFace default, optional OpenAI).

Provides:
- analyze_sentiment_hf: local transformer pipeline
- analyze_sentiment_openai: OpenAI API classification
- attach_sentiment: convenience to annotate a DataFrame
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class SentimentResult:
    sentiment: str
    confidence: float


# ---------------------- HuggingFace ----------------------

def analyze_sentiment_hf(
    texts: Iterable[str],
    model: str = "distilbert-base-uncased-finetuned-sst-2-english",
    neutral_threshold: float = 0.55,
) -> List[SentimentResult]:
    """Analyze sentiment using HuggingFace transformers pipeline.

    The default model outputs POSITIVE/NEGATIVE. We infer NEUTRAL when
    confidence < neutral_threshold.
    """
    from transformers import pipeline  # lazy import to speed CLI startup

    clf = pipeline("sentiment-analysis", model=model)

    results: List[SentimentResult] = []
    for text in tqdm(list(texts), desc="HF sentiment"):
        if not isinstance(text, str) or not text.strip():
            results.append(SentimentResult("NEUTRAL", 0.0))
            continue
        out = clf(text[:4000])  # guard overly long inputs
        if isinstance(out, list) and out:
            out = out[0]
        label = str(out.get("label", "NEUTRAL")).upper()
        score = float(out.get("score", 0.0))
        if score < neutral_threshold:
            results.append(SentimentResult("NEUTRAL", round(1 - score, 4)))
        else:
            label = "POSITIVE" if "POS" in label else ("NEGATIVE" if "NEG" in label else "NEUTRAL")
            results.append(SentimentResult(label, round(score, 4)))
    return results


# ---------------------- OpenAI ----------------------

def analyze_sentiment_openai(
    texts: Iterable[str],
    model: str | None = None,
    request_timeout: int = 30,
) -> List[SentimentResult]:
    """Analyze sentiment using OpenAI (one call per text, robust and simple).

    Requires OPENAI_API_KEY. Uses a terse JSON instruction for reliable parsing.
    """
    from openai import OpenAI  # lazy import

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is required for OpenAI sentiment.")

    client = OpenAI(api_key=api_key)
    model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    sys = (
        "You are a precise sentiment classifier. Given text, output a JSON object with keys "
        "'sentiment' (POSITIVE, NEGATIVE, or NEUTRAL) and 'confidence' (0..1)."
    )

    results: List[SentimentResult] = []
    for text in tqdm(list(texts), desc="OpenAI sentiment"):
        if not isinstance(text, str) or not text.strip():
            results.append(SentimentResult("NEUTRAL", 0.0))
            continue
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": sys},
                    {
                        "role": "user",
                        "content": (
                            "Classify the sentiment of the following App Store review.\n"
                            "Return only JSON, no prose.\n\n"
                            f"Review: " + text[:6000]
                        ),
                    },
                ],
                temperature=0.0,
                max_tokens=30,
                timeout=request_timeout,
            )
            content = resp.choices[0].message.content or "{}"
            data = json.loads(content)
            sentiment = str(data.get("sentiment", "NEUTRAL")).upper()
            if sentiment not in {"POSITIVE", "NEGATIVE", "NEUTRAL"}:
                sentiment = "NEUTRAL"
            conf = float(data.get("confidence", 0.0))
            conf = max(0.0, min(1.0, conf))
            results.append(SentimentResult(sentiment, round(conf, 4)))
        except Exception as e:  # noqa: BLE001
            logger.warning("OpenAI classification failed: %s", e)
            results.append(SentimentResult("NEUTRAL", 0.0))
    return results


# ---------------------- DataFrame helper ----------------------

def attach_sentiment(
    df: pd.DataFrame,
    text_col: str = "body",
    engine: str = "hf",
) -> pd.DataFrame:
    """Attach sentiment columns to a DataFrame of reviews.

    Args:
        df: Input DataFrame.
        text_col: Column with review text.
        engine: 'hf' (default) or 'openai'.

    Returns:
        Same DataFrame with 'sentiment' and 'confidence' columns.
    """
    texts = df[text_col].fillna("").astype(str).tolist()

    if engine == "hf":
        results = analyze_sentiment_hf(texts)
    elif engine == "openai":
        results = analyze_sentiment_openai(texts)
    else:
        raise ValueError("engine must be 'hf' or 'openai'")

    df = df.copy()
    df["sentiment"] = [r.sentiment for r in results]
    df["confidence"] = [r.confidence for r in results]
    return df
