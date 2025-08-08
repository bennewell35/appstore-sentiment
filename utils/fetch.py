"""Utilities to fetch Apple App Store reviews via SerpAPI.

This module provides a resilient fetcher with retries, timeouts, and
normalization to a stable schema suitable for analysis.
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from typing import Any, Dict, List, Optional

import requests

SERPAPI_ENDPOINT = "https://serpapi.com/search.json"
APPLE_RSS_TEMPLATE = "https://itunes.apple.com/{country}/rss/customerreviews/page={page}/id={app_id}/sortby=mostrecent/json"

logger = logging.getLogger(__name__)


def _request_with_retries(
    params: Dict[str, Any],
    timeout: int = 20,
    max_retries: int = 5,
    backoff_factor: float = 1.5,
) -> Dict[str, Any]:
    """Perform a GET request with simple exponential backoff.

    Args:
        params: Query parameters for the SerpAPI endpoint.
        timeout: Seconds before timing out the request.
        max_retries: Maximum number of retry attempts.
        backoff_factor: Multiplier for exponential backoff sleep.

    Returns:
        Parsed JSON response.

    Raises:
        requests.HTTPError: If non-200 response persists after retries.
        requests.RequestException: For network-related errors after retries.
    """
    delay = 1.0
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(SERPAPI_ENDPOINT, params=params, timeout=timeout)
            if resp.status_code == 200:
                return resp.json()
            # Retry on 429/5xx
            if resp.status_code in {429, 500, 502, 503, 504}:
                logger.warning(
                    "SerpAPI returned %s. Attempt %s/%s. Sleeping %.1fs",
                    resp.status_code,
                    attempt,
                    max_retries,
                    delay,
                )
                time.sleep(delay)
                delay *= backoff_factor
                continue
            resp.raise_for_status()
        except requests.RequestException as e:
            if attempt == max_retries:
                logger.error("Request failed after %s attempts: %s", max_retries, e)
                raise
            logger.warning(
                "Request exception: %s. Attempt %s/%s. Sleeping %.1fs",
                e,
                attempt,
                max_retries,
                delay,
            )
            time.sleep(delay)
            delay *= backoff_factor
    # Shouldn't reach here
    raise RuntimeError("Exhausted retries without response")


def _normalize_review(raw: Dict[str, Any], app_id: str, country: str) -> Dict[str, Any]:
    """Normalize varying SerpAPI review shapes to a consistent schema.

    The SerpAPI `apple_app_store_reviews` engine typically returns a `reviews` list
    with fields like: review_id, title, body, rating, user_name, date, language, source.
    We defensively access keys to support minor variations.
    """
    review_id = (
        raw.get("review_id")
        or raw.get("id")
        or hashlib.sha256(
            (
                str(raw.get("title")) + str(raw.get("body")) + str(raw.get("date"))
            ).encode("utf-8")
        ).hexdigest()[:16]
    )

    return {
        "review_id": review_id,
        "user_name": raw.get("user_name") or raw.get("user") or raw.get("author"),
        "rating": raw.get("rating"),
        "title": raw.get("title"),
        "body": raw.get("body") or raw.get("content"),
        "date": raw.get("date") or raw.get("updated"),
        "country": country,
        "app_id": app_id,
        "source": raw.get("source") or "apple_app_store",
        "language": raw.get("language"),
    }


def _request_rss_with_retries(
    url: str,
    timeout: int = 20,
    max_retries: int = 5,
    backoff_factor: float = 1.5,
) -> Dict[str, Any]:
    """GET a JSON RSS URL with retries and simple exponential backoff.

    The Apple RSS endpoint does not require an API key.
    """
    delay = 1.0
    headers = {"User-Agent": "appstore-sentiment/1.0 (+https://example.com)"}
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, headers=headers, timeout=timeout)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code in {429, 500, 502, 503, 504}:
                logger.warning(
                    "Apple RSS returned %s. Attempt %s/%s. Sleeping %.1fs",
                    resp.status_code,
                    attempt,
                    max_retries,
                    delay,
                )
                time.sleep(delay)
                delay *= backoff_factor
                continue
            resp.raise_for_status()
        except requests.RequestException as e:  # noqa: PERF203
            if attempt == max_retries:
                logger.error("RSS request failed after %s attempts: %s", max_retries, e)
                raise
            logger.warning(
                "RSS request exception: %s. Attempt %s/%s. Sleeping %.1fs",
                e,
                attempt,
                max_retries,
                delay,
            )
            time.sleep(delay)
            delay *= backoff_factor
    raise RuntimeError("Exhausted RSS retries without response")


def _normalize_rss_entry(
    raw: Dict[str, Any], app_id: str, country: str
) -> Dict[str, Any]:
    """Normalize an Apple RSS review entry into our schema.

    The RSS JSON uses namespaced keys like 'im:rating'. Values usually live
    under 'label'. We defensively extract them.
    """

    def _get_label(dct: Optional[Dict[str, Any]]) -> Optional[str]:
        if not isinstance(dct, dict):
            return None
        val = dct.get("label")
        return str(val) if val is not None else None

    review_id = (
        _get_label(raw.get("id"))
        or hashlib.sha256(
            (
                str(_get_label(raw.get("title")))
                + str(_get_label(raw.get("content")))
                + str(_get_label(raw.get("updated")))
            ).encode("utf-8")
        ).hexdigest()[:16]
    )

    rating_str = _get_label(raw.get("im:rating"))
    try:
        rating: Optional[int] = int(rating_str) if rating_str is not None else None
    except Exception:  # noqa: BLE001
        rating = None

    return {
        "review_id": review_id,
        "user_name": _get_label((raw.get("author") or {}).get("name")),
        "rating": rating,
        "title": _get_label(raw.get("title")),
        "body": _get_label(raw.get("content")),
        "date": _get_label(raw.get("updated")),
        "country": country,
        "app_id": app_id,
        "source": "apple_rss",
        "language": None,
    }


def fetch_reviews_rss(
    app_id: str,
    country: str,
    pages: int,
    delay_between_pages: float = 0.8,
) -> List[Dict[str, Any]]:
    """Fetch reviews using Apple's public RSS feed (no API key required).

    Apple typically exposes up to ~10 pages. We stop early when no review
    entries are present (note: the first entry is app metadata, so we expect
    at least 2 entries for a non-empty page).
    """
    all_reviews: List[Dict[str, Any]] = []
    empty_pages_in_row = 0

    for page in range(1, max(1, int(pages)) + 1):
        url = APPLE_RSS_TEMPLATE.format(country=country, page=page, app_id=app_id)
        logger.info(
            "Fetching RSS page %s/%s for app_id=%s country=%s",
            page,
            pages,
            app_id,
            country,
        )
        data = _request_rss_with_retries(url)

        feed = data.get("feed", {}) if isinstance(data, dict) else {}
        entries = feed.get("entry") or []
        if not isinstance(entries, list):
            logger.warning(
                "Unexpected RSS 'entry' payload on page %s: %s", page, type(entries)
            )
            entries = []

        # First entry is app metadata; reviews start from second entry.
        review_entries = entries[1:] if len(entries) >= 2 else []
        normalized = [_normalize_rss_entry(e, app_id, country) for e in review_entries]
        logger.info("Fetched %s RSS reviews on page %s", len(normalized), page)
        all_reviews.extend(normalized)

        if not normalized:
            empty_pages_in_row += 1
        else:
            empty_pages_in_row = 0

        if empty_pages_in_row >= 1:  # stop on first empty page to be efficient
            logger.info("No RSS reviews returned. Stopping early at page %s.", page)
            break

        time.sleep(delay_between_pages)

    return all_reviews


def fetch_reviews(
    app_id: str,
    country: str,
    pages: int,
    api_key: Optional[str] = None,
    delay_between_pages: float = 0.8,
) -> List[Dict[str, Any]]:
    """Fetch reviews for a given app from Apple App Store via SerpAPI.

    Args:
        app_id: Apple app ID (numeric string).
        country: App Store country code (e.g., 'us').
        pages: Number of pages to fetch.
        api_key: SerpAPI API key. If None, taken from SERPAPI_API_KEY env.
        delay_between_pages: Sleep seconds between page requests to be polite.

    Returns:
        List of normalized review dicts.
    """
    api_key = api_key or os.getenv("SERPAPI_API_KEY")

    # If no API key is available, fall back to the Apple RSS feed (no key required)
    if not api_key:
        logger.info(
            "SERPAPI_API_KEY not set. Falling back to Apple RSS (no key required)."
        )
        return fetch_reviews_rss(
            app_id=app_id,
            country=country,
            pages=pages,
            delay_between_pages=delay_between_pages,
        )

    all_reviews: List[Dict[str, Any]] = []

    for page in range(1, max(1, int(pages)) + 1):
        params = {
            "engine": "apple_app_store_reviews",
            "app_id": app_id,
            "country": country,
            "page": page,
            "api_key": api_key,
        }
        logger.info(
            "Fetching page %s/%s for app_id=%s country=%s", page, pages, app_id, country
        )
        data = _request_with_retries(params)

        reviews = data.get("reviews") or []
        if not isinstance(reviews, list):
            logger.warning(
                "Unexpected 'reviews' payload on page %s: %s", page, type(reviews)
            )
            reviews = []

        normalized = [_normalize_review(r, app_id, country) for r in reviews]
        logger.info("Fetched %s reviews on page %s", len(normalized), page)
        all_reviews.extend(normalized)

        # If the API indicates no more pages, we can break early
        if not reviews:
            logger.info("No reviews returned. Stopping early at page %s.", page)
            break

        time.sleep(delay_between_pages)

    return all_reviews
