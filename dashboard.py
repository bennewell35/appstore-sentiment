"""Streamlit dashboard to explore App Store reviews with sentiment.

Run:
    streamlit run dashboard.py
"""
from __future__ import annotations

import os
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="App Store Sentiment Dashboard",
    page_icon="📱",
    layout="wide",
)

st.title("📱 App Store Sentiment Dashboard")

with st.sidebar:
    st.header("Data Source")
    default_path = "reviews.csv"
    csv_path = st.text_input("CSV path", value=default_path)
    uploaded = st.file_uploader("...or upload a CSV", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
    else:
        if not csv_path or not Path(csv_path).exists():
            st.info("Provide a valid CSV path, or upload a file from the sidebar.")
            st.stop()
        df = pd.read_csv(csv_path)

# Basic hygiene
if "date" in df.columns:
    # Attempt to parse to datetime
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

# Filters
with st.sidebar:
    st.header("Filters")
    ratings = sorted([r for r in df.get("rating", pd.Series()).dropna().unique().tolist() if pd.notna(r)])
    selected_ratings = st.multiselect("Ratings", ratings, default=ratings)

    sentiments = sorted(df.get("sentiment", pd.Series(["POSITIVE", "NEGATIVE", "NEUTRAL"]))\
                         .dropna().unique().tolist())
    selected_sentiments = st.multiselect("Sentiment", sentiments, default=sentiments)

    countries = sorted(df.get("country", pd.Series()).dropna().unique().tolist())
    selected_countries = st.multiselect("Country", countries, default=countries)

    q = st.text_input("Search text")

# Apply filters
mask = pd.Series([True] * len(df))
if selected_ratings:
    mask &= df["rating"].isin(selected_ratings)
if selected_sentiments:
    mask &= df["sentiment"].isin(selected_sentiments)
if selected_countries:
    mask &= df["country"].isin(selected_countries)
if q:
    qlower = q.lower()
    body = df.get("body", pd.Series([""] * len(df))).fillna("").astype(str)
    title = df.get("title", pd.Series([""] * len(df))).fillna("").astype(str)
    mask &= body.str.lower().str.contains(qlower) | title.str.lower().str.contains(qlower)

fdf = df[mask].copy()

# KPIs
col1, col2, col3, col4 = st.columns(4)
col1.metric("Rows", f"{len(fdf):,}")
col2.metric("POS", f"{(fdf['sentiment']=='POSITIVE').sum():,}")
col3.metric("NEG", f"{(fdf['sentiment']=='NEGATIVE').sum():,}")
col4.metric("NEU", f"{(fdf['sentiment']=='NEUTRAL').sum():,}")

# Charts
st.subheader("Sentiment distribution")
sent_counts = fdf.groupby("sentiment").size().reset_index(name="count")
sent_chart = (
    alt.Chart(sent_counts)
    .mark_bar()
    .encode(x=alt.X("sentiment:N", sort=["POSITIVE", "NEUTRAL", "NEGATIVE"]), y="count:Q", color="sentiment:N")
    .properties(height=250)
)
st.altair_chart(sent_chart, use_container_width=True)

if "rating" in fdf.columns:
    st.subheader("Rating distribution")
    rating_counts = fdf.groupby("rating").size().reset_index(name="count")
    rating_chart = (
        alt.Chart(rating_counts)
        .mark_bar()
        .encode(x=alt.X("rating:O"), y="count:Q", color="rating:O")
        .properties(height=250)
    )
    st.altair_chart(rating_chart, use_container_width=True)

if "date" in fdf.columns and fdf["date"].notna().any():
    st.subheader("Sentiment over time")
    daily = (
        fdf.dropna(subset=["date"]).assign(date=lambda d: d["date"].dt.date)
        .groupby(["date", "sentiment"]).size().reset_index(name="count")
    )
    line = (
        alt.Chart(daily)
        .mark_line(point=True)
        .encode(x="date:T", y="count:Q", color="sentiment:N")
        .properties(height=280)
    )
    st.altair_chart(line, use_container_width=True)

st.subheader("Table")
st.dataframe(
    fdf[[c for c in ["date", "rating", "sentiment", "confidence", "title", "body", "user_name", "country"] if c in fdf.columns]],
    use_container_width=True,
    height=420,
)

st.download_button(
    label="Download filtered CSV",
    data=fdf.to_csv(index=False).encode("utf-8"),
    file_name="reviews_filtered.csv",
    mime="text/csv",
)
