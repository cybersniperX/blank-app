import time
import re
from collections import Counter
from urllib.parse import urlparse, parse_qs

import pandas as pd
import streamlit as st
import plotly.express as px
from google_play_scraper import app, reviews, Sort


model = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-xlm-roberta-base-sentiment"
)

st.set_page_config(page_title="Google Play Scraper and Analysis Dashboard", layout="wide")
st.title("Google Play Scraper")

st.markdown("""
<style>
.small-stat-label {
    font-size: 12px;
    color: #666;
}
.small-stat-value {
    font-size: 18px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

SORT_OPTIONS = [
    ("Newest", Sort.NEWEST),
]

COUNTRY_OPTIONS = ["ph"]

COUNT_PER_BATCH = 200
SCRAPE_MAX_ROUNDS_PER_QUERY = 3000

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text

def format_time(seconds):
    if seconds is None:
        return "-"
    if seconds < 60:
        sec = max(1, round(seconds))
        return f"{sec} second" if sec == 1 else f"{sec} seconds"
    mins = max(1, round(seconds / 60))
    return f"{mins} minute" if mins == 1 else f"{mins} minutes"

def extract_app_id_from_input(value):
    text = (value or "").strip()
    if not text:
        return None

    if "play.google.com" in text:
        try:
            parsed = urlparse(text)
            return parse_qs(parsed.query).get("id", [None])[0]
        except Exception:
            return None

    return text


def build_query_plan():
    plan = []
    for country in COUNTRY_OPTIONS:
        for sort_name, sort_value in SORT_OPTIONS:
            plan.append({
                "country": country,
                "sort_name": sort_name,
                "sort_value": sort_value,
            })
    return plan


def reset_scrape_state():
    st.session_state.scraping = False
    st.session_state.scrape_done = False
    st.session_state.cancel_requested = False
    st.session_state.all_reviews = []
    st.session_state.batch_index = 0
    st.session_state.scrape_start = None
    st.session_state.auto_download_ready = False
    st.session_state.current_plan_index = 0
    st.session_state.current_token = None
    st.session_state.current_query_rounds = 0
    st.session_state.current_stagnant_rounds = 0
    st.session_state.seen_review_ids = set()
    st.session_state.scrapable_reviews = None
    st.session_state.estimated_time_low = None
    st.session_state.estimated_time_high = None
    st.session_state.corrected_total_reviews = st.session_state.total_reviews_reported or 0


def clear_selected_app():
    st.session_state.ready_to_scrape = False
    st.session_state.checked_app_id = None
    st.session_state.checked_app_title = None
    st.session_state.checked_app_icon = None
    st.session_state.loading_app = False
    st.session_state.total_reviews_reported = None
    st.session_state.corrected_total_reviews = None
    st.session_state.scrapable_reviews = None
    st.session_state.estimated_time_low = None
    st.session_state.estimated_time_high = None


def load_app_details():
    app_id = extract_app_id_from_input(st.session_state.app_input)

    if not app_id:
        st.session_state.loading_app = False
        st.error("Invalid link or package ID.")
        return

    reset_scrape_state()
    clear_selected_app()
    st.session_state.loading_app = True

    try:
        with st.spinner("Loading..."):
            result = app(app_id, lang="en", country="ph")
            reported_total_reviews = int(result.get("reviews", 0) or 0)

        st.session_state.checked_app_id = app_id
        st.session_state.checked_app_title = result.get("title", app_id)
        st.session_state.checked_app_icon = result.get("icon")
        st.session_state.total_reviews_reported = reported_total_reviews
        st.session_state.corrected_total_reviews = reported_total_reviews
        st.session_state.ready_to_scrape = True
        st.session_state.loading_app = False

    except Exception as e:
        st.session_state.loading_app = False
        st.error(f"Error loading app details: {e}")


def get_current_plan_item():
    plan = build_query_plan()
    if st.session_state.current_plan_index >= len(plan):
        return None
    return plan[st.session_state.current_plan_index]


def add_new_reviews(batch):
    added = 0
    for row in batch:
        rid = row.get("reviewId")
        if rid and rid not in st.session_state.seen_review_ids:
            st.session_state.seen_review_ids.add(rid)
            st.session_state.all_reviews.append(row)
            added += 1
    return added


def get_sentiment_result(text):
    try:
        result = model(str(text), truncation=True)[0]
        label_map = {
            "LABEL_0": "negative",
            "LABEL_1": "neutral",
            "LABEL_2": "positive"
        }

        sentiment_label = label_map.get(result["label"], "neutral")

        if sentiment_label == "positive":
            sentiment_score = float(result["score"])
        elif sentiment_label == "negative":
            sentiment_score = -float(result["score"])
        else:
            sentiment_score = 0.0

        return pd.Series([sentiment_score, sentiment_label])

    except Exception:
        return pd.Series([0.0, "neutral"])


def run_analysis(raw_df):
    st.title("Analysis Dashboard")

    try:
        df = raw_df.copy()

        # DROP UNNECESSARY COLUMNS
        df.drop(
            ["userName", "userImage", "reviewId", "replyContent", "repliedAt", "appVersion"],
            axis=1,
            inplace=True,
            errors="ignore"
        )

        # MAKE SURE CONTENT EXISTS
        if "content" not in df.columns:
            st.error("The scraped data does not contain a 'content' column.")
            st.stop()

        df["clean_text"] = df["content"].astype(str).str.lower()
        df["clean_text"] = df["clean_text"].str.replace(r"http\S+", "", regex=True)
        df["clean_text"] = df["clean_text"].str.replace(r"[^a-zA-Z\s]", "", regex=True)

        # RENAME DATE COLUMN
        if "at" in df.columns:
            df.rename(columns={"at": "reviewDate"}, inplace=True)

        # DATE CLEANING
        if "reviewDate" in df.columns:
            df["reviewDate"] = pd.to_datetime(df["reviewDate"], errors="coerce")
            df = df.dropna(subset=["reviewDate"])
            df["month"] = df["reviewDate"].dt.to_period("M").astype(str)

        # CLEAN SCORE
        if "score" in df.columns:
            df["score"] = pd.to_numeric(df["score"], errors="coerce")
            df = df.dropna(subset=["score"])
        else:
            st.error("The scraped data does not contain a 'score' column.")
            st.stop()

        # SENTIMENT SCORE AND LABEL
        df[["sentiment_score", "sentiment"]] = df["content"].astype(str).apply(get_sentiment_result)

        # ACTUAL LABEL FROM RATING
        def actual_label(score):
            if score >= 4:
                return "positive"
            elif score <= 2:
                return "negative"
            else:
                return "neutral"

        df["actual_label"] = df["score"].apply(actual_label)

        # REVIEW LENGTH
        df["reviewWordCount"] = df["content"].astype(str).apply(lambda x: len(x.split()))

        # SUMMARY
        st.subheader("Summary")

        col1, col2 = st.columns(2)

        sent_summary = df.groupby("sentiment").agg(
            total_count=("sentiment", "count"),
            weighted=("score", "mean")
        ).reset_index()

        sent_summary["weighted"] = sent_summary["weighted"].round(2)
        sent_summary = sent_summary.sort_values(by="weighted", ascending=False)

        with col1:
            st.markdown("### Sentiment Summary")
            st.dataframe(sent_summary, use_container_width=True)

        class_summary = df.groupby("actual_label").agg(
            total_count=("actual_label", "count"),
            weighted=("score", "mean")
        ).reset_index()

        class_summary["weighted"] = class_summary["weighted"].round(2)
        class_summary = class_summary.sort_values(by="weighted", ascending=False)

        with col2:
            st.markdown("### Classified Summary")
            st.dataframe(class_summary, use_container_width=True)

        summary = df["sentiment"].value_counts().reset_index()
        summary.columns = ["sentiment", "total_count"]

        # TOP 10 HIGHEST AND LOWEST RATED MONTHS
        if "reviewDate" in df.columns:
            df["reviewDate"] = pd.to_datetime(df["reviewDate"], errors="coerce")
            df = df.dropna(subset=["reviewDate"])

            df["month_period"] = df["reviewDate"].dt.to_period("M")
            df["month"] = df["month_period"].astype(str)

            monthly_summary = df.groupby(["month_period", "month"]).agg(
                avg_rating=("score", "mean"),
                review_count=("score", "count")
            ).reset_index()

            monthly_summary["avg_rating"] = monthly_summary["avg_rating"].round(2)
            monthly_summary = monthly_summary[monthly_summary["review_count"] >= 5]

            top_10_highest = monthly_summary.sort_values(
                by="avg_rating", ascending=False
            ).head(10)

            top_10_lowest = monthly_summary.sort_values(
                by="avg_rating", ascending=True
            ).head(10)

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Top 10 Highest Rated Months")
                st.dataframe(
                    top_10_highest[["month", "avg_rating", "review_count"]],
                    use_container_width=True
                )

            with col2:
                st.subheader("Top 10 Lowest Rated Months")
                st.dataframe(
                    top_10_lowest[["month", "avg_rating", "review_count"]],
                    use_container_width=True
                )

        st.subheader("Top 30 Most Frequent Words")

        default_stopwords = [
            "the", "and", "is", "to", "of", "in", "for", "on", "with", "this",
            "that", "it", "my", "app", "very", "so", "but", "are", "was", "be",
            "have", "has", "had", "not", "at", "you", "we", "they", "i", "ang",
            "nag", "your", "nyo"
        ]

        custom_stopwords = st.text_area(
            "Edit excluded words (comma-separated)",
            value=",".join(default_stopwords)
        )

        stopwords = set(word.strip().lower() for word in custom_stopwords.split(","))

        text = " ".join(df["content"].dropna().astype(str))
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        filtered_words = [w for w in words if w not in stopwords and len(w) > 2]
        word_counts = Counter(filtered_words)
        top_words = word_counts.most_common(30)
        top_words_df = pd.DataFrame(top_words, columns=["word", "count"])

        fig_words = px.bar(
            top_words_df,
            x="word",
            y="count",
            title="Top 30 Words in Reviews"
        )

        st.plotly_chart(fig_words, use_container_width=True)

        excluded_words = [
            "the", "and", "is", "to", "of", "in", "for", "on", "with", "this",
            "that", "it", "my", "app", "very", "so", "but", "are", "was", "be",
            "have", "has", "had", "not", "at", "you", "we", "they", "i",
            "ang", "nag", "your", "nyo"
        ]

        st.write("Excluded words:", ", ".join(excluded_words))

        # DISTRIBUTION
        st.subheader("Sentiment")

        fig_bar = px.bar(
            summary,
            x="sentiment",
            y="total_count",
            text="total_count",
            title="Count of Sentiment"
        )
        fig_bar.update_traces(textposition="outside")

        fig_pie = px.pie(
            summary,
            names="sentiment",
            values="total_count",
            title="Sentiment Distribution"
        )

        heatmap = df.groupby(["month", "sentiment"]).size().reset_index(name="count")
        pivot = heatmap.pivot(index="sentiment", columns="month", values="count").fillna(0)

        fig_heatmap = px.imshow(
            pivot,
            text_auto=True,
            aspect="auto",
            title="Monthly Sentiment Heatmap"
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            st.plotly_chart(fig_bar, use_container_width=True)
        with col2:
            st.plotly_chart(fig_pie, use_container_width=True)
        with col3:
            st.plotly_chart(fig_heatmap, use_container_width=True)

        # CLASSIFIED SUMMARY
        st.subheader("Classified Summary")

        actual_summary = df["actual_label"].value_counts().reset_index()
        actual_summary.columns = ["actual_label", "total_count"]

        fig_actual_bar = px.bar(
            actual_summary,
            x="actual_label",
            y="total_count",
            text="total_count",
            title="Count of Classified Labels"
        )
        fig_actual_bar.update_traces(textposition="outside")

        fig_actual_pie = px.pie(
            actual_summary,
            names="actual_label",
            values="total_count",
            title="Classified Label Distribution"
        )

        actual_heatmap = df.groupby(["month", "actual_label"]).size().reset_index(name="count")
        actual_pivot = actual_heatmap.pivot(
            index="actual_label",
            columns="month",
            values="count"
        ).fillna(0)

        fig_actual_heatmap = px.imshow(
            actual_pivot,
            text_auto=True,
            aspect="auto",
            title="Monthly Classified Heatmap"
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            st.plotly_chart(fig_actual_bar, use_container_width=True)
        with col2:
            st.plotly_chart(fig_actual_pie, use_container_width=True)
        with col3:
            st.plotly_chart(fig_actual_heatmap, use_container_width=True)

        # REVIEW SCORE
        st.subheader("Review Score Over Time")

        rating_trend = df.groupby("month")["score"].mean().reset_index()

        fig_rating = px.line(
            rating_trend,
            x="month",
            y="score",
            title="Average Review Score Over Time"
        )

        volume = df.groupby("month").size().reset_index(name="count")

        fig_volume = px.line(
            volume,
            x="month",
            y="count",
            title="Review Volume Over Time"
        )

        # SENTIMENT OVER TIME
        avg_sent = df.groupby("month")["sentiment_score"].mean().reset_index()

        fig_avg = px.line(
            avg_sent,
            x="month",
            y="sentiment_score",
            title="Average Sentiment Score Over Time"
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            st.plotly_chart(fig_rating, use_container_width=True)
        with col2:
            st.plotly_chart(fig_volume, use_container_width=True)
        with col3:
            st.plotly_chart(fig_avg, use_container_width=True)

        # REVIEW LENGTH
        st.subheader("Review Length by Rating")

        col1, col2 = st.columns(2)

        fig_length = px.box(
            df,
            x="score",
            y="reviewWordCount",
            title="Review Length by Rating"
        )

        with col1:
            st.plotly_chart(fig_length, use_container_width=True)

        fig_scatter = px.scatter(
            df,
            x="score",
            y="reviewWordCount",
            color="sentiment",
            title="Review Length vs Rating",
            opacity=0.6
        )

        with col2:
            st.plotly_chart(fig_scatter, use_container_width=True)

        # PREDICTED VS ACTUAL
        st.subheader("Predicted vs Actual")

        cm_df = pd.crosstab(
            df["actual_label"],
            df["sentiment"],
            rownames=["Actual"],
            colnames=["Predicted"]
        )

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Confusion Matrix (Table)")
            st.dataframe(cm_df, use_container_width=True)

        labels = ["positive", "neutral", "negative"]
        report = []

        for label in labels:
            tp = ((df["actual_label"] == label) & (df["sentiment"] == label)).sum()
            fp = ((df["actual_label"] != label) & (df["sentiment"] == label)).sum()
            fn = ((df["actual_label"] == label) & (df["sentiment"] != label)).sum()

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            report.append({
                "label": label,
                "precision": round(precision, 3),
                "recall": round(recall, 3),
                "f1_score": round(f1, 3)
            })

        report_df = pd.DataFrame(report)

        with col2:
            st.markdown("### Classification Report")
            st.dataframe(report_df, use_container_width=True)

        fig_cm = px.imshow(
            cm_df,
            text_auto=True,
            aspect="auto",
            title="Confusion Matrix"
        )

        rating_sentiment = df.groupby("score")["sentiment_score"].mean().reset_index()

        fig = px.line(
            rating_sentiment,
            x="score",
            y="sentiment_score",
            title="Average Sentiment Score per Rating"
        )

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig_cm, use_container_width=True)
        with col2:
            st.plotly_chart(fig, use_container_width=True)

        df["match"] = df["actual_label"] == df["sentiment"]
        accuracy = df["match"].mean()
        st.write(f"Model Accuracy: {accuracy:.2%}")

        # FINAL DATASET
        st.subheader("Processed Dataset")

        final_df = df[
            [
                "reviewDate",
                "content",
                "score",
                "sentiment_score",
                "sentiment",
                "actual_label",
                "reviewWordCount"
            ]
        ].copy()

        if "reviewDate" in final_df.columns:
            final_df = final_df.sort_values(by="reviewDate", ascending=False)

        st.dataframe(final_df, use_container_width=True)

        csv = final_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Processed Data",
            data=csv,
            file_name="processed_reviews.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Error reading file: {e}")


defaults = {
    "checked_app_id": None,
    "checked_app_title": None,
    "checked_app_icon": None,
    "total_reviews_reported": None,
    "corrected_total_reviews": None,
    "scrapable_reviews": None,
    "estimated_time_low": None,
    "estimated_time_high": None,
    "ready_to_scrape": False,
    "scraping": False,
    "scrape_done": False,
    "cancel_requested": False,
    "all_reviews": [],
    "batch_index": 0,
    "scrape_start": None,
    "app_input": "",
    "loading_app": False,
    "auto_download_ready": False,
    "current_plan_index": 0,
    "current_token": None,
    "current_query_rounds": 0,
    "current_stagnant_rounds": 0,
    "seen_review_ids": set(),
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


st.subheader("Find an app")

left, right = st.columns([1, 1])

with left:
    st.text_input(
        "Google Play link or package ID",
        key="app_input",
        placeholder=""
    )

    btn_col1, btn_col2 = st.columns(2)

    with btn_col1:
        if st.button("Load App", disabled=st.session_state.loading_app, use_container_width=True):
            load_app_details()

    with btn_col2:
        if st.session_state.checked_app_id and not st.session_state.scraping and not st.session_state.scrape_done:
            if st.button("Scrape", use_container_width=True):
                reset_scrape_state()
                st.session_state.ready_to_scrape = True
                st.session_state.scraping = True
                st.session_state.scrape_start = time.time()
                st.rerun()

        elif st.session_state.scraping:
            if st.button("Cancel Scrape", use_container_width=True):
                st.session_state.cancel_requested = True
                st.session_state.scraping = False
                st.rerun()

        elif st.session_state.scrape_done and st.session_state.all_reviews:
            df_download = pd.DataFrame(st.session_state.all_reviews)
            csv_bytes = df_download.to_csv(index=False).encode("utf-8")
            safe_name = (st.session_state.checked_app_id or "reviews").replace(".", "_")

            st.download_button(
                label="Download CSV",
                data=csv_bytes,
                file_name=f"{safe_name}_reviews.csv",
                mime="text/csv",
                use_container_width=True
            )

with right:
    if st.session_state.loading_app:
        st.write("Loading app details...")

    if st.session_state.checked_app_id:
        icon_col, title_col = st.columns([1, 4])

        with icon_col:
            if st.session_state.checked_app_icon:
                st.image(st.session_state.checked_app_icon, width=70)

        with title_col:
            st.write(f"**{st.session_state.checked_app_title}**")

            if st.session_state.scraping:
                reported_total = st.session_state.total_reviews_reported or 0
                current_total = len(st.session_state.all_reviews)
                corrected_total = max(reported_total, current_total, 1)

                st.session_state.corrected_total_reviews = corrected_total

                progress_value = min(current_total / corrected_total, 1.0)
                st.progress(progress_value)

                remaining_total = max(corrected_total - current_total, 0)

                if st.session_state.scrape_start:
                    elapsed = time.time() - st.session_state.scrape_start
                    if current_total > 0 and remaining_total > 0:
                        rate = elapsed / current_total
                        remaining_time = remaining_total * rate
                    else:
                        remaining_time = 0
                else:
                    remaining_time = None

                status_text = (
                    f"Estimated Total Reviews: {corrected_total:,} \n "
                    f"Scraped Reviews: {current_total:,} \n "
                    f"Estimated time to complete: {format_time(remaining_time)}"
                )
                st.text(status_text)

                if reported_total > 0 and current_total > reported_total:
                    st.caption(
                        f"Google Play reported {reported_total:,} reviews, "
                        f"but {current_total:,} unique reviews have already been retrieved."
                    )

            elif st.session_state.ready_to_scrape and not st.session_state.scrape_done:
                reported_total = st.session_state.total_reviews_reported or 0
                corrected_total = st.session_state.corrected_total_reviews or reported_total

                if corrected_total:
                    st.write(f"Estimated Total Reviews: {corrected_total:,}")

            elif st.session_state.scrape_done:
                final_total = len(st.session_state.all_reviews)
                corrected_total = max(
                    st.session_state.total_reviews_reported or 0,
                    final_total
                )
                st.write(f"Estimated Total Reviews: {corrected_total:,}")
                st.write(f"Scraped Reviews: {final_total:,}")


if st.session_state.scraping and not st.session_state.cancel_requested:
    current_item = get_current_plan_item()

    if current_item is None:
        st.session_state.scraping = False
        st.session_state.scrape_done = True
        st.rerun()

    try:
        batch, new_token = reviews(
            st.session_state.checked_app_id,
            lang="en",
            country=current_item["country"],
            sort=current_item["sort_value"],
            count=COUNT_PER_BATCH,
            continuation_token=st.session_state.current_token
        )

        added_count = add_new_reviews(batch)
        st.session_state.batch_index += 1
        st.session_state.current_query_rounds += 1

        current_total = len(st.session_state.all_reviews)
        reported_total = st.session_state.total_reviews_reported or 0
        st.session_state.corrected_total_reviews = max(reported_total, current_total)

        empty_batch = len(batch) == 0
        same_token = new_token == st.session_state.current_token

        if empty_batch or added_count == 0 or same_token:
            st.session_state.current_stagnant_rounds += 1
        else:
            st.session_state.current_stagnant_rounds = 0

        st.session_state.current_token = new_token

        should_advance = (
            new_token is None
            or st.session_state.current_stagnant_rounds >= 2
            or st.session_state.current_query_rounds >= SCRAPE_MAX_ROUNDS_PER_QUERY
        )

        if should_advance:
            st.session_state.current_plan_index += 1
            st.session_state.current_token = None
            st.session_state.current_query_rounds = 0
            st.session_state.current_stagnant_rounds = 0

        time.sleep(0.05)
        st.rerun()

    except Exception as e:
        st.error(f"Scraping error: {e}")
        st.session_state.scraping = False


if st.session_state.scrape_done:
    final_total = len(st.session_state.all_reviews)
    st.session_state.corrected_total_reviews = max(
        st.session_state.total_reviews_reported or 0,
        final_total
    )

    st.success(f"Scraping complete. Total unique reviews extracted: {final_total:,}")

    if st.session_state.all_reviews:
        raw_df = pd.DataFrame(st.session_state.all_reviews)
        run_analysis(raw_df)


if st.session_state.cancel_requested:
    partial_total = len(st.session_state.all_reviews)
    st.session_state.corrected_total_reviews = max(
        st.session_state.total_reviews_reported or 0,
        partial_total
    )
    st.warning(f"Scraping cancelled. Partial unique reviews extracted: {partial_total:,}")