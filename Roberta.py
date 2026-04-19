import time
import re
import pandas as pd
import streamlit as st
import plotly.express as px
import os
import glob

from collections import Counter
from urllib.parse import urlparse, parse_qs
from google_play_scraper import app, reviews, Sort
from transformers import pipeline


st.set_page_config(page_title="Google Play Scraper and Analysis Dashboard", layout="wide")
st.title("Google Play Scraper")


@st.cache_resource
def load_roberta_model():
    return pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-xlm-roberta-base-sentiment",
        tokenizer="cardiffnlp/twitter-xlm-roberta-base-sentiment"
    )


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
ROBERTA_BATCH_SIZE = 100 
ROBERTA_MAX_LENGTH = 128
CHECKPOINT_INTERVAL = 50000
CHECKPOINT_DIR = "checkpoints"


def safe_name(text):
    return re.sub(r"[^A-Za-z0-9._-]", "_", str(text or "default"))


def get_analysis_source_id(app_id=None, raw_filename=None):
    if app_id:
        return f"app_{safe_name(app_id)}"
    if raw_filename:
        return f"file_{safe_name(raw_filename)}"
    return "default"


def clear_saved_checkpoints():
    checkpoint_patterns = [
        "latest_checkpoint.csv",
        "checkpoint_*.csv",
        "*checkpoint*.csv",
    ]

    for pattern in checkpoint_patterns:
        for file_path in glob.glob(pattern):
            try:
                os.remove(file_path)
            except Exception:
                pass

    if os.path.exists(CHECKPOINT_DIR):
        for file_path in glob.glob(os.path.join(CHECKPOINT_DIR, "*.csv")):
            try:
                os.remove(file_path)
            except Exception:
                pass


def reset_analysis_state():
    st.session_state.analysis_running = False
    st.session_state.analysis_paused = False
    st.session_state.analysis_done = False
    st.session_state.analysis_raw_df = None
    st.session_state.analysis_processed_parts = []
    st.session_state.analysis_remaining_df = None
    st.session_state.analysis_total_reviews = 0
    st.session_state.analysis_existing_processed = 0
    st.session_state.analysis_processed_count = 0
    st.session_state.analysis_skipped_count = 0
    st.session_state.analysis_app_id = None
    st.session_state.analysis_source_id = None
    st.session_state.download_checkpoint_ready = False
    st.session_state.analysis_pause_requested = False


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
    st.session_state.run_analysis_from_scrape = False
    reset_analysis_state()


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


def restart_app_state():
    keys_to_clear = [
        "checked_app_id",
        "checked_app_title",
        "checked_app_icon",
        "total_reviews_reported",
        "corrected_total_reviews",
        "scrapable_reviews",
        "estimated_time_low",
        "estimated_time_high",
        "ready_to_scrape",
        "scraping",
        "scrape_done",
        "cancel_requested",
        "all_reviews",
        "batch_index",
        "scrape_start",
        "app_input",
        "loading_app",
        "auto_download_ready",
        "current_plan_index",
        "current_token",
        "current_query_rounds",
        "current_stagnant_rounds",
        "seen_review_ids",
        "uploaded_raw_file",
        "run_analysis_from_scrape",
        "analysis_running",
        "analysis_paused",
        "analysis_done",
        "analysis_raw_df",
        "analysis_processed_parts",
        "analysis_remaining_df",
        "analysis_total_reviews",
        "analysis_existing_processed",
        "analysis_processed_count",
        "analysis_skipped_count",
        "analysis_app_id",
        "analysis_source_id",
        "download_checkpoint_ready",
        "analysis_pause_requested",
    ]

    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]

    st.rerun()


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
        load_progress = st.progress(0)
        load_status = st.empty()

        load_status.text("Loading app details...")
        load_progress.progress(20)

        result = app(app_id, lang="en", country="ph")
        reported_total_reviews = int(result.get("reviews", 0) or 0)

        load_progress.progress(100)
        load_status.text("App details loaded.")
        time.sleep(0.3)

        load_progress.empty()
        load_status.empty()

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
        review_at = row.get("at")
        is_post_2024 = True
        if review_at and hasattr(review_at, 'year'):
            if review_at.year < 2024:
                is_post_2024 = False
        
        if rid and rid not in st.session_state.seen_review_ids and is_post_2024:
            st.session_state.seen_review_ids.add(rid)
            st.session_state.all_reviews.append(row)
            added += 1
    return added


def map_roberta_label(label):
    label = str(label).lower()
    if "positive" in label or label == "label_2":
        return "positive"
    elif "neutral" in label or label == "label_1":
        return "neutral"
    else:
        return "negative"


def get_sentiment_score(mapped_label, confidence_score):
    if mapped_label == "positive":
        return float(confidence_score)
    elif mapped_label == "negative":
        return -float(confidence_score)
    else:
        return 0.0


def prepare_analysis_df(raw_df):
    df = raw_df.copy()

    df.drop(
        ["userName", "userImage", "replyContent", "repliedAt", "appVersion"],
        axis=1,
        inplace=True,
        errors="ignore"
    )

    if "content" not in df.columns:
        st.error("The uploaded raw file does not contain a 'content' column.")
        st.stop()

    if "reviewId" not in df.columns:
        st.error("The uploaded raw file does not contain a 'reviewId' column.")
        st.stop()

    if "at" in df.columns and "reviewDate" not in df.columns:
        df.rename(columns={"at": "reviewDate"}, inplace=True)

    if "reviewDate" in df.columns:
        df["reviewDate"] = pd.to_datetime(df["reviewDate"], errors="coerce")

    if "score" not in df.columns:
        st.error("The uploaded raw file does not contain a 'score' column.")
        st.stop()

    df["score"] = pd.to_numeric(df["score"], errors="coerce")

    df = df[df["content"].notna()]
    df = df[df["content"].astype(str).str.strip() != ""]
    df = df.dropna(subset=["score"])

    df["reviewId"] = df["reviewId"].astype(str)

    df["clean_text"] = df["content"].astype(str).str.lower()
    df["clean_text"] = df["clean_text"].str.replace(r"http\S+", "", regex=True)
    df["clean_text"] = df["clean_text"].str.replace(r"[^a-zA-Z\s]", "", regex=True)

    if "reviewDate" in df.columns:
        df = df.dropna(subset=["reviewDate"])
        df["month"] = df["reviewDate"].dt.to_period("M").astype(str)

    return df


def prepare_checkpoint_df(checkpoint_df):
    if checkpoint_df is None:
        return None

    df = checkpoint_df.copy()

    if "reviewId" not in df.columns:
        st.error("The checkpoint file does not contain a 'reviewId' column.")
        st.stop()

    df["reviewId"] = df["reviewId"].astype(str)
    return df


def get_checkpoint_path(source_id):
    safe_source_id = safe_name(source_id)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    return os.path.join(CHECKPOINT_DIR, f"{safe_source_id}_latest_checkpoint.csv")


def save_latest_checkpoint(df, source_id):
    checkpoint_path = get_checkpoint_path(source_id)
    df.to_csv(checkpoint_path, index=False)
    return checkpoint_path


def load_latest_checkpoint(source_id):
    checkpoint_path = get_checkpoint_path(source_id)
    if os.path.exists(checkpoint_path):
        try:
            return pd.read_csv(checkpoint_path)
        except Exception:
            return None
    return None


def delete_latest_checkpoint(source_id):
    checkpoint_path = get_checkpoint_path(source_id)
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)


def start_analysis(raw_df, checkpoint_df=None, source_id=None):
    df = prepare_analysis_df(raw_df)

    if source_id is None:
        source_id = get_analysis_source_id(
            app_id=st.session_state.get("checked_app_id"),
            raw_filename=None
        )

    if checkpoint_df is None:
        checkpoint_df = load_latest_checkpoint(source_id)

    checkpoint_df = prepare_checkpoint_df(checkpoint_df)

    existing_processed = 0
    processed_parts = []

    if checkpoint_df is not None and not checkpoint_df.empty:
        done_ids = set(checkpoint_df["reviewId"].dropna().astype(str))
        remaining_df = df[~df["reviewId"].isin(done_ids)].copy()
        processed_parts.append(checkpoint_df)
        existing_processed = len(checkpoint_df)
    else:
        remaining_df = df.copy()

    st.session_state.analysis_running = True
    st.session_state.analysis_paused = False
    st.session_state.analysis_done = False
    st.session_state.analysis_raw_df = df
    st.session_state.analysis_processed_parts = processed_parts
    st.session_state.analysis_remaining_df = remaining_df
    st.session_state.analysis_total_reviews = len(df)
    st.session_state.analysis_existing_processed = existing_processed
    st.session_state.analysis_processed_count = 0
    st.session_state.analysis_skipped_count = 0
    st.session_state.analysis_app_id = st.session_state.get("checked_app_id")
    st.session_state.analysis_source_id = source_id
    st.session_state.download_checkpoint_ready = False
    st.session_state.analysis_pause_requested = False


def process_analysis_batch():
    if not st.session_state.analysis_running:
        return

    remaining_df = st.session_state.analysis_remaining_df

    if remaining_df is None or remaining_df.empty:
        st.session_state.analysis_running = False
        st.session_state.analysis_done = True
        return

    sentiment_model = load_roberta_model()

    batch_df = remaining_df.iloc[:ROBERTA_BATCH_SIZE].copy()
    next_remaining_df = remaining_df.iloc[ROBERTA_BATCH_SIZE:].copy()

    batch_reviews = batch_df["content"].astype(str).tolist()
    batch_rows = []

    try:
        batch_results = sentiment_model(
            batch_reviews,
            truncation=True,
            max_length=ROBERTA_MAX_LENGTH
        )

        for row_dict, r in zip(batch_df.to_dict("records"), batch_results):
            raw_label = r["label"]
            raw_score = r["score"]
            mapped_label = map_roberta_label(raw_label)
            signed_score = get_sentiment_score(mapped_label, raw_score)

            row_dict["roberta_label"] = raw_label
            row_dict["roberta_score"] = raw_score
            row_dict["sentiment"] = mapped_label
            row_dict["sentiment_score"] = signed_score

            batch_rows.append(row_dict)
        st.session_state.analysis_processed_count += len(batch_df)

    except Exception:
        for row_dict, review_text in zip(batch_df.to_dict("records"), batch_reviews):
            try:
                single_result = sentiment_model(
                    [review_text],
                    truncation=True,
                    max_length=ROBERTA_MAX_LENGTH
                )[0]

                raw_label = single_result["label"]
                raw_score = single_result["score"]
                mapped_label = map_roberta_label(raw_label)
                signed_score = get_sentiment_score(mapped_label, raw_score)

                row_dict["roberta_label"] = raw_label
                row_dict["roberta_score"] = raw_score
                row_dict["sentiment"] = mapped_label
                row_dict["sentiment_score"] = signed_score

            except Exception:
                row_dict["roberta_label"] = "SKIPPED"
                row_dict["roberta_score"] = None
                row_dict["sentiment"] = "skipped"
                row_dict["sentiment_score"] = None
                st.session_state.analysis_skipped_count += 1

            batch_rows.append(row_dict)
        st.session_state.analysis_processed_count += len(batch_df)

    st.session_state.analysis_processed_parts.append(pd.DataFrame(batch_rows))
    st.session_state.analysis_remaining_df = next_remaining_df

    overall_done = (
        st.session_state.analysis_existing_processed
        + st.session_state.analysis_processed_count
    )

    if overall_done % CHECKPOINT_INTERVAL == 0 or st.session_state.analysis_remaining_df.empty:
        current_checkpoint_df = pd.concat(
            st.session_state.analysis_processed_parts,
            ignore_index=True
        )
        save_latest_checkpoint(current_checkpoint_df, st.session_state.analysis_source_id)

    if st.session_state.analysis_pause_requested:
        current_checkpoint_df = pd.concat(
            st.session_state.analysis_processed_parts,
            ignore_index=True
        )
        save_latest_checkpoint(current_checkpoint_df, st.session_state.analysis_source_id)
        st.session_state.analysis_running = False
        st.session_state.analysis_paused = True
        st.session_state.analysis_pause_requested = False
        st.session_state.download_checkpoint_ready = True
        return

    if st.session_state.analysis_remaining_df.empty:
        st.session_state.analysis_running = False
        st.session_state.analysis_done = True


def show_interactive_word_list(category_name, dataframe, stopwords):
    st.subheader(f"Top 200 Words ({category_name})")
    
    if "dynamic_excluded_words" not in st.session_state:
        st.session_state.dynamic_excluded_words = set()
    
    if "word_classifications" not in st.session_state:
        st.session_state.word_classifications = {}

    tam_categories = ["None", "Perceived Usefulness", "Perceived Ease of Use", "Subjective Norm", "Network Effects"]

    text = " ".join(dataframe["content"].dropna().astype(str))
    words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
    
    all_excluded = stopwords.union(st.session_state.dynamic_excluded_words)
    filtered_words = [w for w in words if w not in all_excluded and len(w) > 2]
    word_counts = Counter(filtered_words).most_common(200)

    for word, count in word_counts:
        cols = st.columns([2, 1, 3, 1])
        cols[0].write(f"**{word}** ({count})")
        if cols[1].button("➖", key=f"remove_{category_name}_{word}"):
            st.session_state.dynamic_excluded_words.add(word)
            st.rerun()
            
        current_tag = st.session_state.word_classifications.get(word, "None")
        new_tag = cols[2].selectbox(
            "Tag TAM Model", options=tam_categories,
            index=tam_categories.index(current_tag),
            key=f"tag_{category_name}_{word}", label_visibility="collapsed"
        )
        if new_tag != current_tag:
            st.session_state.word_classifications[word] = new_tag


def show_analysis_results(df_valid, checkpoint_df=None):
    st.title("Analysis Dashboard")

    try:
        df_valid = df_valid.copy()
        skipped_count = (df_valid["sentiment"] == "skipped").sum() if "sentiment" in df_valid.columns else 0
        df_valid = df_valid[df_valid["sentiment"] != "skipped"].copy()

        if df_valid.empty:
            st.error("No reviews were successfully processed by RoBERTa.")
            st.stop()

        def actual_label(score):
            if score >= 4: return "positive"
            elif score <= 2: return "negative"
            else: return "neutral"

        df_valid["actual_label"] = df_valid["score"].apply(actual_label)
        df_valid["reviewWordCount"] = df_valid["content"].astype(str).apply(lambda x: len(x.split()))

        if "reviewDate" in df_valid.columns:
            df_valid["reviewDate"] = pd.to_datetime(df_valid["reviewDate"], errors="coerce")
            df_valid = df_valid.dropna(subset=["reviewDate"])
            df_valid["month_period"] = df_valid["reviewDate"].dt.to_period("M")
            df_valid["month"] = df_valid["month_period"].astype(str)

        st.subheader("Summary")
        col1, col2 = st.columns(2)
        sent_summary = df_valid.groupby("sentiment").agg(total_count=("sentiment", "count")).reset_index()
        sent_summary["weighted"] = (sent_summary["total_count"] / sent_summary["total_count"].sum() * 100).round(2)
        with col1:
            st.markdown("### Sentiment Summary")
            st.dataframe(sent_summary.sort_values(by="weighted", ascending=False), use_container_width=True)

        class_summary = df_valid.groupby("actual_label").agg(total_count=("actual_label", "count")).reset_index()
        class_summary["weighted"] = (class_summary["total_count"] / class_summary["total_count"].sum() * 100).round(2)
        with col2:
            st.markdown("### Classified Summary")
            st.dataframe(class_summary.sort_values(by="weighted", ascending=False), use_container_width=True)

        if "reviewDate" in df_valid.columns:
            monthly_summary = df_valid.groupby(["month_period", "month"]).agg(avg_rating=("score", "mean"), review_count=("score", "count")).reset_index()
            monthly_summary["avg_rating"] = monthly_summary["avg_rating"].round(2)
            monthly_summary = monthly_summary[monthly_summary["review_count"] >= 5]
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Top 10 Highest Rated Months")
                st.dataframe(monthly_summary.sort_values(by="avg_rating", ascending=False).head(10)[["month", "avg_rating", "review_count"]], use_container_width=True)
            with col2:
                st.subheader("Top 10 Lowest Rated Months")
                st.dataframe(monthly_summary.sort_values(by="avg_rating", ascending=True).head(10)[["month", "avg_rating", "review_count"]], use_container_width=True)

        st.subheader("Sentiment")
        summary = df_valid["sentiment"].value_counts().reset_index()
        summary.columns = ["sentiment", "total_count"]
        fig_bar = px.bar(summary, x="sentiment", y="total_count", text="total_count", title="Count of Sentiment")
        fig_pie = px.pie(summary, names="sentiment", values="total_count", title="Sentiment Distribution")
        heatmap = df_valid.groupby(["month", "sentiment"]).size().reset_index(name="count")
        pivot = heatmap.pivot(index="sentiment", columns="month", values="count").fillna(0)
        fig_heatmap = px.imshow(pivot, text_auto=True, aspect="auto", title="Monthly Sentiment Heatmap")
        c1, c2, c3 = st.columns(3)
        c1.plotly_chart(fig_bar, use_container_width=True)
        c2.plotly_chart(fig_pie, use_container_width=True)
        c3.plotly_chart(fig_heatmap, use_container_width=True)

        st.subheader("Classified Summary")
        actual_summary = df_valid["actual_label"].value_counts().reset_index()
        actual_summary.columns = ["actual_label", "total_count"]
        fig_actual_bar = px.bar(actual_summary, x="actual_label", y="total_count", text="total_count", title="Count of Classified Labels")
        fig_actual_pie = px.pie(actual_summary, names="actual_label", values="total_count", title="Classified Label Distribution")
        actual_heatmap = df_valid.groupby(["month", "actual_label"]).size().reset_index(name="count")
        actual_pivot = actual_heatmap.pivot(index="actual_label", columns="month", values="count").fillna(0)
        fig_actual_heatmap = px.imshow(actual_pivot, text_auto=True, aspect="auto", title="Monthly Classified Heatmap")
        c1, c2, c3 = st.columns(3)
        c1.plotly_chart(fig_actual_bar, use_container_width=True)
        c2.plotly_chart(fig_actual_pie, use_container_width=True)
        c3.plotly_chart(fig_actual_heatmap, use_container_width=True)

        st.subheader("Review Score Over Time")
        rating_trend = df_valid.groupby("month")["score"].mean().reset_index()
        fig_rating = px.line(rating_trend, x="month", y="score", title="Average Review Score Over Time")
        volume = df_valid.groupby("month").size().reset_index(name="count")
        fig_volume = px.line(volume, x="month", y="count", title="Review Volume Over Time")
        avg_sent = df_valid.groupby("month")["sentiment_score"].mean().reset_index()
        fig_avg = px.line(avg_sent, x="month", y="sentiment_score", title="Average Sentiment Score Over Time")
        c1, c2, c3 = st.columns(3)
        c1.plotly_chart(fig_rating, use_container_width=True)
        c2.plotly_chart(fig_volume, use_container_width=True)
        c3.plotly_chart(fig_avg, use_container_width=True)

        st.subheader("Review Length by Rating")
        col1, col2 = st.columns(2)
        col1.plotly_chart(px.box(df_valid, x="score", y="reviewWordCount", title="Review Length by Rating"), use_container_width=True)
        col2.plotly_chart(px.scatter(df_valid, x="score", y="reviewWordCount", color="sentiment", title="Review Length vs Rating", opacity=0.6), use_container_width=True)

        st.subheader("Predicted vs Actual")
        cm_df = pd.crosstab(df_valid["actual_label"], df_valid["sentiment"], rownames=["Actual"], colnames=["Predicted"])
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Confusion Matrix (Table)")
            st.dataframe(cm_df, use_container_width=True)
        labels = ["positive", "neutral", "negative"]
        report = []
        for label in labels:
            tp = ((df_valid["actual_label"] == label) & (df_valid["sentiment"] == label)).sum()
            fp = ((df_valid["actual_label"] != label) & (df_valid["sentiment"] == label)).sum()
            fn = ((df_valid["actual_label"] == label) & (df_valid["sentiment"] != label)).sum()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            report.append({"label": label, "precision": round(precision, 3), "recall": round(recall, 3), "f1_score": round(f1, 3)})
        with col2:
            st.markdown("### Classification Report")
            st.dataframe(pd.DataFrame(report), use_container_width=True)
        c1, c2 = st.columns(2)
        c1.plotly_chart(px.imshow(cm_df, text_auto=True, aspect="auto", title="Confusion Matrix"), use_container_width=True)
        rating_sentiment = df_valid.groupby("score")["sentiment_score"].mean().reset_index()
        c2.plotly_chart(px.line(rating_sentiment, x="score", y="sentiment_score", title="Average Sentiment Score per Rating"), use_container_width=True)
        st.write(f"Model Accuracy: {df_valid['actual_label'] == df_valid['sentiment'].mean():.2%}")

        from scipy.stats import spearmanr, chi2_contingency, kruskal, linregress
        st.subheader("Inferential Statistical Analysis")
        stats_df = df_valid.dropna(subset=["score", "sentiment", "sentiment_score", "reviewDate"]).copy()
        stats_df["rating_class"] = stats_df["score"].apply(lambda s: "positive" if s in [4,5] else ("neutral" if s==3 else "negative"))
        col1, col2 = st.columns(2)
        with col1:
            cv, cp = spearmanr(stats_df["sentiment_score"], stats_df["score"])
            st.dataframe(pd.DataFrame({"Metric": ["Spearman", "p-value"], "Value": [round(cv, 4), round(cp, 6)]}), use_container_width=True)
            st.plotly_chart(px.scatter(stats_df, x="sentiment_score", y="score", trendline="ols"), use_container_width=True)
        with col2:
            chi_table = pd.crosstab(stats_df["sentiment"], stats_df["rating_class"])
            chi2, p, _, _ = chi2_contingency(chi_table)
            st.dataframe(pd.DataFrame({"Metric": ["Chi-square", "p-value"], "Value": [round(chi2, 4), round(p, 6)]}), use_container_width=True)
            st.plotly_chart(px.imshow(chi_table, text_auto=True, title="Sentiment Class vs Rating Class"), use_container_width=True)
            
        col3, col4 = st.columns(2)
        with col3:
            month_groups = [v.tolist() for _, v in stats_df.groupby("month")["sentiment_score"] if len(v) > 0]
            if len(month_groups) >= 2:
                kw_s, kw_p = kruskal(*month_groups)
                st.dataframe(pd.DataFrame({"Metric": ["KW Stat", "p-value"], "Value": [round(kw_s, 4), round(kw_p, 6)]}), use_container_width=True)
        with col4:
            slope, inter, r_val, p_val, _ = linregress(stats_df["sentiment_score"], stats_df["score"])
            st.dataframe(pd.DataFrame({"Metric": ["Slope", "R-squared", "p-value"], "Value": [round(slope, 4), round(r_val**2, 4), round(p_val, 6)]}), use_container_width=True)
            st.write(f"Equation: Rating = {round(inter, 4)} + ({round(slope, 4)} * Sentiment Score)")

        st.subheader("Top 200 Words by Sentiment")
        col1, col2 = st.columns(2)
        sw = {"the", "and", "is", "to", "of", "in", "for", "on", "with", "this", "that", "it", "my", "app", "very", "so", "but", "are", "was", "be", "have", "has", "had", "not", "at", "you", "we", "they", "i", "ang", "nag", "your", "nyo"}
        with col1: show_interactive_word_list("Positive", df_valid[df_valid["sentiment"] == "positive"], sw)
        with col2: show_interactive_word_list("Negative", df_valid[df_valid["sentiment"] == "negative"], sw)

        if st.session_state.word_classifications:
            with st.expander("View TAM Model Classification Summary"):
                st.dataframe(pd.DataFrame([{"Word": w, "TAM Category": t} for w, t in st.session_state.word_classifications.items() if t != "None"]), use_container_width=True)

        st.subheader("Processed Dataset")
        final_cols = ["reviewId", "reviewDate", "content", "score", "roberta_label", "roberta_score", "sentiment_score", "sentiment", "actual_label", "reviewWordCount"]
        display_df = df_valid[[c for c in final_cols if c in df_valid.columns]].copy()
        if "reviewDate" in display_df.columns: display_df = display_df.sort_values(by="reviewDate", ascending=False)
        st.dataframe(display_df, use_container_width=True)
        st.download_button(label="Download Processed Data", data=display_df.to_csv(index=False).encode("utf-8"), file_name="processed_reviews.csv", mime="text/csv")
        
        if st.session_state.get("analysis_source_id"): delete_latest_checkpoint(st.session_state.analysis_source_id)
    except Exception as e: st.error(f"Error: {e}")


defaults = {
    "checked_app_id": None, "checked_app_title": None, "checked_app_icon": None, "total_reviews_reported": None,
    "corrected_total_reviews": None, "scrapable_reviews": None, "estimated_time_low": None, "estimated_time_high": None,
    "ready_to_scrape": False, "scraping": False, "scrape_done": False, "cancel_requested": False, "all_reviews": [],
    "batch_index": 0, "scrape_start": None, "app_input": "", "loading_app": False, "auto_download_ready": False,
    "current_plan_index": 0, "current_token": None, "current_query_rounds": 0, "current_stagnant_rounds": 0,
    "seen_review_ids": set(), "run_analysis_from_scrape": False, "analysis_running": False, "analysis_paused": False,
    "analysis_done": False, "analysis_raw_df": None, "analysis_processed_parts": [], "analysis_remaining_df": None,
    "analysis_total_reviews": 0, "analysis_existing_processed": 0, "analysis_processed_count": 0,
    "analysis_skipped_count": 0, "analysis_app_id": None, "analysis_source_id": None,
    "download_checkpoint_ready": False, "analysis_pause_requested": False, "dynamic_excluded_words": set(), "word_classifications": {},
}
for k, v in defaults.items():
    if k not in st.session_state: st.session_state[k] = v

top_col1, top_col2 = st.columns([5, 1])
with top_col2:
    if st.button("Restart", use_container_width=True): clear_saved_checkpoints(); restart_app_state()

tab_find_app, tab_load_files = st.tabs(["Find an App", "Load Files"])

with tab_find_app:
    l_col, r_col = st.columns([1.4, 1])
    with l_col:
        st.text_input("Google Play link or package ID", key="app_input")
        b1, b2, b3, b4 = st.columns(4)
        with b1:
            if not st.session_state.scraping:
                if st.button("Load App", disabled=st.session_state.loading_app, use_container_width=True): load_app_details()
            else: st.button("Load App", disabled=True, use_container_width=True)
        with b2:
            if st.session_state.checked_app_id and not st.session_state.scraping and not st.session_state.scrape_done:
                if st.button("Scrape", use_container_width=True):
                    reset_scrape_state(); st.session_state.ready_to_scrape = True; st.session_state.scraping = True; st.session_state.scrape_start = time.time(); st.rerun()
            else: st.button("Scrape", disabled=True, use_container_width=True)
        with b3:
            if st.session_state.scrape_done and st.session_state.all_reviews:
                if st.session_state.analysis_running and not st.session_state.analysis_paused:
                    if st.button("Pause", use_container_width=True): st.session_state.analysis_pause_requested = True; st.rerun()
                elif st.session_state.download_checkpoint_ready:
                    if st.button("Resume", use_container_width=True): st.session_state.analysis_running = True; st.session_state.analysis_paused = False; st.session_state.download_checkpoint_ready = False; st.rerun()
                else:
                    if st.button("Analyze", use_container_width=True): start_analysis(pd.DataFrame(st.session_state.all_reviews)); st.rerun()
            else: st.button("Analyze", disabled=True, use_container_width=True)
        with b4:
            if st.session_state.scrape_done and st.session_state.all_reviews:
                st.download_button("Save", data=pd.DataFrame(st.session_state.all_reviews).to_csv(index=False).encode("utf-8"), file_name=f"{safe_name(st.session_state.checked_app_id)}_reviews.csv", mime="text/csv", use_container_width=True)
            else: st.button("Save", disabled=True, use_container_width=True)
    with r_col:
        if st.session_state.checked_app_id:
            i1, i2 = st.columns([1, 3])
            if st.session_state.checked_app_icon: i1.image(st.session_state.checked_app_icon, width=80)
            with i2:
                if st.session_state.scraping:
                    cur = len(st.session_state.all_reviews)
                    st.caption(f"Scraped Reviews: {cur:,}"); st.progress(min(cur / max(st.session_state.corrected_total_reviews or 1, 1), 1.0))
                elif st.session_state.scrape_done:
                    st.caption(f"Scraped Reviews: {len(st.session_state.all_reviews):,}"); st.markdown('<div style="padding:4px 8px;background-color:#e8f5e9;color:#2e7d32;border-radius:6px;font-size:12px;margin-top:4px;">Scraping complete</div>', unsafe_allow_html=True)

with tab_load_files:
    up_col, ac_col = st.columns([2, 1])
    with up_col: uploaded_raw_file = st.file_uploader("Raw Data", type=["csv"], key="uploaded_raw_file")
    with ac_col:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        lab = "Pause" if st.session_state.analysis_running and not st.session_state.analysis_paused else ("Resume" if st.session_state.analysis_paused or st.session_state.download_checkpoint_ready else "Analyze")
        if st.button(lab):
            if lab == "Analyze" and uploaded_raw_file: start_analysis(pd.read_csv(uploaded_raw_file)); st.rerun()
            elif lab == "Pause": st.session_state.analysis_pause_requested = True; st.rerun()
            elif lab == "Resume": st.session_state.analysis_running = True; st.session_state.analysis_paused = False; st.session_state.download_checkpoint_ready = False; st.rerun()

if st.session_state.scraping and not st.session_state.cancel_requested:
    item = get_current_plan_item()
    if item is None: st.session_state.scraping = False; st.session_state.scrape_done = True; st.rerun()
    try:
        batch, token = reviews(st.session_state.checked_app_id, lang="en", country=item["country"], sort=item["sort_value"], count=COUNT_PER_BATCH, continuation_token=st.session_state.current_token)
        added = add_new_reviews(batch)
        st.session_state.current_token = token
        if len(batch) == 0 or added == 0 or token == st.session_state.current_token: st.session_state.current_stagnant_rounds += 1
        else: st.session_state.current_stagnant_rounds = 0
        if token is None or st.session_state.current_stagnant_rounds >= 2 or st.session_state.current_query_rounds >= SCRAPE_MAX_ROUNDS_PER_QUERY:
            st.session_state.current_plan_index += 1; st.session_state.current_token = None; st.session_state.current_query_rounds = 0; st.session_state.current_stagnant_rounds = 0
        st.rerun()
    except Exception as e: st.error(f"Scraping error: {e}"); st.session_state.scraping = False

if st.session_state.analysis_running or st.session_state.download_checkpoint_ready:
    overall_done = st.session_state.analysis_existing_processed + st.session_state.analysis_processed_count
    total = st.session_state.analysis_total_reviews
    st.subheader("Analysis Progress")
    st.progress(overall_done / total if total > 0 else 0)
    st.write(f"{overall_done} out of {total} processed. {st.session_state.analysis_skipped_count} skipped.")
    if st.session_state.analysis_running and not st.session_state.analysis_paused:
        process_analysis_batch(); st.rerun()

if st.session_state.analysis_done:
    df_final = pd.concat(st.session_state.analysis_processed_parts, ignore_index=True)
    st.session_state.analysis_done = False; st.session_state.download_checkpoint_ready = False; show_analysis_results(df_final)