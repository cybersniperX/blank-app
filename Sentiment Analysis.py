import streamlit as st
import pandas as pd
import plotly.express as px
import nltk

from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.metrics import confusion_matrix, classification_report

st.set_page_config(page_title="Sentiment Analysis Dashboard", layout="wide")
st.title("Analysis Dashboard")

nltk.download("vader_lexicon")
sia = SentimentIntensityAnalyzer()

uploaded_file = st.file_uploader(
    "Upload file",
    type=["csv", "xlsx", "xls"]
)

if uploaded_file is not None:
    try:
        # READ FILE
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.success("File uploaded successfully")

        # DROP UNNECESSARY COLUMNS BUT KEEP DATE COLUMN
        df.drop(
            ["userName", "userImage", "reviewId", "replyContent", "repliedAt", "appVersion"],
            axis=1,
            inplace=True,
            errors="ignore"
        )

        # RENAME DATE COLUMN
        if "at" in df.columns:
            df.rename(columns={"at": "reviewdate"}, inplace=True)

        # CLEAN DATE
        if "reviewdate" in df.columns:
            df["reviewdate"] = pd.to_datetime(df["reviewdate"], errors="coerce")
            df = df.dropna(subset=["reviewdate"])
            df["month"] = df["reviewdate"].dt.to_period("M").astype(str)

        # CLEAN VERSION COLUMN
        if "reviewCreatedVersion" in df.columns:
            df["reviewCreatedVersion"] = (
                df["reviewCreatedVersion"]
                .replace("None", None)
                .fillna("0.0.0")
            )

        # CLEAN SCORE
        if "score" in df.columns:
            df["score"] = pd.to_numeric(df["score"], errors="coerce")
            df = df.dropna(subset=["score"])

        # SENTIMENT SCORE
        df["sentiment_score"] = df["content"].astype(str).apply(
            lambda x: sia.polarity_scores(x)["compound"]
        )

        # SENTIMENT LABEL
        def classify_sentiment(score):
            if score >= 0.05:
                return "positive"
            elif score <= -0.05:
                return "negative"
            else:
                return "neutral"

        df["sentiment"] = df["sentiment_score"].apply(classify_sentiment)

        # ACTUAL LABEL FROM STAR RATING
        def actual_label(score):
            if score >= 4:
                return "positive"
            elif score <= 2:
                return "negative"
            else:
                return "neutral"

        df["actual_label"] = df["score"].apply(actual_label)

        # REVIEW LENGTH
        df["review_length"] = df["content"].astype(str).apply(lambda x: len(x.split()))

        # SUMMARY TABLE
        summary = df["sentiment"].value_counts().reset_index()
        summary.columns = ["sentiment", "total_count"]
        summary["weighted"] = (summary["total_count"] / summary["total_count"].sum()).round(4)

        # PREVIEW
        st.subheader("Preview")
        st.dataframe(df.head(), use_container_width=True)

        # SUMMARY
        st.subheader("Summary")
        st.dataframe(summary, use_container_width=True)

        # DISTRIBUTION
        st.subheader("Distribution")

        fig_bar = px.bar(
            summary,
            x="sentiment",
            y="total_count",
            text="total_count",
            title="Count of Sentiment"
        )
        fig_bar.update_traces(textposition="outside")
        fig_bar.update_layout(
            xaxis_title="Sentiment",
            yaxis_title="Total Count"
        )

        fig_pie = px.pie(
            summary,
            names="sentiment",
            values="total_count",
            title="Sentiment Distribution"
        )

        if "month" in df.columns:
            heatmap = df.groupby(["month", "sentiment"]).size().reset_index(name="count")
            pivot = heatmap.pivot(index="sentiment", columns="month", values="count").fillna(0)

            fig_heatmap = px.imshow(
                pivot,
                text_auto=True,
                aspect="auto",
                title="Monthly Sentiment Heatmap"
            )
        else:
            fig_heatmap = None

        if fig_heatmap is not None:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.plotly_chart(fig_bar, use_container_width=True)
            with col2:
                st.plotly_chart(fig_pie, use_container_width=True)
            with col3:
                st.plotly_chart(fig_heatmap, use_container_width=True)
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(fig_bar, use_container_width=True)
            with col2:
                st.plotly_chart(fig_pie, use_container_width=True)

        # SENTIMENT SCORE OVER TIME
        if "month" in df.columns:
            st.subheader("Sentiment Score Over Time")

            trend = df.groupby(["month", "sentiment"]).size().reset_index(name="count")

            fig_trend = px.line(
                trend,
                x="month",
                y="count",
                color="sentiment",
                title="Sentiment Trend Over Time"
            )

            avg_trend = df.groupby("month")["sentiment_score"].mean().reset_index()

            fig_avg_sent = px.line(
                avg_trend,
                x="month",
                y="sentiment_score",
                title="Average Sentiment Score Over Time"
            )

            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(fig_trend, use_container_width=True)
            with col2:
                st.plotly_chart(fig_avg_sent, use_container_width=True)

        # REVIEW SCORE OVER TIME
        if "month" in df.columns:
            st.subheader("Review Score Over Time")

            rating_trend = df.groupby("month")["score"].mean().reset_index()

            fig_rating = px.line(
                rating_trend,
                x="month",
                y="score",
                title="Average Rating Over Time"
            )

            volume = df.groupby("month").size().reset_index(name="count")

            fig_volume = px.line(
                volume,
                x="month",
                y="count",
                title="Number of Reviews Over Time"
            )

            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(fig_rating, use_container_width=True)
            with col2:
                st.plotly_chart(fig_volume, use_container_width=True)

        # REVIEW LENGTH BY RATING
        st.subheader("Review Length by Rating")

        fig_length = px.box(
            df,
            x="score",
            y="review_length",
            title="Review Length by Rating"
        )

        st.plotly_chart(fig_length, use_container_width=True)

        # TOP 10 MONTHS
        if "month" in df.columns:
            st.subheader("Top Rated Months")

            monthly_summary = df.groupby("month").agg(
                avg_rating=("score", "mean"),
                review_count=("score", "count")
            ).reset_index()

            monthly_summary["avg_rating"] = monthly_summary["avg_rating"].round(2)

            monthly_summary_filtered = monthly_summary[monthly_summary["review_count"] >= 5].copy()

            top_10_highest = monthly_summary_filtered.sort_values(
                by="avg_rating",
                ascending=False
            ).head(10)

            top_10_lowest = monthly_summary_filtered.sort_values(
                by="avg_rating",
                ascending=True
            ).head(10)

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Top 10 Highest Rated Months")
                st.dataframe(top_10_highest, use_container_width=True)
            with col2:
                st.subheader("Top 10 Lowest Rated Months")
                st.dataframe(top_10_lowest, use_container_width=True)

        # PREDICTED VS ACTUAL LABELS
        st.subheader("Predicted vs Actual Labels")

        report = classification_report(
            df["actual_label"],
            df["sentiment"],
            output_dict=True,
            zero_division=0
        )

        report_df = pd.DataFrame(report).transpose().round(4)

        st.subheader("Classification Report")
        st.dataframe(report_df, use_container_width=True)

        labels = ["negative", "neutral", "positive"]
        cm = confusion_matrix(
            df["actual_label"],
            df["sentiment"],
            labels=labels
        )

        cm_df = pd.DataFrame(
            cm,
            index=[f"actual_{x}" for x in labels],
            columns=[f"pred_{x}" for x in labels]
        )

        st.subheader("Confusion Matrix")
        st.dataframe(cm_df, use_container_width=True)

        fig_cm = px.imshow(
            cm_df,
            text_auto=True,
            aspect="auto",
            title="Confusion Matrix Heatmap"
        )

        st.plotly_chart(fig_cm, use_container_width=True)

    except Exception as e:
        st.error(f"Error reading file: {e}")