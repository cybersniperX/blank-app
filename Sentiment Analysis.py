import streamlit as st
import pandas as pd
import nltk
import plotly.express as px
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.metrics import confusion_matrix, classification_report

st.set_page_config(page_title="Sentiment Analysis", layout="wide")
st.title("File Upload")

nltk.download("vader_lexicon")
sia = SentimentIntensityAnalyzer()

uploaded_file = st.file_uploader(
    "Upload file",
    type=["xlsx", "xls", "csv"]
)

if uploaded_file is not None:
    try:
        # READ FILE
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.success("File uploaded successfully")

        # DROP UNNECESSARY COLUMNS
        df.drop(
            ["userName", "userImage", "reviewId", "replyContent", "repliedAt", "appVersion"],
            axis=1,
            inplace=True,
            errors="ignore"
        )

        # RENAME AND CLEAN DATE
        if "at" in df.columns:
            df.rename(columns={"at": "reviewDate"}, inplace=True)

        if "reviewDate" in df.columns:
            df["reviewDate"] = pd.to_datetime(df["reviewDate"], errors="coerce")
            df["month_period"] = df["reviewDate"].dt.to_period("M")
            df["month"] = df["month_period"].astype(str)

        # CLEAN reviewCreatedVersion IF IT EXISTS
        if "reviewCreatedVersion" in df.columns:
            df["reviewCreatedVersion"] = df["reviewCreatedVersion"].replace("None", None).fillna("0.0.0")

        # CHECK REQUIRED COLUMN
        if "content" not in df.columns:
            st.error("The uploaded file does not contain a 'content' column.")
            st.stop()

        # SENTIMENT ANALYSIS
        df["sentiment_score"] = df["content"].apply(
            lambda x: sia.polarity_scores(str(x))["compound"]
        )

        def classify_sentiment(score):
            if score >= 0.05:
                return "positive"
            elif score <= -0.05:
                return "negative"
            else:
                return "neutral"

        df["sentiment"] = df["sentiment_score"].apply(classify_sentiment)

        # SUMMARY TABLE
        summary = df["sentiment"].value_counts().reset_index()
        summary.columns = ["sentiment", "total_count"]
        summary["weighted"] = summary["total_count"] / summary["total_count"].sum()

        st.title("Sentiment Analysis Dashboard")

        st.subheader("Preview")
        st.dataframe(df.head())

        st.subheader("Summary")
        st.dataframe(summary)

        # MODEL EVALUATION IF actual_sentiment EXISTS
        if "actual_sentiment" in df.columns:
            st.subheader("Model Evaluation")

            eval_df = df.dropna(subset=["actual_sentiment", "sentiment"]).copy()
            eval_df["actual_sentiment"] = eval_df["actual_sentiment"].astype(str).str.strip().str.lower()
            eval_df["sentiment"] = eval_df["sentiment"].astype(str).str.strip().str.lower()

            valid_labels = ["negative", "neutral", "positive"]
            eval_df = eval_df[
                eval_df["actual_sentiment"].isin(valid_labels) &
                eval_df["sentiment"].isin(valid_labels)
            ]

            if len(eval_df) > 0:
                accuracy = (eval_df["sentiment"] == eval_df["actual_sentiment"]).mean()
                st.write("Accuracy:", round(accuracy * 100, 2), "%")

                cm = confusion_matrix(
                    eval_df["actual_sentiment"],
                    eval_df["sentiment"],
                    labels=valid_labels
                )

                cm_df = pd.DataFrame(
                    cm,
                    index=["Actual Negative", "Actual Neutral", "Actual Positive"],
                    columns=["Pred Negative", "Pred Neutral", "Pred Positive"]
                )

                st.subheader("Confusion Matrix")
                st.dataframe(cm_df, use_container_width=True)

                st.subheader("Classification Report")
                report = classification_report(
                    eval_df["actual_sentiment"],
                    eval_df["sentiment"],
                    labels=valid_labels,
                    zero_division=0
                )
                st.text(report)
            else:
                st.warning("The 'actual_sentiment' column exists, but it does not contain usable values like negative, neutral, or positive.")

        else:
            st.info("No 'actual_sentiment' column found, so accuracy and classification report were skipped.")

        # CONTINUE ONLY IF DATE EXISTS
        if "reviewDate" in df.columns and "score" in df.columns:
            df = df.dropna(subset=["reviewDate"]).copy()
            df["month_period"] = df["reviewDate"].dt.to_period("M")
            df["month"] = df["month_period"].astype(str)

            # monthly summary
            monthly_summary = df.groupby(["month_period", "month"]).agg(
                avg_rating=("score", "mean"),
                review_count=("score", "count")
            ).reset_index()

            monthly_summary["avg_rating"] = monthly_summary["avg_rating"].round(2)

            # OPTIONAL: filter low-volume months
            monthly_summary = monthly_summary[monthly_summary["review_count"] >= 5]

            # sort by actual month
            monthly_summary = monthly_summary.sort_values("month_period")

            # top 10 highest
            top_10_highest = monthly_summary.sort_values(
                by="avg_rating", ascending=False
            ).head(10)

            # top 10 lowest
            top_10_lowest = monthly_summary.sort_values(
                by="avg_rating", ascending=True
            ).head(10)

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Top 10 Highest Rated Months")
                st.dataframe(top_10_highest[["month", "avg_rating", "review_count"]], use_container_width=True)

            with col2:
                st.subheader("Top 10 Lowest Rated Months")
                st.dataframe(top_10_lowest[["month", "avg_rating", "review_count"]], use_container_width=True)

            st.subheader("Distribution")

            fig1 = px.bar(
                summary,
                x="sentiment",
                y="total_count",
                text="total_count",
                title="Count of Sentiment",
            )

            fig1.update_layout(
                yaxis_title="Total Count",
                xaxis_title="Sentiment"
            )

            fig2 = px.pie(
                summary,
                names="sentiment",
                values="total_count",
                title="Sentiment Distribution"
            )

            # HEATMAP
            heatmap = df.groupby(["month", "sentiment"]).size().reset_index(name="count")
            month_order = sorted(df["month"].dropna().unique().tolist())

            pivot = heatmap.pivot(index="sentiment", columns="month", values="count").fillna(0)

            # reorder columns by month
            pivot = pivot.reindex(columns=month_order)

            fig3 = px.imshow(
                pivot,
                text_auto=True,
                aspect="auto",
                title="Monthly Sentiment Heatmap"
            )

            col1, col2, col3 = st.columns(3)

            with col1:
                st.plotly_chart(fig1, use_container_width=True)

            with col2:
                st.plotly_chart(fig2, use_container_width=True)

            with col3:
                st.plotly_chart(fig3, use_container_width=True)

            st.subheader("Sentiment Score Over Time")

            df["review_length"] = df["content"].astype(str).apply(len)

            trend = df.groupby(["month_period", "month", "sentiment"]).size().reset_index(name="count")
            trend = trend.sort_values("month_period")

            fig1 = px.line(
                trend,
                x="month",
                y="count",
                color="sentiment",
                title="Sentiment Trend Over Time"
            )

            avg_trend = df.groupby(["month_period", "month"])["sentiment_score"].mean().reset_index()
            avg_trend = avg_trend.sort_values("month_period")

            fig2 = px.line(
                avg_trend,
                x="month",
                y="sentiment_score",
                title="Average Sentiment Score Over Time"
            )

            col1, col2 = st.columns(2)

            with col1:
                st.plotly_chart(fig1, use_container_width=True)

            with col2:
                st.plotly_chart(fig2, use_container_width=True)

            st.subheader("Review Score Over Time")

            rating_trend = df.groupby(["month_period", "month"])["score"].mean().reset_index()
            rating_trend = rating_trend.sort_values("month_period")

            fig1 = px.line(
                rating_trend,
                x="month",
                y="score",
                title="Average Rating Over Time"
            )

            volume = df.groupby(["month_period", "month"]).size().reset_index(name="count")
            volume = volume.sort_values("month_period")

            fig2 = px.line(
                volume,
                x="month",
                y="count",
                title="Number of Reviews Over Time"
            )

            fig3 = px.box(
                df,
                x="score",
                y="review_length",
                title="Review Length by Rating"
            )

            col1, col2, col3 = st.columns(3)

            with col1:
                st.plotly_chart(fig1, use_container_width=True)
            with col2:
                st.plotly_chart(fig2, use_container_width=True)
            with col3:
                st.plotly_chart(fig3, use_container_width=True)

            monthly = df.groupby(["month_period", "month"]).agg(
                total=("sentiment", "count"),
                positive=("sentiment", lambda x: (x == "positive").sum())
            ).reset_index()

            monthly = monthly.sort_values("month_period")
            monthly["positive_rate"] = monthly["positive"] / monthly["total"]

            fig = px.line(
                monthly,
                x="month",
                y="positive_rate",
                title="Positive Sentiment Rate Over Time"
            )

            st.plotly_chart(fig, use_container_width=True)

            combined = df.groupby(["month_period", "month"]).agg(
                review_count=("score", "count"),
                avg_rating=("score", "mean")
            ).reset_index()

            combined = combined.sort_values("month_period")

            fig = px.scatter(
                combined,
                x="review_count",
                y="avg_rating",
                title="Review Volume vs Rating",
                hover_data=["month"]
            )

            st.plotly_chart(fig, use_container_width=True)

            fig = px.bar(
                top_10_lowest,
                x="month",
                y="avg_rating",
                color="avg_rating",
                title="Top Problem Months"
            )

            st.plotly_chart(fig, use_container_width=True)

        else:
            st.warning("The file needs 'reviewDate' and 'score' columns to show the time-based charts.")

    except Exception as e:
        st.error(f"Error reading file: {e}")