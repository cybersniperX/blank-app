import time
import pandas as pd
import streamlit as st
from urllib.parse import urlparse, parse_qs
from google_play_scraper import app, reviews, Sort

st.set_page_config(page_title="Google Play Scraper", layout="wide")
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
            df = pd.DataFrame(st.session_state.all_reviews)
            csv_bytes = df.to_csv(index=False).encode("utf-8")
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



if st.session_state.cancel_requested:
    partial_total = len(st.session_state.all_reviews)
    st.session_state.corrected_total_reviews = max(
        st.session_state.total_reviews_reported or 0,
        partial_total
    )
    st.warning(f"Scraping cancelled. Partial unique reviews extracted: {partial_total:,}")