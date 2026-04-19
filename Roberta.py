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
from sklearn.feature_extraction.text import CountVectorizer

# --- CONFIGURATION & MODEL LOADING ---
st.set_page_config(page_title="TAM Sentiment Analysis Console", layout="wide")
st.title("TAM Sentiment Analysis Console")

@st.cache_resource
def load_roberta_model():
    # Caching the model to prevent reloading on every interaction
    return pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-xlm-roberta-base-sentiment",
        tokenizer="cardiffnlp/twitter-xlm-roberta-base-sentiment"
    )

# --- SESSION STATE MANAGEMENT ---
# Using session state to maintain your lexicon across button clicks
defaults = {
    "all_reviews": [],
    "manual_exclusions": [],
    "tam_mappings": {},
    "checked_app_id": None,
    "ready_to_scrape": False,
    "scraping": False,
    "scrape_done": False,
    "current_token": None,
    "seen_ids": set(),
    "processed_df": None,
    "analysis_running": False
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# --- SCRAPER LOGIC WITH 2024 CUTOFF ---
def add_new_reviews(batch):
    added = 0
    hit_old_review = False
    for row in batch:
        review_date = row.get("at")
        # HARD CUTOFF: Exclude anything older than 2024
        if review_date and review_date.year < 2024:
            hit_old_review = True
            continue
            
        rid = row.get("reviewId")
        if rid and rid not in st.session_state.seen_ids:
            st.session_state.seen_ids.add(rid)
            st.session_state.all_reviews.append(row)
            added += 1
    return added, hit_old_review

def clean_text(text):
    # Preserves hyphens for Taglish morphology (e.g., nag-crash)
    text = str(text).lower()
    return re.sub(r"[^a-zA-Z\s\-]", "", text)

# --- UI TABS ---
tab_scrape, tab_lexicon = st.tabs(["1. Data Acquisition", "2. TAM Lexicon & Analysis"])

with tab_scrape:
    app_input = st.text_input("Enter Google Play App ID or Link:", placeholder="com.shopee.ph")
    
    col1, col2, col3 = st.columns(3)
    
    if col1.button("Load App Info"):
        try:
            # Basic validation
            if "id=" in app_input:
                app_id = parse_qs(urlparse(app_input).query).get("id", [None])[0]
            else:
                app_id = app_input
            
            result = app(app_id, lang="en", country="ph")
            st.session_state.checked_app_id = app_id
            st.session_state.ready_to_scrape = True
            st.success(f"Loaded: {result.get('title')}")
        except Exception as e:
            st.error(f"Error: {e}")

    if st.session_state.ready_to_scrape:
        if col2.button("Start Scraping (2024-Present)"):
            st.session_state.scraping = True
            st.rerun()

    if st.session_state.scraping:
        st.write("Scraping in progress... checking dates...")
        batch, token = reviews(
            st.session_state.checked_app_id,
            lang="en",
            country="ph",
            sort=Sort.NEWEST,
            count=200,
            continuation_token=st.session_state.current_token
        )
        added, stop_signal = add_new_reviews(batch)
        st.session_state.current_token = token
        
        if stop_signal or token is None:
            st.session_state.scraping = False
            st.session_state.scrape_done = True
            st.success(f"Scraping Complete! Total unique 2024+ reviews: {len(st.session_state.all_reviews)}")
            st.rerun()
        else:
            time.sleep(0.1)
            st.rerun()

with tab_lexicon:
    if not st.session_state.all_reviews:
        st.info("Please scrape data or load a file in Tab 1 first.")
    else:
        df_raw = pd.DataFrame(st.session_state.all_reviews)
        df_raw['clean_text'] = df_raw['content'].apply(clean_text)
        
        st.subheader("Manual TAM Classification Console")
        st.write("Assign TAM factors to the top 200 repeating words/phrases.")

        # Vectorize to find top 200 N-grams (1 and 2 words)
        vec = CountVectorizer(ngram_range=(1, 2), stop_words='english')
        X = vec.fit_transform(df_raw['clean_text'])
        word_freq = pd.DataFrame({'word': vec.get_feature_names_out(), 'n': X.sum(axis=0).A1})
        top_200 = word_freq.sort_values("n", ascending=False).head(200)

        # Filter display based on exclusions
        active_list = top_200[~top_200['word'].isin(st.session_state.manual_exclusions)]

        # --- INTERACTIVE LEXICON TABLE ---
        col_main, col_side = st.columns([2, 1])

        with col_main:
            st.markdown("#### Top Words & Bigrams")
            # Header for clarity
            h1, h2, h3 = st.columns([2, 2, 1])
            h1.caption("Word (Frequency)")
            h2.caption("TAM Tag")
            h3.caption("Action")
            
            for _, row in active_list.iterrows():
                w, n = row['word'], row['n']
                c1, c2, c3 = st.columns([2, 2, 1])
                
                c1.write(f"**{w}** ({n})")
                
                # Dynamic Dropdown for TAM Mapping
                current_tag = st.session_state.tam_mappings.get(w, "None")
                tag_options = ["None", "PEOU", "SysBen", "NetEff", "OutQual"]
                
                selected_tag = c2.selectbox(
                    "Tag", tag_options, 
                    index=tag_options.index(current_tag),
                    key=f"sel_{w}",
                    label_visibility="collapsed"
                )
                
                if selected_tag != current_tag:
                    st.session_state.tam_mappings[w] = selected_tag
                
                if c3.button("[-]", key=f"btn_rem_{w}", help="Remove from lexicon"):
                    st.session_state.manual_exclusions.append(w)
                    st.rerun()

        with col_side:
            st.markdown("#### Removed/Excluded")
            if not st.session_state.manual_exclusions:
                st.caption("No words removed yet.")
            for ex_word in st.session_state.manual_exclusions:
                e1, e2 = st.columns([3, 1])
                e1.write(f"~~{ex_word}~~")
                if e2.button("[+]", key=f"btn_add_{ex_word}", help="Restore word"):
                    st.session_state.manual_exclusions.remove(ex_word)
                    st.rerun()

        # --- THE PROCESSING ENGINE ---
        st.divider()
        if st.button("🚀 Apply Classification to All Records", use_container_width=True):
            with st.spinner("Processing thousands of records..."):
                final_df = df_raw.copy()
                
                # 1. Binary Scoring for TAM Factors
                for factor in ["peou", "sysben", "neteff", "outqual"]:
                    final_df[f"tam_{factor}"] = 0
                    
                    # Find all words mapped to this factor
                    keywords = [w for w, t in st.session_state.tam_mappings.items() 
                                if t.lower() == factor and w not in st.session_state.manual_exclusions]
                    
                    if keywords:
                        # Vectorized string search (Fast for millions of rows)
                        pattern = '|'.join([rf"\b{re.escape(k)}\b" for k in keywords])
                        final_df.loc[final_df['clean_text'].str.contains(pattern, na=False), f"tam_{factor}"] = 1

                # 2. Run RoBERTa Sentiment Analysis on the result
                model = load_roberta_model()
                st.write("Running RoBERTa Sentiment Analysis...")
                
                # Process in batches for stability
                all_results = []
                contents = final_df['content'].astype(str).tolist()
                
                # Note: Large datasets might take time on CPU
                for i in range(0, len(contents), 10):
                    batch = contents[i:i+10]
                    res = model(batch, truncation=True, max_length=128)
                    all_results.extend(res)
                
                final_df['roberta_label'] = [r['label'] for r in all_results]
                final_df['roberta_score'] = [r['score'] for r in all_results]
                
                st.session_state.processed_df = final_df
                st.success("Full Analysis Complete!")

        # --- RESULTS SUMMARY ---
        if st.session_state.processed_df is not None:
            pdf = st.session_state.processed_df
            
            st.subheader("Statistical Impact Summary")
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("PEOU Records", pdf['tam_peou'].sum())
            s2.metric("Sys Benefit Records", pdf['tam_sysben'].sum())
            s3.metric("Network Effect Records", pdf['tam_neteff'].sum())
            s4.metric("Out Quality Records", pdf['tam_outqual'].sum())
            
            st.subheader("Final Processed Data (Preview)")
            st.dataframe(pdf, use_container_width=True)
            
            # Export
            csv_data = pdf.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download Final Processed Dataset (.csv)",
                data=csv_data,
                file_name="tam_final_analysis.csv",
                mime="text/csv",
                use_container_width=True
            )