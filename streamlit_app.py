import streamlit as st
import plotly.express as px
import pandas as pd
import yfinance as yf
import requests
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup

st.set_page_config(page_title="MBAN 626 - AI and Data Analytics", layout="wide")

st.title("Most Played Mobile Games")
st.markdown(
    "<p style='font-size:12px;'>"
    "The dashboard provides a clear view of the mobile gaming landscape by combining key metrics, trends, and real-time data in one place. "
    "it highlights popular games, shows publisher and player patterns, and includes console news and stock prices for industry context. "
    "for business owners, it helps identify market trends, understand consumer preferences, and make better data-driven decisions on strategy, investments, and product development."
    "</p>",
    unsafe_allow_html=True
)

# Dataset pull from Google Drive
df = pd.read_csv("https://drive.google.com/uc?export=download&id=1oN1c5cbB4K3Ld8DjsmMzmTZQzE1Sc1dB")

# Quick dataset cleansing
df.columns = df.columns.str.lower().str.replace(" ", "_")
df = df.drop(columns=["ref"], errors="ignore")

if "release_date" in df.columns:
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")

if "player_count[a]" in df.columns:
    df["player_count"] = df["player_count[a]"].astype(str).str.replace(r"[^0-9]", "", regex=True)
    df["player_count"] = pd.to_numeric(df["player_count"], errors="coerce")


# Categorize popularity
def categorize_popularity(players):
    if pd.isna(players):
        return "Unknown"
    elif players >= 500:
        return "Extremely Popular"
    elif players >= 100:
        return "Very Popular"
    else:
        return "Not Popular"


if "player_count" in df.columns:
    df["popularity_level"] = df["player_count"].apply(categorize_popularity)


# GameStats class
class GameStats:
    def __init__(self, dataframe):
        self.df = dataframe

    def total_games(self):
        return len(self.df)

    def top_games(self, n=5):
        return self.df.sort_values("player_count", ascending=False).head(n)[["game", "player_count"]]


stats = GameStats(df)

# Yahoo Finance stock data for chart/table section
tickers = {
    "Nintendo": "NTDOY",
    "Sony (PlayStation)": "SONY",
    "Microsoft (Xbox)": "MSFT"
}

stock_data = []

for company, ticker in tickers.items():
    try:
        stock = yf.Ticker(ticker)
        history = stock.history(period="1d")
        if not history.empty:
            price = history["Close"].iloc[-1]
            stock_data.append({
                "Company": company,
                "Stock Price": round(price, 2)
            })
    except Exception:
        pass

df_stock = pd.DataFrame(stock_data) if stock_data else pd.DataFrame()

# Styles
st.markdown("""
<style>
.news-box {
    height: 260px;
    overflow-y: auto;
    border: 1px solid #ddd;
    padding: 12px;
    border-radius: 10px;
    background-color: white;
    font-size: 14px;
}

.news-title {
    font-weight: 600;
    margin-bottom: 4px;
}

.news-date {
    font-size: 12px;
    color: gray;
    margin-bottom: 8px;
}

.news-divider {
    margin-top: 10px;
    margin-bottom: 10px;
    border-top: 1px solid #eee;
}

.stock-card {
    border: 1px solid #ddd;
    border-radius: 12px;
    padding: 14px;
    background-color: white;
    text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    min-height: 250px;
    display: flex;
    flex-direction: column;
    justify-content: center;
}

.stock-logo {
    height: 70px;
    object-fit: contain;
    margin-bottom: 10px;
    display: block;
    margin-left: auto;
    margin-right: auto;
}

.stock-name {
    font-size: 18px;
    font-weight: 700;
    margin-bottom: 4px;
    white-space: nowrap;
}

.stock-ticker {
    font-size: 12px;
    color: gray;
    margin-bottom: 10px;
    white-space: nowrap;
}

.stock-price {
    font-size: 24px;
    font-weight: 700;
    color: #111;
    white-space: nowrap;
}

.stock-change {
    font-size: 13px;
    margin-top: 6px;
    white-space: nowrap;
}

.positive {
    color: green;
}

.negative {
    color: red;
}
</style>
""", unsafe_allow_html=True)


# News functions
def get_rss_news(rss_url, limit=5):
    items_list = []
    try:
        response = requests.get(rss_url, timeout=10)
        response.raise_for_status()

        root = ET.fromstring(response.content)
        items = root.findall(".//item")

        for item in items[:limit]:
            title = item.find("title").text if item.find("title") is not None else "No title"
            link = item.find("link").text if item.find("link") is not None else ""
            pubdate = item.find("pubDate").text if item.find("pubDate") is not None else ""

            items_list.append({
                "title": title,
                "link": link,
                "date": pubdate
            })

    except Exception as e:
        items_list.append({
            "title": f"Could not load news: {e}",
            "link": "",
            "date": ""
        })

    return items_list


def get_nintendo_news(limit=5):
    items_list = []
    try:
        url = "https://www.nintendo.com/us/whatsnew/"
        response = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=10
        )
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        seen = set()
        count = 0

        for a in soup.find_all("a", href=True):
            href = a["href"]
            title = a.get_text(" ", strip=True)

            if title and "/us/whatsnew/" in href:
                full_link = href if href.startswith("http") else f"https://www.nintendo.com{href}"

                if full_link not in seen:
                    seen.add(full_link)
                    items_list.append({
                        "title": title,
                        "link": full_link,
                        "date": ""
                    })
                    count += 1

                if count >= limit:
                    break

        if not items_list:
            items_list.append({
                "title": "No Nintendo news items found.",
                "link": "",
                "date": ""
            })

    except Exception as e:
        items_list.append({
            "title": f"Could not load news: {e}",
            "link": "",
            "date": ""
        })

    return items_list


def render_news_box(news_items):
    html = '<div class="news-box">'
    for item in news_items:
        title = item["title"]
        link = item["link"]
        date = item["date"]

        if link:
            html += f'<div class="news-title"><a href="{link}" target="_blank">{title}</a></div>'
        else:
            html += f'<div class="news-title">{title}</div>'

        if date:
            html += f'<div class="news-date">{date}</div>'

        html += '<div class="news-divider"></div>'

    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


# Stock card functions
def get_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="2d")

        if hist.empty:
            return None

        current_price = hist["Close"].iloc[-1]
        previous_close = hist["Close"].iloc[-2] if len(hist) > 1 else current_price

        change = current_price - previous_close
        change_pct = (change / previous_close) * 100 if previous_close != 0 else 0

        return {
            "price": current_price,
            "change": change,
            "change_pct": change_pct
        }

    except Exception:
        return None


def render_stock_card(name, ticker, logo_url):
    data = get_stock_data(ticker)

    if data:
        change_class = "positive" if data["change"] >= 0 else "negative"
        change_symbol = "+" if data["change"] >= 0 else ""

        html = f"""
        <div class="stock-card">
            <img src="{logo_url}" class="stock-logo">
            <div class="stock-name">{name}</div>
            <div class="stock-ticker">{ticker}</div>
            <div class="stock-price">${data["price"]:.2f}</div>
            <div class="stock-change {change_class}">
                {change_symbol}{data["change"]:.2f} ({change_symbol}{data["change_pct"]:.2f}%)
            </div>
        </div>
        """
    else:
        html = f"""
        <div class="stock-card">
            <img src="{logo_url}" class="stock-logo">
            <div class="stock-name">{name}</div>
            <div class="stock-ticker">{ticker}</div>
            <div class="stock-price">N/A</div>
            <div class="stock-change">Could not load price</div>
        </div>
        """

    st.markdown(html, unsafe_allow_html=True)


# Summary section
st.subheader("Summary")
col1, col2, col3, col4, col5, col6 = st.columns([1.15, 1, 1, 1, 1, 1])

with col1:
    st.markdown(
        f"""
        <div style='text-align:center; padding:20px; border-radius:10px; background-color:#f5f5f5; height:260px; display:flex; flex-direction:column; justify-content:center;'>
            <p style='font-size:20px;'>Total Games Reviewed</p>
            <p style='font-size:60px; color:#003791; margin:0; font-weight:bold;'>
                {stats.total_games()}
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

# Image cards
games = [
    ("Mario Kart Tour", "https://upload.wikimedia.org/wikipedia/en/e/e0/Mario_Kart_Tour_artwork.png"),
    ("Among Us", "https://upload.wikimedia.org/wikipedia/en/9/9a/Among_Us_cover_art.jpg"),
    ("Mini World", "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQNIFfCXrs5p4HTybdzbOv4op5rSliGMrfjFxYzc_FydofEXG5uTPzYYqBBfOpVntmcIoeJ&s=10"),
    ("Sonic Dash", "https://upload.wikimedia.org/wikipedia/en/3/39/Sonic_Dash.jpg"),
    ("Dragon Ball Z: Dokkan Battle", "https://assets-prd.ignimgs.com/2022/04/23/dbzdokkan-1650673864385.jpg")
]

for i, (name, img) in enumerate(games, start=2):
    with locals()[f"col{i}"]:
        st.markdown(
            f"""
            <div style='text-align:center; padding:10px; border-radius:10px; background-color:#f9f9f9; height:260px;'>
                <img src='{img}' style='width:160px; height:160px; object-fit:contain; border-radius:10px; border:2px solid #d9d9d9; background-color:white;'>
                <p style='font-size:15px; font-weight:bold; margin-top:10px;'>{name}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

st.markdown("<br>", unsafe_allow_html=True)

# Console News + Stock Prices 
st.subheader("Latest Console News and Stock Prices")

st.markdown(
    "<p style='font-size:12px;'>"
    "API call to fetch the latest news on Playstation and Nintendo; and Current stock prices including Xbox"
    "</p>",
    unsafe_allow_html=True
)

left_col, right_col = st.columns(2, gap="large")

with left_col:
    st.markdown("### Console News")

    news_col1, news_col2 = st.columns(2, gap="medium")

    with news_col1:
        st.markdown("**PlayStation**")
        ps_news = get_rss_news("https://blog.playstation.com/feed/", limit=5)
        render_news_box(ps_news)

    with news_col2:
        st.markdown("**Nintendo**")
        nin_news = get_nintendo_news(limit=5)
        render_news_box(nin_news)

with right_col:
    st.markdown("### Current Stock Prices")

    stock_col1, stock_col2, stock_col3 = st.columns(3, gap="small")

    with stock_col1:
        render_stock_card(
            "PlayStation",
            "SONY",
            "https://www.galaxus.de/productimages/5/3/1/5/1/8/7/7/6/3/5/3/8/6/5/0/6/9/4/76427826-8bc0-4b8b-936e-94ad46b226a4.png"
        )

    with stock_col2:
        render_stock_card(
            "Nintendo",
            "NTDOY",
            "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQy2b9uDcbkDCxHLD7vrgm4wOzi61wEfP4TBA&s"
        )

    with stock_col3:
        render_stock_card(
            "Xbox",
            "MSFT",
            "https://logosandtypes.com/wp-content/uploads/2020/10/Xbox.png"
        )

st.markdown("<br>", unsafe_allow_html=True)

# Charts section
st.subheader("Charts")
chart_col1, chart_col2, chart_col3 = st.columns(3)

with chart_col1:
    if "publisher(s)" in df.columns:
        publisher_counts = df["publisher(s)"].value_counts().head(8).reset_index()
        publisher_counts.columns = ["Publisher", "Number of Games"]

        fig_publishers = px.bar(
            publisher_counts,
            x="Publisher",
            y="Number of Games",
            title="Top 8 Mobile Game Publishers",
            color="Number of Games",
            color_continuous_scale="Blues"
        )
        fig_publishers.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            title_x=0.5,
            title_xanchor="center",
            title_font_size=13
        )
        st.plotly_chart(fig_publishers, use_container_width=True)

with chart_col2:
    if "player_count" in df.columns:
        top_games = df.sort_values("player_count", ascending=False).head(10)

        fig_games = px.bar(
            top_games,
            x="game",
            y="player_count",
            title="Top 10 Mobile Games by Monthly Gamers",
            color="player_count",
            color_continuous_scale="Blues"
        )
        fig_games.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            title_x=0.5,
            title_xanchor="center",
            title_font_size=13
        )
        st.plotly_chart(fig_games, use_container_width=True)

with chart_col3:
    if "release_date" in df.columns and "player_count" in df.columns:

        trend_df = df.dropna(subset=["release_date", "player_count"]).copy()
        trend_df["year"] = trend_df["release_date"].dt.year

        yearly_players = trend_df.groupby("year")["player_count"].sum().reset_index()

        fig_scatter = px.scatter(
            yearly_players,
            x="year",
            y="player_count",
            title="Total Mobile Gamers per Year in Million",
        )

        
        fig_scatter.update_traces(
            marker=dict(
                size=10,
                color="#003791"  
            )
        )

        fig_scatter.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            title_x=0.5,
            title_xanchor="center",
            title_font_size=13
        )

        st.plotly_chart(fig_scatter, use_container_width=True)

# Game Reviews and Raw Data
st.subheader("Game Reviews and Data Tables")

left_side, right_side = st.columns([1, 1.4], gap="large")

with left_side:
    st.markdown("### Mobile Games Reviews")
    st.markdown(
        "<p style='font-size:12px;'>"
        "Interested on what made the top 3 most popular games reached millions of gamers? Check out the reviews below!"
        "</p>",
        unsafe_allow_html=True
    )

    review_col1, review_col2, review_col3 = st.columns(3, gap="small")

with review_col1:
    st.markdown("<div style='text-align:center;'>", unsafe_allow_html=True)
    st.image("https://img.youtube.com/vi/0s2DnKmZZ44/0.jpg", width=110)
    st.markdown(
        "<p style='font-size:11px;'>"
        "<a href='https://www.youtube.com/watch?v=0s2DnKmZZ44' target='_blank'>Mario Kart Tour</a>"
        "</p>",
        unsafe_allow_html=True
    )
    st.markdown("</div>", unsafe_allow_html=True)

with review_col2:
    st.markdown("<div style='text-align:center;'>", unsafe_allow_html=True)
    st.image("https://img.youtube.com/vi/F6nMSY3UaE0/0.jpg", width=110)
    st.markdown(
        "<p style='font-size:11px;'>"
        "<a href='https://www.youtube.com/watch?v=F6nMSY3UaE0' target='_blank'>Among Us</a>"
        "</p>",
        unsafe_allow_html=True
    )
    st.markdown("</div>", unsafe_allow_html=True)

with review_col3:
    st.markdown("<div style='text-align:center;'>", unsafe_allow_html=True)
    st.image("https://img.youtube.com/vi/Giupjw1EQlE/0.jpg", width=110)
    st.markdown(
        "<p style='font-size:11px;'>"
        "<a href='https://www.youtube.com/watch?v=Giupjw1EQlE' target='_blank'>Mini World</a>"
        "</p>",
        unsafe_allow_html=True
    )
    st.markdown("</div>", unsafe_allow_html=True)
        
with right_side:
    st.markdown("### Raw Data Tables")

    table_option = st.selectbox(
        "Select table to display",
        [
            "Top Games",
            "Current Stock Prices",
            "Raw Dataset",
            "Dataset Information",
            "Descriptive Statistics",
            "Popularity Classification"
        ]
    )

    if table_option == "Top Games":
        st.dataframe(stats.top_games(), use_container_width=True)

    elif table_option == "Current Stock Prices":
        if not df_stock.empty:
            st.dataframe(df_stock, use_container_width=True)
        else:
            st.info("No stock price data available.")

    elif table_option == "Raw Dataset":
        st.dataframe(df.head(), use_container_width=True)
        st.write("This shows the first 5 rows from the dataset used for this dashboard.")

    elif table_option == "Dataset Information":
        info_df = pd.DataFrame({
            "Column": df.columns,
            "Non-Null Count": df.notnull().sum().values,
            "Data Type": df.dtypes.astype(str).values
        })
        st.dataframe(info_df, use_container_width=True)

    elif table_option == "Descriptive Statistics":
        st.dataframe(df.describe(include="all"), use_container_width=True)

    elif table_option == "Popularity Classification":
        if "player_count" in df.columns:
            st.dataframe(df[["game", "player_count", "popularity_level"]], use_container_width=True)
        else:
            st.info("Popularity classification is not available.")