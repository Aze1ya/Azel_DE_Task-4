import os
import re
import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import streamlit as st
from pathlib import Path
from dateutil import parser as dateutil_parser

DATA_ROOT = Path(os.environ.get("DATA_ROOT", Path(__file__).parent / "data"))

st.set_page_config(
    page_title="Book Sales Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

def parse_price(raw):
    if pd.isna(raw):
        return np.nan
    s = str(raw).strip()
    is_euro = bool(re.search(r"€|EUR", s, re.I))
    cent_match = re.search(r"(\d+)[€$¢\s]*[¢](\d+)", s)
    if cent_match:
        val = int(cent_match.group(1)) + int(cent_match.group(2)) / 100
    else:
        digits = re.sub(r"[^\d.]", "", s)
        if not digits or digits == ".":
            return np.nan
        try:
            val = float(digits)
        except ValueError:
            return np.nan
    if is_euro:
        val *= 1.2
    return round(val, 2)

def parse_ts(raw):
    if pd.isna(raw):
        return pd.NaT
    s = str(raw).strip().replace(";", " ").replace(",", " ")
    s = re.sub(r"A\.M\.", "AM", s, flags=re.I)
    s = re.sub(r"P\.M\.", "PM", s, flags=re.I)
    try:
        return dateutil_parser.parse(s, dayfirst=False)
    except Exception:
        try:
            return dateutil_parser.parse(s, dayfirst=True)
        except Exception:
            return pd.NaT

@st.cache_data(show_spinner=False)
def load_dataset(root: str, name: str):
    folder = Path(root) / name

    users = pd.read_csv(folder / "users.csv")
    users.columns = users.columns.str.strip()
    users = users.drop_duplicates()
    users["id"] = pd.to_numeric(users["id"], errors="coerce")
    users = users.dropna(subset=["id"])
    users["id"] = users["id"].astype(int)
    for col in ["name", "address", "phone", "email"]:
        users[col] = users[col].astype(str).str.strip()
        users[col] = users[col].replace({"": np.nan, "nan": np.nan, "NULL": np.nan})

    with open(folder / "books.yaml", encoding="utf-8") as f:
        raw_books = yaml.safe_load(f)
    books = pd.DataFrame(raw_books)
    books.columns = [c.lstrip(":") for c in books.columns]
    books = books.drop_duplicates(subset=["id"])
    books["id"] = pd.to_numeric(books["id"], errors="coerce").astype("Int64")

    orders = pd.read_parquet(folder / "orders.parquet")
    orders.columns = orders.columns.str.strip()
    orders = orders.drop_duplicates()
    for col in ["id", "user_id", "book_id", "quantity"]:
        orders[col] = pd.to_numeric(orders[col], errors="coerce")
    orders = orders.dropna(subset=["id", "user_id", "book_id", "quantity"])
    orders["unit_price_usd"] = orders["unit_price"].apply(parse_price)
    orders["timestamp_dt"] = orders["timestamp"].apply(parse_ts)
    orders = orders.dropna(subset=["unit_price_usd", "timestamp_dt"])
    orders["paid_price"] = (orders["quantity"] * orders["unit_price_usd"]).round(2)
    orders["date"] = orders["timestamp_dt"].dt.date
    orders["year"] = orders["timestamp_dt"].dt.year
    orders["month"] = orders["timestamp_dt"].dt.month

    return users, books, orders

def daily_revenue(orders):
    dr = (
        orders.groupby("date")["paid_price"]
        .sum()
        .reset_index()
        .rename(columns={"paid_price": "revenue"})
        .sort_values("date")
    )
    top5 = dr.nlargest(5, "revenue").reset_index(drop=True)
    top5["date_str"] = top5["date"].astype(str)
    return dr, top5

def reconcile_users(users):
    ids = users["id"].tolist()
    parent = {i: i for i in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for field in ["email", "phone", "name"]:
        idx: dict = {}
        for _, row in users.iterrows():
            val = row[field]
            if pd.isna(val) or str(val).strip() == "":
                continue
            val = str(val).strip().lower()
            idx.setdefault(val, []).append(row["id"])
        for uid_list in idx.values():
            for i in range(1, len(uid_list)):
                union(uid_list[0], uid_list[i])

    roots = {find(i) for i in ids}
    return parent, len(roots)

def author_sets(books):
    def parse_authors(s):
        if pd.isna(s):
            return frozenset()
        return frozenset(a.strip() for a in str(s).split(",") if a.strip())

    sets = books["author"].apply(parse_authors)
    unique = {s for s in sets if s}
    return unique, len(unique)

def most_popular_author(books, orders):
    def parse_authors(s):
        if pd.isna(s):
            return frozenset()
        return frozenset(a.strip() for a in str(s).split(",") if a.strip())

    b = books.copy()
    b["author_set"] = b["author"].apply(parse_authors)
    b["book_id"] = b["id"]
    merged = orders.merge(b[["book_id", "author_set"]], on="book_id", how="left")
    merged = merged.dropna(subset=["author_set"])
    merged = merged[merged["author_set"].apply(bool)]
    merged["author_key"] = merged["author_set"].apply(lambda s: ", ".join(sorted(s)))
    sold = merged.groupby("author_key")["quantity"].sum()
    top_key = sold.idxmax()
    return top_key, int(sold.max())

def top_customer(users, orders, parent):
    def find(x, p):
        while p[x] != x:
            p[x] = p[p[x]]
            x = p[x]
        return x

    root_map = {uid: find(uid, dict(parent)) for uid in parent}
    o = orders.copy()
    o["root_id"] = o["user_id"].map(root_map)
    spending = o.groupby("root_id")["paid_price"].sum()
    top_root = spending.idxmax()
    cluster_ids = sorted(uid for uid, root in root_map.items() if root == top_root)
    return cluster_ids, round(spending.max(), 2)

def make_revenue_fig(dr, title):
    fig, ax = plt.subplots(figsize=(10, 3.5))
    dates = pd.to_datetime(dr["date"])
    ax.plot(dates, dr["revenue"], color="#4F8EF7", linewidth=2)
    ax.fill_between(dates, dr["revenue"], alpha=0.12, color="#4F8EF7")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Date")
    ax.set_ylabel("Revenue (USD)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate(rotation=35)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig

st.title("Book Sales Dashboard")
st.caption("Revenue · Users · Authors · Top Buyers")

DATASETS = ["DATA1", "DATA2", "DATA3"]
found = [ds for ds in DATASETS if (DATA_ROOT / ds).is_dir()]

if not found:
    st.error(
        f"**Can not find folders DATA1/DATA2/DATA3** in `{DATA_ROOT}`.\n\n"
        "Check, that:\n"
        "- data in `data/DATA1`, `data/DATA2`, `data/DATA3` with `app.py`\n"
        "- or environmental variables `DATA_ROOT` point to the desired directory"
    )
    st.stop()

tabs = st.tabs([f" {ds}" for ds in found])

for tab, ds in zip(tabs, found):
    with tab:
        with st.spinner(f"Processing {ds}…"):
            try:
                users, books, orders = load_dataset(str(DATA_ROOT), ds)
            except Exception as exc:
                st.error(f"Loading error {ds}: {exc}")
                continue

        dr, top5 = daily_revenue(orders)
        parent, n_unique = reconcile_users(users)
        _, n_sets = author_sets(books)
        pop_author, pop_count = most_popular_author(books, orders)
        cluster_ids, top_spend = top_customer(users, orders, parent)

        c1, c2, c3 = st.columns(3)
        c1.metric("Orders", f"{len(orders):,}")
        c2.metric("Unique users", f"{n_unique:,}")
        c3.metric("Unique authors", f"{n_sets:,}")

        st.divider()

        col_chart, col_table = st.columns([2, 1])
        with col_chart:
            st.subheader("Daily Revenue")
            fig = make_revenue_fig(dr, f"{ds} — Daily Revenue")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with col_table:
            st.subheader("Top 5 days")
            t5 = top5[["date_str", "revenue"]].copy()
            t5.columns = ["Date", "Revenue"]
            t5["Revenue"] = t5["Revenue"].map("${:,.2f}".format)
            st.dataframe(t5, hide_index=True, use_container_width=True)

        st.divider()

        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("The most popular author(-s)")
            st.info(f"**{pop_author}**\n\nBooks sold: **{pop_count:,}**")
        with col_b:
            st.subheader("Top buyer")
            st.markdown(f"**Total spending: ${top_spend:,.2f}**")
            st.markdown(f"**All user IDs:** `{cluster_ids}`")

        st.subheader("Top buyer — full profile (all aliases)")
        top_users_df = users[users["id"].isin(cluster_ids)].reset_index(drop=True)
        st.dataframe(top_users_df, use_container_width=True, hide_index=True)

        with st.expander("Raw data (first 100 rows)"):
            t1, t2, t3 = st.tabs(["users", "books", "orders"])
            with t1:
                st.dataframe(users.head(100), use_container_width=True)
            with t2:
                st.dataframe(books.head(100), use_container_width=True)
            with t3:
                st.dataframe(orders.head(100), use_container_width=True)