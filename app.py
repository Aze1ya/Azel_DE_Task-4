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
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

*, *::before, *::after { box-sizing: border-box; }
html, body, [data-testid="stAppViewContainer"] {
    background: #0b0f1a !important;
    color: #e2e8f0 !important;
    font-family: 'Space Grotesk', sans-serif !important;
}

#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1526 0%, #0b1220 100%) !important;
    border-right: 1px solid #1e2d45 !important;
}
[data-testid="stSidebar"] * { color: #94a3b8 !important; }

[data-testid="stMain"] > div { padding: 2rem 2.5rem !important; }

.dash-title {
    font-size: 28px;
    font-weight: 700;
    color: #f1f5f9;
    letter-spacing: -0.5px;
    margin-bottom: 2px;
}
.dash-subtitle {
    font-size: 12px;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 28px;
}

.kpi-row { display: flex; gap: 16px; margin-bottom: 28px; }
.kpi-card {
    flex: 1;
    background: #111827;
    border: 1px solid #1e2d45;
    border-radius: 12px;
    padding: 20px 24px;
    position: relative;
    overflow: hidden;
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 12px 12px 0 0;
}
.kpi-card.blue::before   { background: linear-gradient(90deg, #3b82f6, #60a5fa); }
.kpi-card.cyan::before   { background: linear-gradient(90deg, #06b6d4, #67e8f9); }
.kpi-card.violet::before { background: linear-gradient(90deg, #8b5cf6, #c4b5fd); }
.kpi-card.emerald::before{ background: linear-gradient(90deg, #10b981, #6ee7b7); }

.kpi-label {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #64748b;
    margin-bottom: 8px;
}
.kpi-value {
    font-size: 30px;
    font-weight: 700;
    color: #f1f5f9;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: -1px;
    line-height: 1;
}
.kpi-icon {
    position: absolute;
    top: 18px; right: 18px;
    font-size: 20px;
    opacity: 0.25;
}

.section-header {
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #38bdf8;
    border-left: 3px solid #38bdf8;
    padding-left: 10px;
    margin-bottom: 14px;
}

.info-box {
    background: #111827;
    border: 1px solid #1e2d45;
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 10px;
}
.info-box .label {
    font-size: 10px;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 6px;
}
.info-box .value { font-size: 15px; font-weight: 600; color: #e2e8f0; }
.info-box .value.accent { color: #38bdf8; }
.info-box .value.mono {
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    color: #a78bfa;
    word-break: break-all;
}

[data-testid="stTabs"] [role="tablist"] {
    background: #111827 !important;
    border-radius: 10px !important;
    border: 1px solid #1e2d45 !important;
    padding: 4px !important;
    gap: 4px !important;
}
[data-testid="stTabs"] [role="tab"] {
    background: transparent !important;
    color: #64748b !important;
    border-radius: 7px !important;
    font-weight: 500 !important;
    font-size: 13px !important;
    padding: 8px 22px !important;
    border: none !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    background: #1e3a5f !important;
    color: #38bdf8 !important;
    font-weight: 600 !important;
}

[data-testid="stDataFrame"] {
    border: 1px solid #1e2d45 !important;
    border-radius: 10px !important;
    overflow: hidden !important;
}

hr { border-color: #1e2d45 !important; margin: 24px 0 !important; }

[data-testid="stExpander"] {
    background: #111827 !important;
    border: 1px solid #1e2d45 !important;
    border-radius: 10px !important;
}

.nav-section {
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #334155;
    margin: 20px 0 8px;
}
.nav-item {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 9px 12px;
    border-radius: 8px;
    margin-bottom: 4px;
    font-size: 14px;
    font-weight: 500;
    color: #64748b;
}
.nav-item.active { background: #1e3a5f; color: #38bdf8; }
.sidebar-logo { font-size: 20px; font-weight: 700; color: #38bdf8 !important; letter-spacing: -0.5px; }
.sidebar-sub { font-size: 10px; color: #334155 !important; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 16px; }
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_price(raw):
    """Parse price strings in all formats found in this dataset:
      - Plain number: '22.75', '100', 'USD71.25'
      - Euro: '€50', '21.99EUR', 'EUR 73.00'  → value * 1.2
      - Cents with ¢: '49$50¢', '$16¢75', '$58¢25', '€28¢50', '22$75¢'
        All ¢-formats: extract all digit groups → [dollars, cents]
    """
    if pd.isna(raw): return np.nan
    s = str(raw).strip()
    is_euro = bool(re.search(r'€|EUR', s, re.I))

    # Any string containing ¢ — extract all digit groups in order: [dollars, cents]
    if '¢' in s:
        nums = re.findall(r'\d+', s)
        if len(nums) >= 2:
            val = int(nums[0]) + int(nums[1]) / 100
        elif len(nums) == 1:
            val = int(nums[0]) / 100  # bare cents like '75¢'
        else:
            return np.nan
        if is_euro: val *= 1.2
        return round(val, 2)

    digits = re.sub(r'[^\d.]', '', s)
    if not digits or digits == '.': return np.nan
    try: val = float(digits)
    except ValueError: return np.nan
    if is_euro: val *= 1.2
    return round(val, 2)

def parse_ts(raw):
    """Parse timestamps from 3 formats present in this dataset:
      - DD.MM.YYYY [time]   — dot-separated, day-first  (confirmed: entries like 19.02.2025)
      - YYYY-MM-DD [time]   — ISO 8601, must NOT use dayfirst
      - MM/DD/YY [time]     — US slash format           (confirmed: second part goes > 12)
      - everything else     — dateutil dayfirst=True fallback
    """
    if pd.isna(raw): return pd.NaT
    s = str(raw).strip().replace(";", " ").replace(",", " ")
    s = re.sub(r"A\.M\.", "AM", s, flags=re.I)
    s = re.sub(r"P\.M\.", "PM", s, flags=re.I)

    # 1) DD.MM.YYYY — dot format, always day-first in this dataset
    dot_match = re.search(r"\b(\d{1,2})\.(\d{1,2})\.(\d{4})\b", s)
    if dot_match:
        day, month, year = dot_match.groups()
        rest = s[:dot_match.start()] + s[dot_match.end():]
        iso = f"{year}-{int(month):02d}-{int(day):02d} {rest}".strip()
        try: return dateutil_parser.parse(iso)
        except Exception: return pd.NaT

    # 2) ISO YYYY-MM-DD — parse directly, no dayfirst
    if re.search(r"\d{4}-\d{2}-\d{2}", s):
        try: return dateutil_parser.parse(s, dayfirst=False)
        except Exception: return pd.NaT

    # 3) US slash MM/DD/YY[YY] — month-first
    if re.search(r"\d{1,2}/\d{1,2}/\d{2,4}", s):
        try: return dateutil_parser.parse(s, dayfirst=False)
        except Exception: return pd.NaT

    # 4) Everything else (named months, dashes, etc.) — dayfirst=True
    try: return dateutil_parser.parse(s, dayfirst=True)
    except Exception: return pd.NaT

@st.cache_data(show_spinner=False, ttl=0)
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
        orders.groupby("date")
        .agg(revenue=("paid_price", "sum"), total_orders=("id", "count"))
        .reset_index()
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
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb: parent[rb] = ra
    for field in ["email", "phone", "name"]:
        idx: dict = {}
        for _, row in users.iterrows():
            val = row[field]
            if pd.isna(val) or str(val).strip() == "": continue
            val = str(val).strip().lower()
            idx.setdefault(val, []).append(row["id"])
        for uid_list in idx.values():
            for i in range(1, len(uid_list)): union(uid_list[0], uid_list[i])
    return parent, len({find(i) for i in ids})

def author_sets(books):
    def pa(s):
        if pd.isna(s): return frozenset()
        return frozenset(a.strip() for a in str(s).split(",") if a.strip())
    sets = books["author"].apply(pa)
    return {s for s in sets if s}, len({s for s in sets if s})

def most_popular_author(books, orders):
    def pa(s):
        if pd.isna(s): return frozenset()
        return frozenset(a.strip() for a in str(s).split(",") if a.strip())
    b = books.copy(); b["author_set"] = b["author"].apply(pa); b["book_id"] = b["id"]
    merged = orders.merge(b[["book_id", "author_set"]], on="book_id", how="left")
    merged = merged.dropna(subset=["author_set"])
    merged = merged[merged["author_set"].apply(bool)]
    merged["author_key"] = merged["author_set"].apply(lambda s: ", ".join(sorted(s)))
    sold = merged.groupby("author_key")["quantity"].sum()
    return sold.idxmax(), int(sold.max())

def top_customer(users, orders, parent):
    def find(x, p):
        while p[x] != x: p[x] = p[p[x]]; x = p[x]
        return x
    root_map = {uid: find(uid, dict(parent)) for uid in parent}
    o = orders.copy(); o["root_id"] = o["user_id"].map(root_map)
    spending = o.groupby("root_id")["paid_price"].sum()
    top_root = spending.idxmax()
    return sorted(uid for uid, root in root_map.items() if root == top_root), round(spending.max(), 2)

# ── Chart factories ───────────────────────────────────────────────────────────

def make_revenue_fig(dr):
    BG, GRID, LINE, TEXT = "#111827", "#1e2d45", "#38bdf8", "#64748b"
    fig, ax = plt.subplots(figsize=(11, 3.6), facecolor=BG)
    ax.set_facecolor(BG)
    dates = pd.to_datetime(dr["date"])
    ax.plot(dates, dr["revenue"], color=LINE, linewidth=1.8, solid_capstyle="round")
    ax.fill_between(dates, dr["revenue"], alpha=0.09, color=LINE)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate(rotation=30)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.tick_params(colors=TEXT, labelsize=8.5)
    for spine in ax.spines.values(): spine.set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.7, linestyle="-")
    ax.grid(axis="x", visible=False)
    ax.set_xlim(dates.min(), dates.max())
    plt.tight_layout(pad=1.4)
    return fig

def make_top5_fig(top5):
    BG, GRID, TEXT = "#111827", "#1e2d45", "#64748b"
    COLORS = ["#3b82f6", "#2563eb", "#1d4ed8", "#1e40af", "#1e3a8a"]
    fig, ax = plt.subplots(figsize=(4.5, 3.6), facecolor=BG)
    ax.set_facecolor(BG)
    labels = top5["date_str"].tolist()
    values = top5["revenue"].tolist()
    bars = ax.barh(labels[::-1], values[::-1], color=COLORS, height=0.52, edgecolor="none")
    mx = max(values)
    for bar, val in zip(bars, values[::-1]):
        ax.text(bar.get_width() + mx * 0.025, bar.get_y() + bar.get_height() / 2,
                f"${val:,.0f}", va="center", ha="left", color="#60a5fa",
                fontsize=8, fontfamily="monospace", fontweight="600")
    ax.tick_params(colors=TEXT, labelsize=8)
    ax.set_xlim(0, mx * 1.35)
    for spine in ax.spines.values(): spine.set_visible(False)
    ax.grid(axis="x", color=GRID, linewidth=0.7)
    ax.grid(axis="y", visible=False)
    ax.xaxis.set_visible(False)
    plt.tight_layout(pad=1.2)
    return fig

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown('<div class="sidebar-logo">📚 BookSales</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-sub">Analytics Platform</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<div class="nav-section">Navigation</div>', unsafe_allow_html=True)
    st.markdown('<div class="nav-item active">📊 &nbsp; Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="nav-item">📦 &nbsp; Orders</div>', unsafe_allow_html=True)
    st.markdown('<div class="nav-item">👥 &nbsp; Customers</div>', unsafe_allow_html=True)
    st.markdown('<div class="nav-item">📖 &nbsp; Books</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<div class="nav-section">Datasets</div>', unsafe_allow_html=True)

# ── Page header ───────────────────────────────────────────────────────────────

st.markdown('<div class="dash-title">📊 Book Sales Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="dash-subtitle">Revenue · Users · Authors · Top Buyers</div>', unsafe_allow_html=True)

# ── Data ──────────────────────────────────────────────────────────────────────

DATASETS = ["DATA1", "DATA2", "DATA3"]
found = [ds for ds in DATASETS if (DATA_ROOT / ds).is_dir()]

if not found:
    st.error(
        f"**Cannot find DATA1/DATA2/DATA3** in `{DATA_ROOT}`.\n\n"
        "- Place data in `data/DATA1`, `data/DATA2`, `data/DATA3` alongside `app.py`\n"
        "- Or set `DATA_ROOT` environment variable to the correct path."
    )
    st.stop()

# Clear stale cache on every reload so parse fixes take effect immediately
load_dataset.clear()

tabs = st.tabs([f"  {ds}  " for ds in found])

for tab, ds in zip(tabs, found):
    with tab:
        with st.spinner(f"Loading {ds}…"):
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

        # KPI cards
        st.markdown(f"""
        <div class="kpi-row">
          <div class="kpi-card blue">
            <div class="kpi-icon">🛒</div>
            <div class="kpi-label">Total Orders</div>
            <div class="kpi-value">{len(orders):,}</div>
          </div>
          <div class="kpi-card cyan">
            <div class="kpi-icon">👥</div>
            <div class="kpi-label">Unique Users</div>
            <div class="kpi-value">{n_unique:,}</div>
          </div>
          <div class="kpi-card violet">
            <div class="kpi-icon">✍️</div>
            <div class="kpi-label">Unique Authors</div>
            <div class="kpi-value">{n_sets:,}</div>
          </div>
          <div class="kpi-card emerald">
            <div class="kpi-icon">💰</div>
            <div class="kpi-label">Top Buyer Spend</div>
            <div class="kpi-value">${top_spend:,.0f}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Revenue chart + top 5 bar
        col_chart, col_bar = st.columns([3, 1.5])
        with col_chart:
            st.markdown('<div class="section-header">Daily Revenue</div>', unsafe_allow_html=True)
            fig = make_revenue_fig(dr)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        with col_bar:
            st.markdown('<div class="section-header">Top 5 Days</div>', unsafe_allow_html=True)
            fig2 = make_top5_fig(top5)
            st.pyplot(fig2, use_container_width=True)
            plt.close(fig2)

        st.markdown("<hr>", unsafe_allow_html=True)

        # Daily stats table
        st.markdown('<div class="section-header">Daily Revenue & Orders</div>', unsafe_allow_html=True)
        daily_table = dr[["date", "revenue", "total_orders"]].copy()
        daily_table.columns = ["Day", "Daily Revenue", "Total Orders"]
        daily_table["Day"] = daily_table["Day"].astype(str)
        daily_table["Daily Revenue"] = daily_table["Daily Revenue"].map("${:,.2f}".format)
        daily_table = daily_table.sort_values("Day", ascending=False).reset_index(drop=True)
        st.dataframe(daily_table, use_container_width=True, hide_index=True, height=320)


        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown('<div class="section-header">Most Popular Author(s)</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="info-box">
              <div class="label">Author name</div>
              <div class="value accent">{pop_author}</div>
            </div>
            <div class="info-box">
              <div class="label">Total books sold</div>
              <div class="value">{pop_count:,}</div>
            </div>
            """, unsafe_allow_html=True)
        with col_b:
            st.markdown('<div class="section-header">Top Buyer</div>', unsafe_allow_html=True)
            ids_str = ", ".join(str(i) for i in cluster_ids)
            st.markdown(f"""
            <div class="info-box">
              <div class="label">Total spending</div>
              <div class="value">${top_spend:,.2f}</div>
            </div>
            <div class="info-box">
              <div class="label">All user IDs (aliases)</div>
              <div class="value mono">[{ids_str}]</div>
            </div>
            """, unsafe_allow_html=True)

        # Full profile table
        st.markdown('<div class="section-header" style="margin-top:24px;">Top Buyer — Full Profile (All Aliases)</div>', unsafe_allow_html=True)
        top_users_df = users[users["id"].isin(cluster_ids)].reset_index(drop=True)
        st.dataframe(top_users_df, use_container_width=True, hide_index=True)

        st.markdown("<hr>", unsafe_allow_html=True)

        with st.expander("🗄️  Raw data (first 100 rows)"):
            t1, t2, t3 = st.tabs(["Users", "Books", "Orders"])
            with t1: st.dataframe(users.head(100), use_container_width=True)
            with t2: st.dataframe(books.head(100), use_container_width=True)
            with t3: st.dataframe(orders.head(100), use_container_width=True)