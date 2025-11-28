# app.py — Cloud BI (instant filters, stable behavior)
import streamlit as st
import io, os
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime

# plotting
import plotly.express as px
import plotly.graph_objects as go

# forecasting
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except Exception:
    PROPHET_AVAILABLE = False

from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore")

# helpers (assume these exist in your project)
from ingestion.csv_loader import load_csv_from_fileobj
from ingestion.pdf_loader import extract_tables_from_pdf_bytes
from ingestion.schema_detector import detect_schema
from profiling.profiler import profile_df
from utils.s3_uploader import upload_fileobj_to_s3

REFERENCE_PROJECT_DOC = "/mnt/data/7th_sem_review_final.docx"
S3_BUCKET = os.getenv("S3_BUCKET_NAME", "bi-sales-forecasting-uploads-your-unique-suffix")

st.set_page_config(page_title="Cloud BI", layout="wide")
st.title("Cloud BI : A Business Intelligence & Sales Forecasting App")

# -----------------------
# Sidebar: global interactive controls
# -----------------------
st.sidebar.header("Controls")
st.sidebar.info(f"S3 bucket: {S3_BUCKET}")

uploaded_file = st.sidebar.file_uploader("Upload CSV or PDF (business dataset)", type=["csv", "pdf"]) 
file_bytes = None
if uploaded_file is not None:
    file_bytes = uploaded_file.read()
    try:
        upload_fileobj_to_s3(io.BytesIO(file_bytes), uploaded_file.name)
        st.sidebar.success("Saved raw file to S3")
    except Exception:
        st.sidebar.warning("Could not save to S3 (check credentials)")

# persistent UI options
agg_freq = st.sidebar.selectbox(
    "Aggregation frequency", options=["D", "W", "MS", "QS", "YS"], index=2,
    help="D: daily, W: weekly, MS: month start, QS: quarter start, YS: year start"
)
show_moving_avg = st.sidebar.checkbox("Show moving average", value=True)
ma_window = st.sidebar.number_input("MA window (periods)", min_value=1, max_value=12, value=3)

# optional debug toggle
DEBUG = False
if DEBUG:
    st.sidebar.markdown("**DEBUG ON**")
st.sidebar.markdown("---")

# -----------------------
# Main page layout and file handling
# -----------------------
if uploaded_file is None:
    st.info("Upload a CSV or a PDF containing tabular sales/business data to begin.")
    st.caption(f"Reference project doc: {REFERENCE_PROJECT_DOC}")
    st.stop()

# Load df depending on file type
if uploaded_file.name.lower().endswith(".csv"):
    df = load_csv_from_fileobj(io.BytesIO(file_bytes))
else:
    tables = extract_tables_from_pdf_bytes(file_bytes)
    if len(tables) == 0:
        st.error("No table detected in PDF. Please upload CSV or a PDF with clear tables.")
        st.stop()
    df = tables[0]

# light cleaning and schema detection
schema = detect_schema(df)

# -----------------------
# Utility detectors
# -----------------------
def find_date_columns(df, sample_frac=0.3, threshold=0.6):
    rows = df if sample_frac >= 1.0 else df.sample(frac=min(sample_frac, 1.0), random_state=1)
    candidates = []
    for c in df.columns:
        try:
            parsed = pd.to_datetime(rows[c], errors="coerce")
            if parsed.notna().mean() >= threshold:
                candidates.append(c)
        except Exception:
            continue
    return candidates

def find_numeric_columns(df, min_fraction=0.6):
    candidates = []
    for c in df.columns:
        try:
            coerced = pd.to_numeric(df[c], errors="coerce")
            if coerced.notna().mean() >= min_fraction:
                candidates.append(c)
        except Exception:
            continue
    return candidates

def find_product_like_columns(df, exclude_cols=None, prefer_tokens=None, max_unique=2000):
    if exclude_cols is None:
        exclude_cols = []
    if prefer_tokens is None:
        prefer_tokens = ["product", "prod", "item", "sku", "model", "category", "cat", "name"]
    cols = []
    for c in df.columns:
        if c in exclude_cols:
            continue
        low = c.lower()
        if any(tok in low for tok in prefer_tokens):
            cols.append(c)
    if cols:
        return cols
    # fallback: categorical-like columns
    cand = []
    for c in df.columns:
        if c in exclude_cols:
            continue
        try:
            nunique = int(df[c].nunique(dropna=True))
        except Exception:
            continue
        if (df[c].dtype == "O" or pd.api.types.is_categorical_dtype(df[c])) and 2 <= nunique <= max_unique:
            cand.append(c)
        elif 2 <= nunique <= 500 and not pd.api.types.is_datetime64_any_dtype(df[c]):
            cand.append(c)
    return cand

# -----------------------
# Candidate lists for selectors
# -----------------------
date_candidates = list(dict.fromkeys(
    [c for c, m in schema.items() if m.get("is_date")] + find_date_columns(df, sample_frac=0.3, threshold=0.6)
))
num_candidates = list(dict.fromkeys(
    [c for c, m in schema.items() if m.get("is_numeric")] + find_numeric_columns(df, min_fraction=0.6)
))
# exclude_cols expects an iterable of column names; create a set of names
exclude_for_prod = set(date_candidates + num_candidates)
prod_candidates = find_product_like_columns(df, exclude_cols=exclude_for_prod)
region_candidates = [c for c in df.columns if any(k in c.lower() for k in ["region", "state", "city", "location", "country"])]

# Fallbacks
if not date_candidates:
    st.warning("No obvious date columns detected — select the date column manually.")
    date_candidates = list(df.columns)
if not num_candidates:
    num_candidates = list(df.columns)

# -----------------------
# UI: preview + basic selectors
# -----------------------
col1, col2 = st.columns([3, 1])
with col1:
    st.subheader("Data preview")
    st.dataframe(df.head(10))
with col2:
    st.subheader("Auto-detected schema")
    st.json(schema)

date_col = st.selectbox("Select date column", options=date_candidates, index=0)
value_col = st.selectbox("Select revenue/quantity column", options=num_candidates, index=0)

# product / region column dropdown — keep selection lightweight (columns only)
product_col_options = [None] + prod_candidates + [c for c in df.columns if c not in (date_candidates + num_candidates)]
product_col_choice = st.selectbox("Select product column (optional)", options=product_col_options[:200])

region_col_options = [None] + region_candidates
region_col_choice = st.selectbox("Select region column (optional)", options=region_col_options[:200])

# -----------------------
# Helper functions & safe unique values
# -----------------------
def safe_unique_values(series, sample_max=5000, max_options=1000):
    try:
        s = series.dropna()
        if len(s) > sample_max:
            s = s.sample(sample_max, random_state=1)
        vals = s.astype(str).unique().tolist()
        vals.sort()
        total_unique = int(series.nunique(dropna=True))
        return vals[:max_options], total_unique
    except Exception:
        return [], 0

# ---------- Replace build_timeseries (no cache) ----------
def build_timeseries(data, date_col, value_col, freq):
    tmp = data[[date_col, value_col]].copy()
    tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
    tmp = tmp.dropna(subset=[date_col])
    tmp.set_index(date_col, inplace=True)
    ts = tmp.resample(freq)[value_col].sum().reset_index().rename(columns={date_col: "ds", value_col: "y"})
    return ts

# ---------- Date range picker (explicit key) ----------
df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
min_date, max_date = df[date_col].min(), df[date_col].max()

default_start = min_date.date() if pd.notna(min_date) else None
default_end = max_date.date() if pd.notna(max_date) else None

start_date, end_date = st.date_input(
    "Date range",
    value=[default_start, default_end],
    key="date_range_picker"
)
if start_date is None or end_date is None:
    st.warning("Please select a valid date range.")
    st.stop()

# --------- Product and Region filters (no Apply button, no forced defaults) ---------
st.subheader("Product and Region Filters")

# session-state keys
_prod_sel_key = "prod_filter"
_prod_col_key = "prod_filter_col"
_reg_sel_key = "region_filter"
_reg_col_key = "region_filter_col"

# reset product multiselect if the selected product column changed
if st.session_state.get(_prod_col_key) != product_col_choice:
    # clear previous product selections when column changes
    st.session_state[_prod_sel_key] = []
    st.session_state[_prod_col_key] = product_col_choice

# Product filter (do NOT supply a runtime-default that forces reset)
products_selected = []
if product_col_choice:
    vals_prod, total_prod = safe_unique_values(df[product_col_choice], sample_max=2000, max_options=1000)

    if total_prod == 0:
        st.caption("Selected product column has no non-null values.")
        products_selected = []
    else:
        if total_prod > 1000:
            st.caption(f"Product column has {total_prod} unique values — showing up to 1000 sampled options.")
        # use the session-state key and pass the current session value as default
        default_prod = st.session_state.get(_prod_sel_key, [])
        products_selected = st.multiselect(
            "Products (multi-select — sample shown)",
            options=vals_prod[:500],
            default=default_prod,
            key=_prod_sel_key
        )
else:
    products_selected = []

# reset region multiselect if the selected region column changed
if st.session_state.get(_reg_col_key) != region_col_choice:
    st.session_state[_reg_sel_key] = []
    st.session_state[_reg_col_key] = region_col_choice

# Region filter
regions_selected = []
if region_col_choice:
    vals_reg, total_reg = safe_unique_values(df[region_col_choice], sample_max=2000, max_options=1000)

    if total_reg == 0:
        st.caption("Selected region column has no non-null values.")
        regions_selected = []
    else:
        if total_reg > 1000:
            st.caption(f"Region column has {total_reg} unique values — showing up to 1000 sampled options.")
        default_reg = st.session_state.get(_reg_sel_key, [])
        regions_selected = st.multiselect(
            "Regions (multi-select — sample shown)",
            options=vals_reg[:500],
            default=default_reg,
            key=_reg_sel_key
        )
else:
    regions_selected = []

# optional debug block (enable by setting DEBUG = True at top)
if DEBUG:
    st.sidebar.write("DEBUG: widget values")
    st.sidebar.write("start_date, end_date:", start_date, end_date)
    st.sidebar.write("prod_filter (raw):", st.session_state.get("prod_filter", None))
    st.sidebar.write("region_filter (raw):", st.session_state.get("region_filter", None))

# -----------------------
# Apply filters safely (instant)
# -----------------------
# start with all True mask to avoid alignment pitfalls
mask = pd.Series(True, index=df.index)

try:
    # ensure df[date_col] is datetime
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    date_mask = (df[date_col].dt.date >= pd.to_datetime(start_date).date()) & (df[date_col].dt.date <= pd.to_datetime(end_date).date())
    mask &= date_mask
except Exception as e:
    st.error(f"Date filtering failed: {e}")

if product_col_choice and products_selected:
    try:
        mask &= df[product_col_choice].astype(str).isin([str(x) for x in products_selected])
    except Exception as e:
        st.error(f"Product filter failed — ignoring product filter. Error: {e}")

if region_col_choice and regions_selected:
    try:
        mask &= df[region_col_choice].astype(str).isin([str(x) for x in regions_selected])
    except Exception as e:
        st.error(f"Region filter failed — ignoring region filter. Error: {e}")

try:
    df_filtered = df.loc[mask].copy()
except Exception as e:
    st.error(f"Failed to apply filters to dataframe: {e}")
    df_filtered = pd.DataFrame()

if DEBUG:
    try:
        st.sidebar.write("mask sum:", int(mask.sum()))
        st.sidebar.write("df_filtered shape:", df_filtered.shape)
    except Exception:
        pass

if df_filtered.empty:
    st.warning("No data after applying filters. Adjust filters or date range.")
    st.stop()

# canonical names
product_col = product_col_choice
region_col = region_col_choice

# -------------------
# Numeric-safe value column
# -------------------
value_numeric_col = "__value_numeric__"
df_filtered[value_numeric_col] = pd.to_numeric(df_filtered[value_col], errors="coerce")

n_total = len(df_filtered)
n_numeric = int(df_filtered[value_numeric_col].notna().sum()) if n_total > 0 else 0
if n_total == 0:
    st.warning("Filtered dataset is empty.")
else:
    frac_valid = n_numeric / n_total
    if frac_valid < 0.7:
        st.warning(
            f"Only {n_numeric}/{n_total} rows ({frac_valid:.0%}) in the selected column could be parsed as numbers. "
            "Check the selected value column or clean your data. Calculations will proceed using numeric values available."
        )

# ---------- Tabs for organisation ----------
tab_overview, tab_forecast, tab_inventory, tab_dead = st.tabs(["Overview", "Forecast", "Inventory", "Dead Stock"])

# ---------- Overview tab ----------
with tab_overview:
    st.header("Overview & KPIs")
    total_revenue = float(df_filtered[value_numeric_col].sum(skipna=True))
    total_orders = int(df_filtered.shape[0])
    try:
        avg_order = float(df_filtered[value_numeric_col].sum(skipna=True) / df_filtered["Order ID"].nunique()) if "Order ID" in df_filtered.columns else float(df_filtered[value_numeric_col].mean(skipna=True))
    except Exception:
        avg_order = float(df_filtered[value_numeric_col].mean(skipna=True))

    k1, k2, k3, k4, k5 = st.columns([1.2, 1.2, 1.2, 1.2, 1.2])
    k1.metric("Total Revenue", f"{total_revenue:,.2f}")
    k2.metric("Total Orders", f"{total_orders}")
    k3.metric("Avg Order Value", f"{avg_order:,.2f}")

    profit_col = next((c for c in df.columns if "profit" in c.lower()), None)
    if profit_col:
        total_profit = float(pd.to_numeric(df_filtered[profit_col], errors="coerce").sum(skipna=True))
        k4.metric("Total Profit", f"{total_profit:,.2f}")
        k5.metric("Profit Margin", f"{(total_profit / total_revenue * 100) if total_revenue else 0:.2f}%")
    else:
        k4.metric("Total Profit", "N/A")
        k5.metric("Profit Margin", "N/A")

    st.markdown("---")
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.subheader("Last 90 days - Daily Sales")
        daily = build_timeseries(df_filtered, date_col, value_numeric_col, "D")
        if not daily.empty:
            last_90 = daily[daily["ds"] >= (daily["ds"].max() - pd.Timedelta(days=90))]
            if not last_90.empty:
                last_90["ma7"] = last_90["y"].rolling(7, min_periods=1).mean()
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=last_90["ds"], y=last_90["y"], name="Daily"))
                fig.add_trace(go.Scatter(x=last_90["ds"], y=last_90["ma7"], name="7-day MA", line=dict(dash="dash")))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Not enough daily data for last 90 days.")
        else:
            st.info("No daily timeseries data available.")

    with col_b:
        st.subheader("Monthly Sales & Month over Month Growth")
        monthly = build_timeseries(df_filtered, date_col, value_numeric_col, "MS")
        if len(monthly) >= 2:
            figm = px.bar(monthly, x="ds", y="y", title="Monthly Sales")
            figm.update_layout(height=300)
            st.plotly_chart(figm, use_container_width=True)
            last = monthly.iloc[-1]["y"]
            prev = monthly.iloc[-2]["y"]
            change = (last - prev) / prev * 100 if prev else 0
            st.metric(label="Last month vs previous", value=f"{last:,.2f}", delta=f"{change:.2f}%")
        else:
            st.info("Not enough monthly data to show chart.")

    with col_c:
        st.subheader("Sales by Weekday")
        temp = df_filtered.copy()
        temp["weekday"] = temp[date_col].dt.day_name()
        weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        weekday_agg = temp.groupby("weekday")[value_numeric_col].sum().reindex(weekday_order).reset_index().fillna(0)
        figw = px.bar(weekday_agg, x="weekday", y=value_numeric_col, title="Sales by Weekday")
        figw.update_layout(height=300)
        st.plotly_chart(figw, use_container_width=True)

    st.markdown("---")
    col_d, col_e = st.columns(2)

    with col_d:
        st.subheader("Top Products - Pareto")
        if product_col:
            try:
                prod_agg = df_filtered.groupby(product_col)[value_numeric_col].sum().reset_index().sort_values(value_numeric_col, ascending=False)
                prod_agg["cum_sum"] = prod_agg[value_numeric_col].cumsum()
                prod_agg["cum_perc"] = 100 * prod_agg["cum_sum"] / prod_agg[value_numeric_col].sum()
                top_n = st.slider("Top N products to show", min_value=5, max_value=50, value=10)
                pa = prod_agg.head(top_n)
                figp = go.Figure()
                figp.add_trace(go.Bar(x=pa[product_col], y=pa[value_numeric_col], name="Revenue"))
                figp.add_trace(go.Scatter(x=pa[product_col], y=pa["cum_perc"], name="Cumulative %", yaxis="y2", mode="lines+markers"))
                figp.update_layout(yaxis2=dict(overlaying="y", side="right", title="Cumulative %"))
                st.plotly_chart(figp, use_container_width=True)
            except Exception as e:
                st.error(f"Pareto chart failed: {e}")
        else:
            st.info("No product column selected for Pareto.")

    with col_e:
        st.subheader("Sales vs Profit")
        if product_col and profit_col in df.columns:
            try:
                sp = df_filtered.groupby(product_col).agg({value_numeric_col: "sum", profit_col: "sum"}).reset_index()
                figsp = px.scatter(sp, x=value_numeric_col, y=profit_col, hover_data=[product_col], title="Sales vs Profit (by product)", size=value_numeric_col)
                st.plotly_chart(figsp, use_container_width=True)
            except Exception as e:
                st.error(f"Sales vs Profit failed: {e}")
        elif region_col and profit_col in df.columns:
            try:
                rp = df_filtered.groupby(region_col).agg({value_numeric_col: "sum", profit_col: "sum"}).reset_index()
                figsp = px.scatter(rp, x=value_numeric_col, y=profit_col, hover_data=[region_col], title="Sales vs Profit (by region)", size=value_numeric_col)
                st.plotly_chart(figsp, use_container_width=True)
            except Exception as e:
                st.error(f"Sales vs Profit (region) failed: {e}")
        else:
            st.info("Profit column or product/region not available for this chart.")

# ---------- Forecast tab ----------
with tab_forecast:
    st.header("Demand Forecasting")
    engine = st.selectbox("Engine", options=["Auto", "Prophet", "ARIMA"]) if PROPHET_AVAILABLE else st.selectbox("Engine", options=["ARIMA"]) 
    horizon = st.number_input("Forecast horizon (periods)", min_value=1, value=6)
    ts = build_timeseries(df_filtered, date_col, value_numeric_col, agg_freq)

    def run_prophet(ts_df, periods=6):
        if not PROPHET_AVAILABLE:
            st.warning("Prophet not installed.")
            return None
        m = Prophet()
        m.fit(ts_df.rename(columns={"ds": "ds", "y": "y"}))
        future = m.make_future_dataframe(periods=periods, freq=agg_freq)
        forecast = m.predict(future)
        return forecast

    def run_arima(ts_df, periods=6):
        try:
            model = ARIMA(ts_df["y"].astype(float), order=(1, 1, 1))
            res = model.fit()
            fc = res.get_forecast(steps=periods)
            idx = pd.date_range(start=ts_df["ds"].max() + pd.offsets.MonthBegin(1), periods=periods, freq=agg_freq)
            df_fc = pd.DataFrame({"ds": idx, "yhat": fc.predicted_mean})
            return df_fc
        except Exception as e:
            st.error(f"ARIMA error: {e}")
            return None

    if st.button("Run Forecast", key="run_fc"):
        with st.spinner("Forecasting..."):
            if engine == "Prophet" and PROPHET_AVAILABLE:
                fc = run_prophet(ts, periods=horizon)
                if fc is not None:
                    st.dataframe(fc[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(horizon))
                    fig = px.line(fc, x="ds", y="yhat", title="Prophet forecast")
                    st.plotly_chart(fig, use_container_width=True)
            else:
                fc = run_arima(ts, periods=horizon)
                if fc is not None:
                    st.dataframe(fc)
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=ts["ds"], y=ts["y"], name="historical"))
                    fig.add_trace(go.Scatter(x=fc["ds"], y=fc["yhat"], name="forecast"))
                    st.plotly_chart(fig, use_container_width=True)

# ---------- Inventory tab ----------
with tab_inventory:
    st.header("Inventory Optimization")
    lead_time_days = st.number_input("Lead time (days)", min_value=1, value=7)
    ordering_cost = st.number_input("Ordering cost per order (S)", min_value=0.0, value=50.0)
    holding_cost_rate = st.number_input("Annual holding cost rate (as fraction)", min_value=0.0, value=0.2)
    unit_cost = st.number_input("Avg unit cost (C)", min_value=0.0, value=10.0)

    def compute_inventory(data, date_col, prod_col, qty_col):
        if prod_col is None:
            return None
        tmp = data[[date_col, prod_col, qty_col]].copy()
        tmp[date_col] = pd.to_datetime(tmp[date_col])
        monthly = tmp.groupby([prod_col, pd.Grouper(key=date_col, freq="MS")])[qty_col].sum().reset_index()
        rows = []
        for prod, g in monthly.groupby(prod_col):
            demand_monthly = g[qty_col].mean()
            D = demand_monthly * 12
            if D <= 0 or np.isnan(D):
                continue
            H = unit_cost * holding_cost_rate
            Q_star = np.sqrt((2 * D * ordering_cost) / H) if H > 0 else np.nan
            daily = D / 365
            R = daily * lead_time_days + demand_monthly
            rows.append({prod_col: prod, "annual_D": round(D, 2), "EOQ": round(Q_star, 2) if not np.isnan(Q_star) else None, "ROP": round(R, 2)})
        return pd.DataFrame(rows)

    inv = compute_inventory(df_filtered, date_col, product_col, value_numeric_col)
    if inv is not None and not inv.empty:
        st.dataframe(inv.sort_values("annual_D", ascending=False))
        fig_eoq = px.histogram(inv, x="EOQ", nbins=30, title="EOQ distribution across products")
        st.plotly_chart(fig_eoq, use_container_width=True)
        csv_buf = io.StringIO()
        inv.to_csv(csv_buf, index=False)
        st.download_button("Download inventory CSV", data=csv_buf.getvalue(), file_name="inventory_metrics.csv")
    else:
        st.info("No product-level data available to compute inventory metrics.")

# ---------- Dead stock tab ----------
with tab_dead:
    st.header("Dead Stock Identification")
    months = st.number_input("No-sales threshold (months)", min_value=1, value=6)

    def identify_dead(data, date_col, prod_col, qty_col, months):
        if prod_col is None:
            return None
        cutoff = pd.to_datetime(data[date_col]).max() - pd.DateOffset(months=months)
        tmp = data[[date_col, prod_col, qty_col]].copy()
        tmp[date_col] = pd.to_datetime(tmp[date_col])
        recent = tmp[tmp[date_col] > cutoff].groupby(prod_col)[qty_col].sum().reset_index().rename(columns={qty_col: "recent_qty"})
        all_prods = tmp[prod_col].unique()
        df_recent = pd.DataFrame({prod_col: all_prods}).merge(recent, how="left", on=prod_col).fillna(0)
        dead = df_recent[df_recent["recent_qty"] == 0]
        return dead

    dead_df = identify_dead(df_filtered, date_col, product_col, value_numeric_col, months)
    if dead_df is not None and not dead_df.empty:
        st.dataframe(dead_df)
        total_skus = df_filtered[product_col].nunique() if product_col else 0
        dead_count = dead_df.shape[0]
        fig_pie = px.pie(names=["Active SKUs", "Dead SKUs"], values=[total_skus - dead_count, dead_count], title="SKU health")
        st.plotly_chart(fig_pie, use_container_width=True)
        csv_buf = io.StringIO()
        dead_df.to_csv(csv_buf, index=False)
        st.download_button("Export dead-stock CSV", data=csv_buf.getvalue(), file_name="dead_stock.csv")
    else:
        st.info("No dead stock detected for selected filters.")

# ---------------- SQL Explorer (run SQL on the filtered dataset) ----------------
with st.expander("SQL Explorer — run SQL on filtered data", expanded=False):
    st.markdown("Run SQL queries on the currently filtered dataset (in-memory SQLite).")
    try:
        conn = sqlite3.connect(":memory:")
        df_filtered.to_sql("data", conn, if_exists="replace", index=False)
    except Exception as e:
        st.error(f"Could not create in-memory database: {e}")
        conn = None

    if conn is not None:
        try:
            tables_df = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;", conn)
            st.write("Tables loaded in SQL engine:", tables_df["name"].tolist())
        except Exception:
            pass

        try:
            schema_df = pd.read_sql_query("PRAGMA table_info('data');", conn)
            if not schema_df.empty:
                st.text("Schema for table 'data' (cid, name, type, notnull, dflt_value, pk):")
                st.dataframe(schema_df)
        except Exception:
            pass

        default_sql = "SELECT * FROM data LIMIT 100;"
        sql = st.text_area("Athena query", value=default_sql, height=160, key="sql_explorer")
        col_run, col_reset = st.columns([1, 1])
        with col_run:
            if st.button("Athena SQL"):
                try:
                    res = pd.read_sql_query(sql, conn)
                    st.success(f"Query returned {len(res)} rows.")
                    st.dataframe(res.head(1000))
                    csv_buf = io.StringIO()
                    res.to_csv(csv_buf, index=False)
                    st.download_button("Download Athena result (CSV)", data=csv_buf.getvalue(), file_name="sql_result.csv")
                except Exception as e:
                    st.error(f"Athena error: {e}")
        with col_reset:
            if st.button("Reset SQL"):
                st.experimental_rerun()
                

# final: let user download filtered dataset
st.markdown("---")
out_df = df_filtered.copy()
st.download_button("Download filtered dataset (CSV)", data=out_df.to_csv(index=False), file_name="filtered_data.csv")
st.success("Interactive analysis ready — adjust filters to explore different slices.")
