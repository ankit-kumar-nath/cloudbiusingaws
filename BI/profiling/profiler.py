import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet

# Heuristic column name lists
DATE_CANDIDATES = ['date','order_date','orderdate','sale_date','transaction_date','ts']
REVENUE_CANDIDATES = ['revenue','sales','amount','total','net_sales','gross']
PRICE_CANDIDATES = ['price','unit_price','selling_price']
QTY_CANDIDATES = ['qty','quantity','units','order_qty']
COST_CANDIDATES = ['cost','unit_cost','cost_price']
PRODUCT_CANDIDATES = ['product','product_name','item','sku','product_id']
CUSTOMER_CANDIDATES = ['customer_id','customer','client_id','client']
REGION_CANDIDATES = ['region','state','city','country']

def _find_column(df, candidates):
    cols = [c for c in df.columns for cand in candidates if cand in str(c).lower()]
    return cols[0] if cols else None

def safe_to_datetime(series):
    try:
        return pd.to_datetime(series)
    except Exception:
        return None

def profile_df(df, date_col=None, value_col=None):
    """
    Performs in-depth profiling and generates interactive visualizations and time-series data used later for forecasting.
    Returns a dict with figures and dataframes.
    The function attempts to auto-detect sensible columns but will work with explicit selections.
    """
    df = df.copy()

    # auto-detect columns when not provided
    if date_col is None:
        date_col = _find_column(df, DATE_CANDIDATES)
    if value_col is None:
        value_col = _find_column(df, REVENUE_CANDIDATES)

    # try to coerce date column
    if date_col is not None and date_col in df.columns:
        try:
            df[date_col] = pd.to_datetime(df[date_col])
        except Exception:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

    # identify other useful columns
    price_col = _find_column(df, PRICE_CANDIDATES)
    qty_col = _find_column(df, QTY_CANDIDATES)
    cost_col = _find_column(df, COST_CANDIDATES)
    product_col = _find_column(df, PRODUCT_CANDIDATES)
    customer_col = _find_column(df, CUSTOMER_CANDIDATES)
    region_col = _find_column(df, REGION_CANDIDATES)

    # create a standardized revenue column
    if value_col and value_col in df.columns:
        df['_revenue'] = pd.to_numeric(df[value_col], errors='coerce').fillna(0)
    else:
        # if price * qty available
        if price_col in df.columns and qty_col in df.columns:
            df['_revenue'] = pd.to_numeric(df[price_col], errors='coerce').fillna(0) * pd.to_numeric(df[qty_col], errors='coerce').fillna(0)
        else:
            df['_revenue'] = 0

    # create profit column if possible
    lower_cols = [c.lower() for c in df.columns]
    if 'profit' in lower_cols:
        profit_col = [c for c in df.columns if c.lower()=='profit'][0]
        df['_profit'] = pd.to_numeric(df[profit_col], errors='coerce').fillna(0)
    elif cost_col in df.columns and price_col in df.columns:
        per_unit_margin = pd.to_numeric(df[price_col], errors='coerce').fillna(0) - pd.to_numeric(df[cost_col], errors='coerce').fillna(0)
        if qty_col in df.columns:
            df['_profit'] = per_unit_margin * pd.to_numeric(df[qty_col], errors='coerce').fillna(1)
        else:
            df['_profit'] = per_unit_margin
    else:
        # fallback: assume 20% margin on revenue
        df['_profit'] = df['_revenue'] * 0.20

    # basic KPIs
    total_revenue = df['_revenue'].sum()
    total_profit = df['_profit'].sum()

    # Aggregated time-series (daily) for main metric
    if date_col in df.columns:
        ts = df.groupby(pd.Grouper(key=date_col, freq='D'))['_revenue'].sum().reset_index().dropna()
        ts.columns = [date_col, 'revenue']
    else:
        ts = pd.DataFrame({'revenue': df['_revenue']})

    # Regional sales
    region_sales = None
    if region_col in df.columns:
        region_sales = df.groupby(region_col)['_revenue'].sum().reset_index().sort_values('_revenue', ascending=False)
        region_sales.columns = [region_col, 'revenue']

    # Top products by revenue and profit
    top_products = None
    if product_col in df.columns:
        top_products = df.groupby(product_col).agg({'_revenue':'sum','_profit':'sum'}).reset_index().sort_values('_revenue', ascending=False)
        top_products.columns = [product_col, 'revenue','profit']

    # Customer segmentation (RFM-like)
    rfm = None
    if customer_col in df.columns and date_col in df.columns:
        snapshot_date = df[date_col].max() + pd.Timedelta(days=1)
        cust = df.groupby(customer_col).agg({date_col: 'max', '_revenue': 'sum', customer_col: 'count'})
        cust.columns = ['last_date', 'monetary', 'frequency']
        cust['recency'] = (snapshot_date - cust['last_date']).dt.days
        # simple scoring based on quantiles
        cust['r_score'] = pd.qcut(cust['recency'].rank(method='first'), 4, labels=[4,3,2,1]).astype(int)
        cust['f_score'] = pd.qcut(cust['frequency'].rank(method='first'), 4, labels=[1,2,3,4]).astype(int)
        cust['m_score'] = pd.qcut(cust['monetary'].rank(method='first'), 4, labels=[1,2,3,4]).astype(int)
        cust['rfm_score'] = cust['r_score']*100 + cust['f_score']*10 + cust['m_score']
        rfm = cust.reset_index().sort_values('monetary', ascending=False)

    # Highest selling product
    highest_selling = None
    if top_products is not None and not top_products.empty:
        highest_selling = top_products.iloc[0].to_dict()

    # Region with highest revenue and profit
    highest_region = None
    if region_sales is not None and not region_sales.empty:
        highest_region = region_sales.iloc[0].to_dict()

    # Visualizations
    figs = {}
    try:
        figs['timeseries'] = px.line(ts, x=ts.columns[0], y='revenue', title='Revenue over time')
    except Exception:
        figs['timeseries'] = None

    if region_sales is not None:
        figs['regional_sales'] = px.bar(region_sales, x=region_col, y='revenue', title='Regional Revenue')
    else:
        figs['regional_sales'] = None

    if top_products is not None:
        figs['top_products_revenue'] = px.bar(top_products.head(20), x=product_col, y='revenue', title='Top products by revenue')
        figs['top_products_profit'] = px.bar(top_products.head(20), x=product_col, y='profit', title='Top products by profit')
    else:
        figs['top_products_revenue'] = None
        figs['top_products_profit'] = None

    if rfm is not None:
        figs['rfm_hist'] = px.histogram(rfm, x='rfm_score', nbins=20, title='RFM scores distribution')
    else:
        figs['rfm_hist'] = None

    if top_products is not None:
        df_pareto = top_products.copy()
        df_pareto['cum_revenue'] = df_pareto['revenue'].cumsum()
        df_pareto['cum_pct'] = df_pareto['cum_revenue'] / df_pareto['revenue'].sum()
        fig = go.Figure()
        fig.add_trace(go.Bar(x=df_pareto[product_col], y=df_pareto['revenue'], name='Revenue'))
        fig.add_trace(go.Scatter(x=df_pareto[product_col], y=df_pareto['cum_pct'], name='Cum %', yaxis='y2'))
        fig.update_layout(title='Product Pareto (Revenue)', yaxis2=dict(overlaying='y', side='right', tickformat='.0%'))
        figs['pareto'] = fig
    else:
        figs['pareto'] = None

    # Forecasts for multiple horizons: monthly, quarterly, half-year (6M), yearly
    forecasts = {}
    if date_col in df.columns:
        monthly = df.groupby(pd.Grouper(key=date_col, freq='M'))['_revenue'].sum().reset_index().rename(columns={date_col:'ds','_revenue':'y'})
        quarterly = df.groupby(pd.Grouper(key=date_col, freq='Q'))['_revenue'].sum().reset_index().rename(columns={date_col:'ds','_revenue':'y'})
        halfyear = df.groupby(pd.Grouper(key=date_col, freq='6M'))['_revenue'].sum().reset_index().rename(columns={date_col:'ds','_revenue':'y'})
        yearly = df.groupby(pd.Grouper(key=date_col, freq='Y'))['_revenue'].sum().reset_index().rename(columns={date_col:'ds','_revenue':'y'})

        def _fit_prophet(series_df, periods, freq):
            if series_df.shape[0] < 2:
                return None
            m = Prophet(daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=True)
            try:
                m.fit(series_df)
                future = m.make_future_dataframe(periods=periods, freq=freq)
                fc = m.predict(future)
                return fc
            except Exception:
                return None

        forecasts['monthly'] = _fit_prophet(monthly, periods=3, freq='M')
        forecasts['quarterly'] = _fit_prophet(quarterly, periods=4, freq='Q')
        forecasts['halfyear'] = _fit_prophet(halfyear, periods=2, freq='6M')
        forecasts['yearly'] = _fit_prophet(yearly, periods=3, freq='Y')

    result = {
        'total_revenue': total_revenue,
        'total_profit': total_profit,
        'ts_df': ts,
        'region_sales': region_sales,
        'top_products': top_products,
        'highest_selling': highest_selling,
        'highest_region': highest_region,
        'rfm': rfm,
        'figs': figs,
        'forecasts': forecasts,
        'detected_columns': {
            'date_col': date_col,
            'value_col': value_col,
            'price_col': price_col,
            'qty_col': qty_col,
            'cost_col': cost_col,
            'product_col': product_col,
            'customer_col': customer_col,
            'region_col': region_col,
        }
    }

    return result
