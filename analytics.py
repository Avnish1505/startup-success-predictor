import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

import theme
from src.data import schema

DATA_PATH = "data/processed/startups_features_v1.parquet"
MIN_GROUP_SIZE = schema.MIN_COHORT_SIZE

FUNDING_BIN_EDGES = schema.FUNDING_BIN_EDGES
FUNDING_BIN_LABELS = schema.FUNDING_BIN_LABELS

AGE_BIN_EDGES = [0, 2, 5, 10, 20, np.inf]
AGE_BIN_LABELS = ["0-2y", "2-5y", "5-10y", "10-20y", "20y+"]

TOP_N_GROUPS = 15


@st.cache_data
def _load_dataset() -> pd.DataFrame:
    return pd.read_parquet(DATA_PATH)


def _grouped_success_rate(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    grouped = df.groupby(group_col, observed=True)["label"].agg(success_rate="mean", n="count").reset_index()
    return grouped[grouped["n"] >= MIN_GROUP_SIZE].sort_values("n", ascending=False)


def _bar_chart(df: pd.DataFrame, x: str, title: str, order: list[str] | None = None):
    total_n = int(df["n"].sum())
    fig = px.bar(
        df, x=x, y="success_rate", text="n",
        category_orders={x: order} if order else None,
        title=f"{title} (n={total_n:,})",
        labels={"success_rate": "Success rate"},
    )
    fig.update_traces(texttemplate="n=%{text}", textposition="outside", marker_color=theme.ACCENT)
    fig.update_yaxes(tickformat=".0%")
    theme.apply_theme(fig)
    return fig


def show_dashboard() -> None:
    """Renders the analytics dashboard from the real processed dataset - every
    number here is computed live, nothing is hand-typed."""
    st.header("Startup analytics")
    st.caption("Every number below is computed live from the real processed dataset - nothing is hand-typed.")

    df = _load_dataset()
    overall = df["label"].mean()
    st.metric("Overall success rate", f"{overall:.1%}", help=f"n={len(df):,} labeled companies (operating excluded)")
    st.caption(
        f"Coverage: {df['country_code'].notna().sum():,} of {len(df):,} rows have a country "
        f"({df['country_code'].notna().mean():.1%}), {df['primary_category'].notna().sum():,} have a category "
        f"({df['primary_category'].notna().mean():.1%}), {df['funding_total_usd_log1p'].notna().sum():,} have "
        f"funding data ({df['funding_total_usd_log1p'].notna().mean():.1%}). Each chart's own n reflects its "
        f"column's non-null, n>=30-suppressed rows - the differing totals below are expected, not an error."
    )

    st.subheader("By country")
    by_country = _grouped_success_rate(df, "country_code").head(TOP_N_GROUPS)
    st.plotly_chart(_bar_chart(by_country, "country_code", "Success rate by country"), use_container_width=True)

    st.subheader("By primary category")
    by_category = _grouped_success_rate(df, "primary_category").head(TOP_N_GROUPS)
    st.plotly_chart(_bar_chart(by_category, "primary_category", "Success rate by primary category"), use_container_width=True)

    st.subheader("By funding band")
    funding_band = pd.cut(np.expm1(df["funding_total_usd_log1p"]), bins=FUNDING_BIN_EDGES, labels=FUNDING_BIN_LABELS)
    by_funding = _grouped_success_rate(df.assign(funding_band=funding_band), "funding_band")
    st.plotly_chart(_bar_chart(by_funding, "funding_band", "Success rate by funding band", order=FUNDING_BIN_LABELS), use_container_width=True)
    st.caption(
        "Caveat: `funding_total_usd` is measured at scrape time, after the outcome is known "
        "(see DATA_CARD.md/MODEL_CARD.md's leakage finding). This is a correlation among "
        "resolved companies, not a lever a founder can causally pull by raising more."
    )

    st.subheader("By company age")
    age_years = df["founded_year"].max() - df["founded_year"]
    age_band = pd.cut(age_years, bins=AGE_BIN_EDGES, labels=AGE_BIN_LABELS, right=False)
    by_age = _grouped_success_rate(df.assign(age_band=age_band), "age_band")
    st.plotly_chart(_bar_chart(by_age, "age_band", "Success rate by company age", order=AGE_BIN_LABELS), use_container_width=True)
    st.caption(
        "Caveat: not leakage (age is known upfront), but largely mechanical: younger companies "
        "(at scrape time) have had less time to resolve to acquired/ipo at all, so this "
        "mirrors the same survivorship/censoring dynamic documented in DATA_CARD.md."
    )
