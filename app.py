from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.analytics import (
    FilterState,
    apply_filters,
    build_daily_timeseries,
    build_device_conversion,
    build_dropoff_summary,
    build_geo_summary,
    build_recommendations,
    build_source_metrics,
    build_source_stage_rates,
    format_number,
    format_percent,
    period_delta,
    previous_period,
    summarize_funnel,
)
from src.data_generator import DATABASE_PATH, ensure_database, load_dataset


PALETTE = {
    "teal": "#0F766E",
    "green": "#3F8C59",
    "forest": "#102A24",
    "mint": "#D7EFE7",
    "sand": "#F3E8D7",
    "amber": "#D97706",
    "rose": "#BE4B49",
    "ink": "#1F2937",
}

SOURCE_COLORS = {
    "Indeed": "#0F766E",
    "LinkedIn": "#14532D",
    "Google Jobs": "#1D4ED8",
    "Meta": "#BE4B49",
    "Organic": "#3F8C59",
    "Referral": "#7C3AED",
    "Email": "#D97706",
    "Programmatic": "#334155",
}


st.set_page_config(
    page_title="Recruitment Funnel Analytics Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@500;600;700;800&display=swap');

        html, body, [class*="css"]  {
            font-family: 'Manrope', sans-serif;
        }

        [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(circle at top left, rgba(15,118,110,0.16), transparent 28%),
                radial-gradient(circle at top right, rgba(63,140,89,0.12), transparent 20%),
                linear-gradient(180deg, #F7FBF9 0%, #F2F7F4 100%);
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(16,42,36,0.98) 0%, rgba(12,28,24,0.98) 100%);
            border-right: 1px solid rgba(255,255,255,0.08);
        }

        [data-testid="stSidebar"] * {
            color: #F8FBFA;
        }

        .block-container {
            max-width: 1420px;
            padding-top: 1.4rem;
            padding-bottom: 2.2rem;
        }

        .hero {
            background: linear-gradient(135deg, #102A24 0%, #0F766E 52%, #3F8C59 100%);
            border-radius: 26px;
            padding: 1.7rem 1.8rem;
            color: white;
            box-shadow: 0 20px 50px rgba(16,42,36,0.18);
            margin-bottom: 1rem;
        }

        .hero h1 {
            margin: 0;
            font-size: 2rem;
            font-weight: 800;
            letter-spacing: -0.04em;
        }

        .hero p {
            margin: 0.45rem 0 0 0;
            max-width: 920px;
            line-height: 1.55;
            color: rgba(255,255,255,0.88);
        }

        .hero-strip {
            display: flex;
            gap: 0.85rem;
            flex-wrap: wrap;
            margin-top: 1rem;
        }

        .hero-pill {
            background: rgba(255,255,255,0.14);
            border: 1px solid rgba(255,255,255,0.14);
            border-radius: 999px;
            padding: 0.45rem 0.8rem;
            font-size: 0.88rem;
        }

        .metric-card {
            background: rgba(255,255,255,0.82);
            border: 1px solid rgba(16,42,36,0.08);
            border-radius: 22px;
            padding: 1rem 1.1rem;
            box-shadow: 0 12px 34px rgba(16,42,36,0.08);
            min-height: 150px;
            backdrop-filter: blur(12px);
        }

        .metric-label {
            color: #4C635D;
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-weight: 700;
        }

        .metric-value {
            color: #102A24;
            font-size: 2rem;
            font-weight: 800;
            letter-spacing: -0.04em;
            margin-top: 0.55rem;
        }

        .metric-detail {
            color: #37514A;
            font-size: 0.95rem;
            margin-top: 0.55rem;
        }

        .metric-delta {
            font-size: 0.88rem;
            margin-top: 0.6rem;
            font-weight: 700;
        }

        .metric-delta.good {
            color: #0F766E;
        }

        .metric-delta.warn {
            color: #BE4B49;
        }

        .section-card {
            background: rgba(255,255,255,0.78);
            border: 1px solid rgba(16,42,36,0.08);
            border-radius: 24px;
            padding: 0.75rem;
            box-shadow: 0 12px 30px rgba(16,42,36,0.08);
        }

        .insight-card {
            background: rgba(255,255,255,0.92);
            border-radius: 20px;
            padding: 1rem 1.1rem;
            border: 1px solid rgba(16,42,36,0.08);
            border-left: 5px solid #0F766E;
            margin-bottom: 0.85rem;
            box-shadow: 0 10px 26px rgba(16,42,36,0.06);
        }

        .insight-priority {
            display: inline-block;
            padding: 0.22rem 0.55rem;
            border-radius: 999px;
            font-size: 0.75rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin-bottom: 0.6rem;
            background: #E8F1EE;
            color: #0F766E;
        }

        .insight-title {
            color: #102A24;
            font-size: 1rem;
            font-weight: 800;
            margin-bottom: 0.35rem;
        }

        .insight-detail {
            color: #36514A;
            line-height: 1.6;
            font-size: 0.92rem;
        }

        .tab-note {
            color: #4C635D;
            font-size: 0.94rem;
            margin-bottom: 0.35rem;
        }

        div[data-testid="stMetric"] {
            background: white;
            border-radius: 18px;
            padding: 0.8rem 1rem;
            border: 1px solid rgba(16,42,36,0.08);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def metric_card(title: str, value: str, detail: str, delta: str | None = None, positive: bool = True) -> None:
    delta_class = "good" if positive else "warn"
    delta_html = f"<div class='metric-delta {delta_class}'>{delta}</div>" if delta else ""
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{title}</div>
            <div class="metric-value">{value}</div>
            {delta_html}
            <div class="metric-detail">{detail}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def ensure_db_ready(force_rebuild: bool = False) -> str:
    return str(ensure_database(force_rebuild=force_rebuild))


@st.cache_data(show_spinner=False)
def load_data(_: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    clicks_df, applications_df = load_dataset(DATABASE_PATH)
    return clicks_df, applications_df


def create_line_chart(timeseries: pd.DataFrame):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=timeseries["date"],
            y=timeseries["clicks"],
            mode="lines",
            name="Clicks",
            line=dict(color=PALETTE["forest"], width=2.5),
            fill="tozeroy",
            fillcolor="rgba(16,42,36,0.05)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=timeseries["date"],
            y=timeseries["applications"],
            mode="lines",
            name="Applications",
            line=dict(color=PALETTE["teal"], width=2.5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=timeseries["date"],
            y=timeseries["qualified"],
            mode="lines",
            name="Qualified",
            line=dict(color=PALETTE["amber"], width=2.3),
        )
    )
    fig.update_layout(
        margin=dict(l=10, r=10, t=15, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=360,
        yaxis_title=None,
        xaxis_title=None,
    )
    return fig


def create_funnel_chart(funnel: dict[str, float]):
    fig = go.Figure(
        go.Funnel(
            y=["Clicks", "Applications", "Completed", "Qualified", "Hires"],
            x=[
                funnel["clicks"],
                funnel["applications"],
                funnel["completed_applications"],
                funnel["qualified"],
                funnel["hires"],
            ],
            textinfo="value+percent initial",
            marker={"color": [PALETTE["forest"], PALETTE["teal"], "#19A48B", PALETTE["amber"], PALETTE["rose"]]},
            opacity=0.9,
        )
    )
    fig.update_layout(
        margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=360,
    )
    return fig


def create_cpa_chart(source_metrics: pd.DataFrame):
    ordered = source_metrics.sort_values("cost_per_applicant", ascending=True)
    fig = px.bar(
        ordered,
        x="cost_per_applicant",
        y="traffic_source",
        color="traffic_source",
        text="cost_per_applicant",
        orientation="h",
        color_discrete_map=SOURCE_COLORS,
        custom_data=["completed", "qualified_rate", "avg_cpc"],
    )
    fig.update_traces(
        texttemplate="$%{text:.2f}",
        textposition="outside",
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Cost / applicant: $%{x:.2f}<br>"
            "Completed applicants: %{customdata[0]:,.0f}<br>"
            "Qualified rate: %{customdata[1]:.1%}<br>"
            "Avg CPC: $%{customdata[2]:.2f}<extra></extra>"
        ),
    )
    fig.update_layout(
        showlegend=False,
        margin=dict(l=10, r=20, t=15, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title=None,
        yaxis_title=None,
        height=420,
    )
    return fig


def create_device_chart(device_metrics: pd.DataFrame):
    melted = device_metrics.melt(
        id_vars="device_type",
        value_vars=["click_to_apply", "apply_to_complete", "complete_to_qualified", "qualified_to_hire"],
        var_name="stage",
        value_name="rate",
    )
    label_map = {
        "click_to_apply": "Click → Apply",
        "apply_to_complete": "Apply → Complete",
        "complete_to_qualified": "Complete → Qualified",
        "qualified_to_hire": "Qualified → Hire",
    }
    melted["stage"] = melted["stage"].map(label_map)
    fig = px.bar(
        melted,
        x="device_type",
        y="rate",
        color="stage",
        barmode="group",
        text="rate",
        color_discrete_sequence=[PALETTE["forest"], PALETTE["teal"], PALETTE["amber"], PALETTE["rose"]],
    )
    fig.update_traces(texttemplate="%{text:.1%}", textposition="outside")
    fig.update_layout(
        margin=dict(l=10, r=10, t=15, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis_tickformat=".0%",
        yaxis_title=None,
        xaxis_title=None,
        legend_title=None,
        height=420,
    )
    return fig


def create_geo_chart(geo_summary: pd.DataFrame):
    fig = px.choropleth(
        geo_summary,
        locations="geo_state",
        locationmode="USA-states",
        scope="usa",
        color="qualified",
        hover_name="geo_name",
        hover_data={
            "clicks": ":,.0f",
            "applications": ":,.0f",
            "completed": ":,.0f",
            "qualified_rate": ":.1%",
            "cost_per_applicant": ":.2f",
            "geo_state": False,
            "qualified": ":,.0f",
        },
        color_continuous_scale=["#DFF5EE", "#8ED1BD", "#0F766E", "#102A24"],
    )
    fig.update_layout(
        margin=dict(l=10, r=10, t=15, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        geo_bgcolor="rgba(0,0,0,0)",
        height=430,
        coloraxis_colorbar_title="Qualified",
    )
    return fig


def create_dropoff_chart(dropoff_summary: pd.DataFrame):
    fig = px.bar(
        dropoff_summary,
        x="dropoff_count",
        y="from_stage",
        color="dropoff_rate",
        orientation="h",
        text="dropoff_rate",
        color_continuous_scale=["#D7EFE7", "#D97706", "#BE4B49"],
    )
    fig.update_traces(texttemplate="%{text:.1%}", textposition="outside")
    fig.update_layout(
        margin=dict(l=10, r=10, t=15, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Users lost",
        yaxis_title=None,
        height=320,
        coloraxis_colorbar_title="Dropoff",
    )
    return fig


def create_source_stage_heatmap(stage_rates: pd.DataFrame):
    pivoted = stage_rates.pivot(index="traffic_source", columns="stage", values="rate")
    fig = px.imshow(
        pivoted,
        text_auto=".1%",
        color_continuous_scale=["#F3F7F5", "#A7DCCB", "#0F766E", "#102A24"],
        aspect="auto",
    )
    fig.update_layout(
        margin=dict(l=10, r=10, t=15, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title=None,
        yaxis_title=None,
        height=410,
        coloraxis_colorbar_title="Rate",
    )
    return fig


def render_recommendations(recommendations: list[dict[str, str]]) -> None:
    for item in recommendations:
        st.markdown(
            f"""
            <div class="insight-card">
                <div class="insight-priority">{item["priority"]}</div>
                <div class="insight-title">{item["title"]}</div>
                <div class="insight-detail">{item["detail"]}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


inject_styles()

with st.spinner("Preparing SQLite dataset and dashboard context..."):
    database_key = ensure_db_ready()
    clicks, applications = load_data(database_key)

clicks["click_date"] = clicks["click_timestamp"].dt.normalize()
date_min = clicks["click_date"].min().date()
date_max = clicks["click_date"].max().date()

with st.sidebar:
    st.markdown("## Dashboard Controls")
    if st.button("Regenerate Synthetic Data", width="stretch"):
        ensure_db_ready.clear()
        load_data.clear()
        with st.spinner("Rebuilding 250,000 clicks and 35,000 applications..."):
            database_key = ensure_db_ready(force_rebuild=True)
            clicks, applications = load_data(database_key)
            clicks["click_date"] = clicks["click_timestamp"].dt.normalize()
        st.success("Dataset regenerated.")
        st.rerun()

    selected_dates = st.date_input(
        "Date range",
        value=(date_min, date_max),
        min_value=date_min,
        max_value=date_max,
    )
    if len(selected_dates) != 2:
        selected_dates = (date_min, date_max)

    sources = sorted(clicks["traffic_source"].unique().tolist())
    devices = sorted(clicks["device_type"].unique().tolist())
    geos = sorted(clicks["geo_state"].unique().tolist())
    job_families = sorted(clicks["job_family"].unique().tolist())

    selected_sources = st.multiselect("Traffic source", options=sources, default=sources)
    selected_devices = st.multiselect("Device type", options=devices, default=devices)
    selected_geos = st.multiselect("Geo", options=geos, default=geos)
    selected_job_families = st.multiselect("Job family", options=job_families, default=job_families)

    st.markdown("---")
    st.caption("Synthetic data is generated locally with Faker, Pandas, and SQLite for portfolio/demo use.")
    st.caption(f"SQLite database: `{Path(database_key).as_posix()}`")


filters = FilterState(
    start_date=pd.Timestamp(selected_dates[0]),
    end_date=pd.Timestamp(selected_dates[1]),
    sources=selected_sources or sources,
    devices=selected_devices or devices,
    geos=selected_geos or geos,
    job_families=selected_job_families or job_families,
)

filtered_clicks, filtered_applications = apply_filters(clicks, applications, filters)
previous_clicks, previous_applications = apply_filters(clicks, applications, previous_period(filters))

funnel = summarize_funnel(filtered_clicks, filtered_applications)
previous_funnel = summarize_funnel(previous_clicks, previous_applications)
timeseries = build_daily_timeseries(filtered_clicks, filtered_applications)
source_metrics = build_source_metrics(filtered_clicks, filtered_applications)
device_metrics = build_device_conversion(filtered_clicks, filtered_applications)
geo_summary = build_geo_summary(filtered_clicks, filtered_applications)
dropoff_summary = build_dropoff_summary(filtered_clicks, filtered_applications)
source_stage_rates = build_source_stage_rates(filtered_clicks, filtered_applications)
recommendations = build_recommendations(funnel, source_metrics, device_metrics, geo_summary, dropoff_summary)

st.markdown(
    f"""
    <div class="hero">
        <h1>Recruitment Funnel Analytics Dashboard</h1>
        <p>
            A portfolio-grade recruitment marketing dashboard for an ad-tech team tracking how paid and owned traffic
            flows from click through application, qualification, and hire. Use the filters to isolate source, device,
            geo, and job-family performance across the funnel.
        </p>
        <div class="hero-strip">
            <div class="hero-pill">Spend tracked: ${funnel["spend"]:,.0f}</div>
            <div class="hero-pill">Completion rate: {format_percent(funnel["completion_rate"])}</div>
            <div class="hero-pill">Click → Hire: {format_percent(funnel["click_to_hire_rate"])}</div>
            <div class="hero-pill">Cost per hire: ${funnel["cost_per_hire"]:,.0f}</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

metric_columns = st.columns(5)
with metric_columns[0]:
    metric_card("Clicks", format_number(funnel["clicks"]), "Top-of-funnel ad traffic", period_delta(funnel["clicks"], previous_funnel["clicks"]), True)
with metric_columns[1]:
    metric_card("Applications", format_number(funnel["applications"]), f"Apply rate {format_percent(funnel['apply_rate'])}", period_delta(funnel["applications"], previous_funnel["applications"]), True)
with metric_columns[2]:
    metric_card("Qualified", format_number(funnel["qualified"]), f"Qualified rate {format_percent(funnel['qualification_rate'])}", period_delta(funnel["qualified"], previous_funnel["qualified"]), True)
with metric_columns[3]:
    metric_card("Hires", format_number(funnel["hires"]), f"Qualified → Hire {format_percent(funnel['hire_rate'])}", period_delta(funnel["hires"], previous_funnel["hires"]), True)
with metric_columns[4]:
    metric_card(
        "Completed Apply Rate",
        format_percent(funnel["completion_rate"]),
        f"Cost per applicant ${funnel['cost_per_applicant']:.2f}",
        period_delta(funnel["completion_rate"], previous_funnel["completion_rate"]),
        funnel["completion_rate"] >= previous_funnel["completion_rate"],
    )

overview_tab, channel_tab, geo_tab, insight_tab = st.tabs(
    ["Executive View", "Channel Performance", "Geo & Dropoff", "Recommendations"]
)

with overview_tab:
    st.markdown("<div class='tab-note'>Daily funnel momentum and the current stage-by-stage shape of the recruitment pipeline.</div>", unsafe_allow_html=True)
    overview_left, overview_right = st.columns([1.5, 1], gap="large")
    with overview_left:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.subheader("Volume Trend")
        st.plotly_chart(create_line_chart(timeseries), width="stretch", config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)
    with overview_right:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.subheader("Funnel Snapshot")
        st.plotly_chart(create_funnel_chart(funnel), width="stretch", config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    source_table = source_metrics.sort_values("hires", ascending=False)[
        ["traffic_source", "clicks", "completed", "qualified", "hires", "cost_per_applicant", "cost_per_hire"]
    ].copy()
    source_table.columns = ["Source", "Clicks", "Completed Applies", "Qualified", "Hires", "Cost / Applicant", "Cost / Hire"]
    st.dataframe(
        source_table.style.format(
            {
                "Clicks": "{:,.0f}",
                "Completed Applies": "{:,.0f}",
                "Qualified": "{:,.0f}",
                "Hires": "{:,.0f}",
                "Cost / Applicant": "${:,.2f}",
                "Cost / Hire": "${:,.0f}",
            }
        ),
        width="stretch",
        hide_index=True,
    )

with channel_tab:
    st.markdown("<div class='tab-note'>Source efficiency and device conversion patterns help isolate where media and UX improvements will pay back fastest.</div>", unsafe_allow_html=True)
    channel_left, channel_right = st.columns(2, gap="large")
    with channel_left:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.subheader("Cost per Applicant by Source")
        st.plotly_chart(create_cpa_chart(source_metrics), width="stretch", config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)
    with channel_right:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.subheader("Conversion by Device")
        st.plotly_chart(create_device_chart(device_metrics), width="stretch", config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

with geo_tab:
    st.markdown("<div class='tab-note'>Geo concentration and stage leakage reveal where demand quality changes across markets and funnel steps.</div>", unsafe_allow_html=True)
    geo_left, geo_right = st.columns([1.1, 0.9], gap="large")
    with geo_left:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.subheader("Qualified Candidates by State")
        st.plotly_chart(create_geo_chart(geo_summary), width="stretch", config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)
    with geo_right:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.subheader("Dropoff Analysis")
        st.plotly_chart(create_dropoff_chart(dropoff_summary), width="stretch", config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("Source Stage Conversion Matrix")
    st.plotly_chart(create_source_stage_heatmap(source_stage_rates), width="stretch", config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)

with insight_tab:
    insight_left, insight_right = st.columns([1.15, 0.85], gap="large")
    with insight_left:
        st.subheader("Rule-Based Recommendations")
        render_recommendations(recommendations)
    with insight_right:
        st.subheader("Executive Benchmarks")
        st.metric("Total spend", f"${funnel['spend']:,.0f}", period_delta(funnel["spend"], previous_funnel["spend"]), delta_color="inverse")
        st.metric(
            "Cost per applicant",
            f"${funnel['cost_per_applicant']:.2f}",
            period_delta(funnel["cost_per_applicant"], previous_funnel["cost_per_applicant"]),
            delta_color="inverse",
        )
        st.metric(
            "Cost per hire",
            f"${funnel['cost_per_hire']:,.0f}",
            period_delta(funnel["cost_per_hire"], previous_funnel["cost_per_hire"]),
            delta_color="inverse",
        )
        st.metric("Completed applies", f"{funnel['completed_applications']:,.0f}", period_delta(funnel["completed_applications"], previous_funnel["completed_applications"]))

        top_source = source_metrics.sort_values("hires", ascending=False).iloc[0] if not source_metrics.empty else None
        if top_source is not None:
            st.markdown("### Best Hiring Source")
            st.write(
                f"**{top_source['traffic_source']}** generated {int(top_source['hires']):,} hires at "
                f"${top_source['cost_per_hire']:,.0f} cost per hire with a {top_source['qualified_rate']:.1%} qualified rate."
            )

st.caption("Built with Streamlit, Pandas, Plotly, SQLite, and Faker. Data is synthetic and generated locally for portfolio demonstration.")
