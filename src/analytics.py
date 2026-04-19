from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(slots=True)
class FilterState:
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    sources: list[str]
    devices: list[str]
    geos: list[str]
    job_families: list[str]


def apply_filters(
    clicks: pd.DataFrame,
    applications: pd.DataFrame,
    filters: FilterState,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    click_mask = (
        clicks["click_timestamp"].dt.normalize().between(filters.start_date, filters.end_date)
        & clicks["traffic_source"].isin(filters.sources)
        & clicks["device_type"].isin(filters.devices)
        & clicks["geo_state"].isin(filters.geos)
        & clicks["job_family"].isin(filters.job_families)
    )
    filtered_clicks = clicks.loc[click_mask].copy()
    filtered_apps = applications.loc[applications["click_id"].isin(filtered_clicks["click_id"])].copy()
    return filtered_clicks, filtered_apps


def previous_period(filters: FilterState) -> FilterState:
    days = int((filters.end_date - filters.start_date).days) + 1
    previous_end = filters.start_date - pd.Timedelta(days=1)
    previous_start = previous_end - pd.Timedelta(days=days - 1)
    return FilterState(
        start_date=previous_start,
        end_date=previous_end,
        sources=filters.sources,
        devices=filters.devices,
        geos=filters.geos,
        job_families=filters.job_families,
    )


def summarize_funnel(clicks: pd.DataFrame, applications: pd.DataFrame) -> dict[str, float]:
    clicks_count = int(len(clicks))
    applications_started = int(len(applications))
    completed = int(applications["application_completed"].sum())
    qualified = int(applications["qualified_candidate"].sum())
    hires = int(applications["hire_flag"].sum())
    spend = float(clicks["cpc"].sum())

    return {
        "clicks": clicks_count,
        "applications": applications_started,
        "completed_applications": completed,
        "qualified": qualified,
        "hires": hires,
        "spend": spend,
        "apply_rate": applications_started / clicks_count if clicks_count else 0.0,
        "completion_rate": completed / applications_started if applications_started else 0.0,
        "qualification_rate": qualified / completed if completed else 0.0,
        "hire_rate": hires / qualified if qualified else 0.0,
        "click_to_hire_rate": hires / clicks_count if clicks_count else 0.0,
        "cost_per_applicant": spend / completed if completed else 0.0,
        "cost_per_hire": spend / hires if hires else 0.0,
    }


def period_delta(current: float, previous: float) -> str:
    if previous == 0:
        return "New"
    delta = ((current - previous) / previous) * 100
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.1f}% vs prior"


def build_daily_timeseries(clicks: pd.DataFrame, applications: pd.DataFrame) -> pd.DataFrame:
    click_daily = (
        clicks.assign(date=clicks["click_timestamp"].dt.date)
        .groupby("date", as_index=False)
        .agg(clicks=("click_id", "count"), spend=("cpc", "sum"))
    )

    app_daily = (
        applications.assign(date=applications["application_started_at"].dt.date)
        .groupby("date", as_index=False)
        .agg(
            applications=("application_id", "count"),
            completed=("application_completed", "sum"),
            qualified=("qualified_candidate", "sum"),
            hires=("hire_flag", "sum"),
        )
    )

    merged = click_daily.merge(app_daily, how="left", on="date").fillna(0)
    return merged.sort_values("date")


def build_source_metrics(clicks: pd.DataFrame, applications: pd.DataFrame) -> pd.DataFrame:
    click_summary = (
        clicks.groupby("traffic_source", as_index=False)
        .agg(clicks=("click_id", "count"), spend=("cpc", "sum"), avg_cpc=("cpc", "mean"))
    )

    app_summary = (
        applications.groupby("traffic_source", as_index=False)
        .agg(
            applications=("application_id", "count"),
            completed=("application_completed", "sum"),
            qualified=("qualified_candidate", "sum"),
            hires=("hire_flag", "sum"),
        )
    )

    merged = click_summary.merge(app_summary, how="left", on="traffic_source").fillna(0)
    merged["apply_rate"] = np.where(merged["clicks"] > 0, merged["applications"] / merged["clicks"], 0.0)
    merged["completion_rate"] = np.where(merged["applications"] > 0, merged["completed"] / merged["applications"], 0.0)
    merged["qualified_rate"] = np.where(merged["completed"] > 0, merged["qualified"] / merged["completed"], 0.0)
    merged["hire_rate"] = np.where(merged["qualified"] > 0, merged["hires"] / merged["qualified"], 0.0)
    merged["cost_per_applicant"] = np.where(merged["completed"] > 0, merged["spend"] / merged["completed"], 0.0)
    merged["cost_per_hire"] = np.where(merged["hires"] > 0, merged["spend"] / merged["hires"], 0.0)
    return merged.sort_values("cost_per_applicant", ascending=False)


def build_device_conversion(clicks: pd.DataFrame, applications: pd.DataFrame) -> pd.DataFrame:
    click_summary = clicks.groupby("device_type", as_index=False).agg(clicks=("click_id", "count"))
    app_summary = (
        applications.groupby("device_type", as_index=False)
        .agg(
            applications=("application_id", "count"),
            completed=("application_completed", "sum"),
            qualified=("qualified_candidate", "sum"),
            hires=("hire_flag", "sum"),
        )
    )
    merged = click_summary.merge(app_summary, how="left", on="device_type").fillna(0)
    merged["click_to_apply"] = np.where(merged["clicks"] > 0, merged["applications"] / merged["clicks"], 0.0)
    merged["apply_to_complete"] = np.where(merged["applications"] > 0, merged["completed"] / merged["applications"], 0.0)
    merged["complete_to_qualified"] = np.where(merged["completed"] > 0, merged["qualified"] / merged["completed"], 0.0)
    merged["qualified_to_hire"] = np.where(merged["qualified"] > 0, merged["hires"] / merged["qualified"], 0.0)
    return merged.sort_values("click_to_apply", ascending=False)


def build_geo_summary(clicks: pd.DataFrame, applications: pd.DataFrame) -> pd.DataFrame:
    click_summary = clicks.groupby(["geo_state", "geo_name"], as_index=False).agg(clicks=("click_id", "count"), spend=("cpc", "sum"))
    app_summary = (
        applications.groupby(["geo_state", "geo_name"], as_index=False)
        .agg(
            applications=("application_id", "count"),
            completed=("application_completed", "sum"),
            qualified=("qualified_candidate", "sum"),
            hires=("hire_flag", "sum"),
        )
    )
    merged = click_summary.merge(app_summary, how="left", on=["geo_state", "geo_name"]).fillna(0)
    merged["apply_rate"] = np.where(merged["clicks"] > 0, merged["applications"] / merged["clicks"], 0.0)
    merged["qualified_rate"] = np.where(merged["completed"] > 0, merged["qualified"] / merged["completed"], 0.0)
    merged["cost_per_applicant"] = np.where(merged["completed"] > 0, merged["spend"] / merged["completed"], 0.0)
    return merged.sort_values("qualified", ascending=False)


def build_dropoff_summary(clicks: pd.DataFrame, applications: pd.DataFrame) -> pd.DataFrame:
    funnel = summarize_funnel(clicks, applications)
    stages = [
        ("Clicks", funnel["clicks"]),
        ("Applications", funnel["applications"]),
        ("Completed", funnel["completed_applications"]),
        ("Qualified", funnel["qualified"]),
        ("Hires", funnel["hires"]),
    ]

    rows = []
    for index in range(len(stages) - 1):
        stage, count = stages[index]
        next_stage, next_count = stages[index + 1]
        dropoff = count - next_count
        rate = dropoff / count if count else 0.0
        rows.append(
            {
                "from_stage": stage,
                "to_stage": next_stage,
                "from_count": count,
                "to_count": next_count,
                "dropoff_count": dropoff,
                "dropoff_rate": rate,
            }
        )
    return pd.DataFrame(rows)


def build_source_stage_rates(clicks: pd.DataFrame, applications: pd.DataFrame) -> pd.DataFrame:
    click_summary = clicks.groupby("traffic_source", as_index=False).agg(clicks=("click_id", "count"))
    app_summary = (
        applications.groupby("traffic_source", as_index=False)
        .agg(
            applications=("application_id", "count"),
            completed=("application_completed", "sum"),
            qualified=("qualified_candidate", "sum"),
            hires=("hire_flag", "sum"),
        )
    )
    merged = click_summary.merge(app_summary, how="left", on="traffic_source").fillna(0)

    long_rows = []
    for _, row in merged.iterrows():
        stage_rates = {
            "Click → Apply": row["applications"] / row["clicks"] if row["clicks"] else 0.0,
            "Apply → Complete": row["completed"] / row["applications"] if row["applications"] else 0.0,
            "Complete → Qualified": row["qualified"] / row["completed"] if row["completed"] else 0.0,
            "Qualified → Hire": row["hires"] / row["qualified"] if row["qualified"] else 0.0,
        }
        for stage, value in stage_rates.items():
            long_rows.append({"traffic_source": row["traffic_source"], "stage": stage, "rate": value})
    return pd.DataFrame(long_rows)


def build_recommendations(
    funnel: dict[str, float],
    source_metrics: pd.DataFrame,
    device_metrics: pd.DataFrame,
    geo_metrics: pd.DataFrame,
    dropoff_summary: pd.DataFrame,
) -> list[dict[str, str]]:
    recommendations: list[dict[str, str]] = []

    expensive_sources = source_metrics.loc[source_metrics["completed"] >= 200]
    if not expensive_sources.empty:
        worst_cpa = expensive_sources.sort_values("cost_per_applicant", ascending=False).iloc[0]
        median_cpa = expensive_sources["cost_per_applicant"].median()
        if worst_cpa["cost_per_applicant"] > median_cpa * 1.18:
            recommendations.append(
                {
                    "priority": "High",
                    "title": f"Trim inefficient spend in {worst_cpa['traffic_source']}",
                    "detail": (
                        f"{worst_cpa['traffic_source']} is running at ${worst_cpa['cost_per_applicant']:.2f} per completed applicant, "
                        f"well above the cohort median of ${median_cpa:.2f}. Pull bids back or tighten targeting before scaling."
                    ),
                }
            )

    if not device_metrics.empty:
        weakest_device = device_metrics.sort_values("apply_to_complete").iloc[0]
        strongest_device = device_metrics.sort_values("apply_to_complete", ascending=False).iloc[0]
        if strongest_device["apply_to_complete"] - weakest_device["apply_to_complete"] > 0.08:
            recommendations.append(
                {
                    "priority": "Medium",
                    "title": f"Reduce form friction on {weakest_device['device_type']}",
                    "detail": (
                        f"{weakest_device['device_type']} completion is {weakest_device['apply_to_complete']:.1%}, "
                        f"vs {strongest_device['apply_to_complete']:.1%} on {strongest_device['device_type']}. "
                        "A shorter apply flow or autofill prompts should recover abandoned starts."
                    ),
                }
            )

    if not geo_metrics.empty:
        scalable_geo = geo_metrics.loc[(geo_metrics["qualified_rate"] > geo_metrics["qualified_rate"].median()) & (geo_metrics["clicks"] > 7_500)]
        if not scalable_geo.empty:
            top_geo = scalable_geo.sort_values("qualified_rate", ascending=False).iloc[0]
            recommendations.append(
                {
                    "priority": "Medium",
                    "title": f"Scale budget into {top_geo['geo_name']}",
                    "detail": (
                        f"{top_geo['geo_name']} shows a {top_geo['qualified_rate']:.1%} completed-to-qualified rate "
                        f"with {int(top_geo['clicks']):,} clicks. This market is converting efficiently enough to justify budget expansion."
                    ),
                }
            )

    if not dropoff_summary.empty:
        largest_dropoff = dropoff_summary.sort_values("dropoff_rate", ascending=False).iloc[0]
        if largest_dropoff["dropoff_rate"] > 0.55:
            recommendations.append(
                {
                    "priority": "High",
                    "title": f"Focus optimization between {largest_dropoff['from_stage']} and {largest_dropoff['to_stage']}",
                    "detail": (
                        f"The steepest stage loss is {largest_dropoff['dropoff_rate']:.1%}, "
                        f"which indicates the fastest ROI will likely come from fixing that transition first."
                    ),
                }
            )

    if funnel["click_to_hire_rate"] < 0.01:
        recommendations.append(
            {
                "priority": "Medium",
                "title": "Tighten top-of-funnel traffic qualification",
                "detail": (
                    f"Click-to-hire is {funnel['click_to_hire_rate']:.2%}. Use pre-qualifying ad copy, stronger landing-page matching, "
                    "and source-level suppression lists to reduce low-intent traffic."
                ),
            }
        )

    if not recommendations:
        recommendations.append(
            {
                "priority": "Low",
                "title": "Performance is balanced across the funnel",
                "detail": "No major efficiency outliers were detected in the current filter set. Focus on controlled scaling and creative refresh tests.",
            }
        )

    return recommendations[:5]


def format_number(value: float) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{value:,.0f}"


def format_percent(value: float) -> str:
    return f"{value:.1%}"

