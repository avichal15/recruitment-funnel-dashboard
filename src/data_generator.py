from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker


DATABASE_PATH = Path("data/recruitment_funnel_analytics.db")
TOTAL_CLICKS = 250_000
TOTAL_APPLICATIONS = 35_000
DEFAULT_SEED = 42

SOURCE_PROFILES = {
    "Indeed": {
        "share": 0.23,
        "cpc_mean": 1.95,
        "cpc_std": 0.32,
        "apply_bias": 1.08,
        "completion_bias": 1.03,
        "qualified_bias": 0.98,
        "hire_bias": 0.94,
        "device_weights": [0.44, 0.47, 0.09],
    },
    "LinkedIn": {
        "share": 0.12,
        "cpc_mean": 3.85,
        "cpc_std": 0.48,
        "apply_bias": 0.92,
        "completion_bias": 1.06,
        "qualified_bias": 1.22,
        "hire_bias": 1.16,
        "device_weights": [0.58, 0.34, 0.08],
    },
    "Google Jobs": {
        "share": 0.14,
        "cpc_mean": 2.28,
        "cpc_std": 0.34,
        "apply_bias": 1.04,
        "completion_bias": 1.04,
        "qualified_bias": 1.01,
        "hire_bias": 1.02,
        "device_weights": [0.48, 0.43, 0.09],
    },
    "Meta": {
        "share": 0.11,
        "cpc_mean": 1.58,
        "cpc_std": 0.26,
        "apply_bias": 0.88,
        "completion_bias": 0.86,
        "qualified_bias": 0.82,
        "hire_bias": 0.8,
        "device_weights": [0.24, 0.68, 0.08],
    },
    "Organic": {
        "share": 0.09,
        "cpc_mean": 0.45,
        "cpc_std": 0.08,
        "apply_bias": 1.16,
        "completion_bias": 1.07,
        "qualified_bias": 1.17,
        "hire_bias": 1.14,
        "device_weights": [0.52, 0.39, 0.09],
    },
    "Referral": {
        "share": 0.07,
        "cpc_mean": 0.72,
        "cpc_std": 0.12,
        "apply_bias": 1.21,
        "completion_bias": 1.09,
        "qualified_bias": 1.26,
        "hire_bias": 1.22,
        "device_weights": [0.55, 0.36, 0.09],
    },
    "Email": {
        "share": 0.06,
        "cpc_mean": 0.98,
        "cpc_std": 0.15,
        "apply_bias": 1.13,
        "completion_bias": 1.08,
        "qualified_bias": 1.04,
        "hire_bias": 1.01,
        "device_weights": [0.49, 0.42, 0.09],
    },
    "Programmatic": {
        "share": 0.18,
        "cpc_mean": 2.68,
        "cpc_std": 0.37,
        "apply_bias": 0.9,
        "completion_bias": 0.91,
        "qualified_bias": 0.89,
        "hire_bias": 0.87,
        "device_weights": [0.37, 0.55, 0.08],
    },
}

DEVICE_TYPES = ["Desktop", "Mobile", "Tablet"]
DEVICE_FACTORS = {
    "Desktop": {"apply": 1.08, "completion": 1.05, "qualified": 1.04, "hire": 1.02},
    "Mobile": {"apply": 0.94, "completion": 0.9, "qualified": 0.94, "hire": 0.95},
    "Tablet": {"apply": 0.85, "completion": 0.87, "qualified": 0.88, "hire": 0.9},
}

GEO_PROFILES = [
    {"state_code": "CA", "state_name": "California", "weight": 0.16, "apply_bias": 1.05, "qualified_bias": 1.03, "hire_bias": 1.02},
    {"state_code": "TX", "state_name": "Texas", "weight": 0.12, "apply_bias": 1.02, "qualified_bias": 0.99, "hire_bias": 0.98},
    {"state_code": "NY", "state_name": "New York", "weight": 0.1, "apply_bias": 1.0, "qualified_bias": 1.05, "hire_bias": 1.04},
    {"state_code": "FL", "state_name": "Florida", "weight": 0.09, "apply_bias": 1.01, "qualified_bias": 0.97, "hire_bias": 0.96},
    {"state_code": "IL", "state_name": "Illinois", "weight": 0.07, "apply_bias": 0.99, "qualified_bias": 1.0, "hire_bias": 1.0},
    {"state_code": "WA", "state_name": "Washington", "weight": 0.06, "apply_bias": 1.03, "qualified_bias": 1.08, "hire_bias": 1.06},
    {"state_code": "GA", "state_name": "Georgia", "weight": 0.06, "apply_bias": 1.0, "qualified_bias": 0.96, "hire_bias": 0.96},
    {"state_code": "NC", "state_name": "North Carolina", "weight": 0.06, "apply_bias": 1.0, "qualified_bias": 0.98, "hire_bias": 0.97},
    {"state_code": "PA", "state_name": "Pennsylvania", "weight": 0.06, "apply_bias": 0.98, "qualified_bias": 0.99, "hire_bias": 0.98},
    {"state_code": "NJ", "state_name": "New Jersey", "weight": 0.05, "apply_bias": 0.97, "qualified_bias": 1.02, "hire_bias": 1.01},
    {"state_code": "MA", "state_name": "Massachusetts", "weight": 0.04, "apply_bias": 0.96, "qualified_bias": 1.09, "hire_bias": 1.08},
    {"state_code": "OH", "state_name": "Ohio", "weight": 0.04, "apply_bias": 0.98, "qualified_bias": 0.97, "hire_bias": 0.96},
    {"state_code": "AZ", "state_name": "Arizona", "weight": 0.04, "apply_bias": 0.99, "qualified_bias": 0.96, "hire_bias": 0.95},
    {"state_code": "CO", "state_name": "Colorado", "weight": 0.03, "apply_bias": 1.02, "qualified_bias": 1.01, "hire_bias": 1.0},
    {"state_code": "TN", "state_name": "Tennessee", "weight": 0.02, "apply_bias": 0.97, "qualified_bias": 0.95, "hire_bias": 0.94},
]

JOB_FAMILY_PROFILES = {
    "Software Engineering": {"weight": 0.18, "apply_bias": 0.96, "qualified_bias": 1.17, "hire_bias": 1.12},
    "Sales": {"weight": 0.14, "apply_bias": 1.08, "qualified_bias": 0.98, "hire_bias": 0.97},
    "Customer Success": {"weight": 0.12, "apply_bias": 1.06, "qualified_bias": 0.99, "hire_bias": 0.98},
    "Healthcare": {"weight": 0.14, "apply_bias": 1.05, "qualified_bias": 1.02, "hire_bias": 1.01},
    "Warehouse Operations": {"weight": 0.11, "apply_bias": 1.14, "qualified_bias": 0.87, "hire_bias": 0.85},
    "Marketing": {"weight": 0.1, "apply_bias": 0.97, "qualified_bias": 1.08, "hire_bias": 1.05},
    "Finance": {"weight": 0.08, "apply_bias": 0.92, "qualified_bias": 1.12, "hire_bias": 1.09},
    "Retail": {"weight": 0.13, "apply_bias": 1.1, "qualified_bias": 0.89, "hire_bias": 0.9},
}

SENIORITY_WEIGHTS = {
    "Entry": 0.34,
    "Mid": 0.42,
    "Senior": 0.18,
    "Executive": 0.06,
}


def _clamp(values: np.ndarray, lower: float, upper: float) -> np.ndarray:
    return np.minimum(np.maximum(values, lower), upper)


def _generate_pools(seed: int) -> dict[str, list[str]]:
    faker = Faker()
    faker.seed_instance(seed)

    client_names = [faker.company() for _ in range(18)]
    cities = [faker.city() for _ in range(24)]
    campaign_suffixes = ["Always On", "High Intent", "Retargeting", "Talent Expansion", "Priority Hiring"]
    campaign_names = [
        f"{client} | {job} | {suffix}"
        for client in client_names[:10]
        for job in list(JOB_FAMILY_PROFILES.keys())[:4]
        for suffix in campaign_suffixes[:2]
    ]

    return {
        "client_names": client_names,
        "cities": cities,
        "campaign_names": campaign_names,
    }


def _generate_clicks(seed: int = DEFAULT_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pools = _generate_pools(seed)

    end_date = pd.Timestamp.today().normalize()
    start_date = end_date - pd.Timedelta(days=364)

    source_names = list(SOURCE_PROFILES.keys())
    source_weights = np.array([SOURCE_PROFILES[source]["share"] for source in source_names], dtype=float)
    source_weights = source_weights / source_weights.sum()

    geo_codes = [geo["state_code"] for geo in GEO_PROFILES]
    geo_names = {geo["state_code"]: geo["state_name"] for geo in GEO_PROFILES}
    geo_weights = np.array([geo["weight"] for geo in GEO_PROFILES], dtype=float)
    geo_weights = geo_weights / geo_weights.sum()

    job_families = list(JOB_FAMILY_PROFILES.keys())
    job_weights = np.array([JOB_FAMILY_PROFILES[job]["weight"] for job in job_families], dtype=float)
    job_weights = job_weights / job_weights.sum()

    seniority_levels = list(SENIORITY_WEIGHTS.keys())
    seniority_probs = np.array(list(SENIORITY_WEIGHTS.values()), dtype=float)

    click_ids = np.arange(1, TOTAL_CLICKS + 1)
    click_timestamps = start_date + pd.to_timedelta(
        rng.integers(0, 365 * 24 * 60 * 60, size=TOTAL_CLICKS),
        unit="s",
    )

    clicks = pd.DataFrame(
        {
            "click_id": click_ids,
            "click_timestamp": click_timestamps,
            "traffic_source": rng.choice(source_names, size=TOTAL_CLICKS, p=source_weights),
            "geo_state": rng.choice(geo_codes, size=TOTAL_CLICKS, p=geo_weights),
            "job_family": rng.choice(job_families, size=TOTAL_CLICKS, p=job_weights),
            "seniority": rng.choice(seniority_levels, size=TOTAL_CLICKS, p=seniority_probs),
            "metro": rng.choice(pools["cities"], size=TOTAL_CLICKS),
            "client_name": rng.choice(pools["client_names"], size=TOTAL_CLICKS),
            "campaign_name": rng.choice(pools["campaign_names"], size=TOTAL_CLICKS),
        }
    )

    clicks["geo_name"] = clicks["geo_state"].map(geo_names)
    clicks["day_of_week"] = clicks["click_timestamp"].dt.day_name()
    clicks["hour_of_day"] = clicks["click_timestamp"].dt.hour

    device_assignments = np.empty(TOTAL_CLICKS, dtype=object)
    for source_name, profile in SOURCE_PROFILES.items():
        mask = clicks["traffic_source"] == source_name
        count = int(mask.sum())
        device_assignments[mask.to_numpy()] = rng.choice(DEVICE_TYPES, size=count, p=profile["device_weights"])
    clicks["device_type"] = device_assignments

    cpc_values = np.zeros(TOTAL_CLICKS, dtype=float)
    for source_name, profile in SOURCE_PROFILES.items():
        mask = clicks["traffic_source"] == source_name
        count = int(mask.sum())
        sampled = rng.normal(loc=profile["cpc_mean"], scale=profile["cpc_std"], size=count)
        cpc_values[mask.to_numpy()] = _clamp(sampled, 0.22, 7.5)
    clicks["cpc"] = np.round(cpc_values, 2)

    weekday_factor = np.where(clicks["day_of_week"].isin(["Tuesday", "Wednesday", "Thursday"]), 1.04, 0.98)
    hour_factor = np.where(clicks["hour_of_day"].between(8, 19), 1.03, 0.97)

    clicks["apply_weight"] = (
        clicks["traffic_source"].map({name: profile["apply_bias"] for name, profile in SOURCE_PROFILES.items()}).astype(float)
        * clicks["device_type"].map({name: profile["apply"] for name, profile in DEVICE_FACTORS.items()}).astype(float)
        * clicks["geo_state"].map({geo["state_code"]: geo["apply_bias"] for geo in GEO_PROFILES}).astype(float)
        * clicks["job_family"].map({name: profile["apply_bias"] for name, profile in JOB_FAMILY_PROFILES.items()}).astype(float)
        * weekday_factor
        * hour_factor
        * rng.uniform(0.88, 1.14, TOTAL_CLICKS)
    )

    return clicks


def _generate_applications(clicks: pd.DataFrame, seed: int = DEFAULT_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 7)

    weights = clicks["apply_weight"].to_numpy(dtype=float)
    weights = weights / weights.sum()
    selected_positions = rng.choice(clicks.index.to_numpy(), size=TOTAL_APPLICATIONS, replace=False, p=weights)
    selected_clicks = clicks.loc[selected_positions].copy().reset_index(drop=True)

    started_offset_minutes = rng.integers(4, 4 * 24 * 60, size=TOTAL_APPLICATIONS)
    completed_offset_minutes = started_offset_minutes + rng.integers(6, 85, size=TOTAL_APPLICATIONS)

    source_completion = selected_clicks["traffic_source"].map(
        {name: profile["completion_bias"] for name, profile in SOURCE_PROFILES.items()}
    ).astype(float)
    device_completion = selected_clicks["device_type"].map(
        {name: profile["completion"] for name, profile in DEVICE_FACTORS.items()}
    ).astype(float)
    completion_prob = _clamp(0.79 * source_completion * device_completion * rng.uniform(0.9, 1.08, TOTAL_APPLICATIONS), 0.42, 0.97)
    application_completed = rng.random(TOTAL_APPLICATIONS) < completion_prob

    source_qualified = selected_clicks["traffic_source"].map(
        {name: profile["qualified_bias"] for name, profile in SOURCE_PROFILES.items()}
    ).astype(float)
    device_qualified = selected_clicks["device_type"].map(
        {name: profile["qualified"] for name, profile in DEVICE_FACTORS.items()}
    ).astype(float)
    geo_qualified = selected_clicks["geo_state"].map(
        {geo["state_code"]: geo["qualified_bias"] for geo in GEO_PROFILES}
    ).astype(float)
    job_qualified = selected_clicks["job_family"].map(
        {name: profile["qualified_bias"] for name, profile in JOB_FAMILY_PROFILES.items()}
    ).astype(float)
    qualified_prob = _clamp(
        0.34 * source_qualified * device_qualified * geo_qualified * job_qualified * rng.uniform(0.9, 1.1, TOTAL_APPLICATIONS),
        0.06,
        0.76,
    )
    qualified_candidate = application_completed & (rng.random(TOTAL_APPLICATIONS) < qualified_prob)

    source_hire = selected_clicks["traffic_source"].map(
        {name: profile["hire_bias"] for name, profile in SOURCE_PROFILES.items()}
    ).astype(float)
    device_hire = selected_clicks["device_type"].map(
        {name: profile["hire"] for name, profile in DEVICE_FACTORS.items()}
    ).astype(float)
    geo_hire = selected_clicks["geo_state"].map(
        {geo["state_code"]: geo["hire_bias"] for geo in GEO_PROFILES}
    ).astype(float)
    job_hire = selected_clicks["job_family"].map(
        {name: profile["hire_bias"] for name, profile in JOB_FAMILY_PROFILES.items()}
    ).astype(float)
    hire_prob = _clamp(
        0.18 * source_hire * device_hire * geo_hire * job_hire * rng.uniform(0.92, 1.08, TOTAL_APPLICATIONS),
        0.03,
        0.42,
    )
    hire_flag = qualified_candidate & (rng.random(TOTAL_APPLICATIONS) < hire_prob)

    application_starts = selected_clicks["click_timestamp"] + pd.to_timedelta(started_offset_minutes, unit="m")
    application_completed_at = pd.Series(pd.NaT, index=selected_clicks.index, dtype="datetime64[ns]")
    application_completed_at.loc[application_completed] = (
        selected_clicks.loc[application_completed, "click_timestamp"]
        + pd.to_timedelta(completed_offset_minutes[application_completed], unit="m")
    )

    applications = pd.DataFrame(
        {
            "application_id": np.arange(1, TOTAL_APPLICATIONS + 1),
            "click_id": selected_clicks["click_id"].to_numpy(),
            "application_started_at": application_starts,
            "application_completed_at": application_completed_at,
            "application_completed": application_completed,
            "qualified_candidate": qualified_candidate,
            "hire_flag": hire_flag,
            "traffic_source": selected_clicks["traffic_source"].to_numpy(),
            "device_type": selected_clicks["device_type"].to_numpy(),
            "geo_state": selected_clicks["geo_state"].to_numpy(),
            "geo_name": selected_clicks["geo_name"].to_numpy(),
            "job_family": selected_clicks["job_family"].to_numpy(),
            "seniority": selected_clicks["seniority"].to_numpy(),
            "client_name": selected_clicks["client_name"].to_numpy(),
        }
    )

    applications["time_to_apply_hours"] = np.round(started_offset_minutes / 60, 2)
    applications["application_duration_minutes"] = np.where(
        application_completed,
        np.round(completed_offset_minutes - started_offset_minutes, 1),
        np.nan,
    )

    return applications


def _prepare_for_sql(clicks: pd.DataFrame, applications: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    clicks_sql = clicks.copy()
    applications_sql = applications.copy()

    for column in ["click_timestamp"]:
        clicks_sql[column] = clicks_sql[column].dt.strftime("%Y-%m-%d %H:%M:%S")

    for column in ["application_started_at", "application_completed_at"]:
        applications_sql[column] = pd.to_datetime(applications_sql[column]).dt.strftime("%Y-%m-%d %H:%M:%S")

    clicks_sql["cpc"] = clicks_sql["cpc"].astype(float)
    clicks_sql = clicks_sql.drop(columns=["apply_weight"])

    bool_columns = ["application_completed", "qualified_candidate", "hire_flag"]
    for column in bool_columns:
        applications_sql[column] = applications_sql[column].astype(int)

    return clicks_sql, applications_sql


def generate_database(db_path: Path = DATABASE_PATH, seed: int = DEFAULT_SEED) -> Path:
    db_path.parent.mkdir(parents=True, exist_ok=True)

    clicks = _generate_clicks(seed=seed)
    applications = _generate_applications(clicks, seed=seed)
    clicks_sql, applications_sql = _prepare_for_sql(clicks, applications)

    with sqlite3.connect(db_path) as conn:
        clicks_sql.to_sql("clicks", conn, if_exists="replace", index=False, chunksize=25_000)
        applications_sql.to_sql("applications", conn, if_exists="replace", index=False, chunksize=10_000)

        conn.executescript(
            """
            CREATE INDEX IF NOT EXISTS idx_clicks_click_id ON clicks(click_id);
            CREATE INDEX IF NOT EXISTS idx_clicks_timestamp ON clicks(click_timestamp);
            CREATE INDEX IF NOT EXISTS idx_clicks_source ON clicks(traffic_source);
            CREATE INDEX IF NOT EXISTS idx_applications_click_id ON applications(click_id);
            CREATE INDEX IF NOT EXISTS idx_applications_started_at ON applications(application_started_at);
            CREATE INDEX IF NOT EXISTS idx_applications_source ON applications(traffic_source);
            """
        )

    return db_path


def _row_count(conn: sqlite3.Connection, table_name: str) -> int:
    result = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()
    return int(result[0]) if result else 0


def ensure_database(db_path: Path = DATABASE_PATH, force_rebuild: bool = False) -> Path:
    if force_rebuild or not db_path.exists():
        return generate_database(db_path=db_path)

    try:
        with sqlite3.connect(db_path) as conn:
            click_count = _row_count(conn, "clicks")
            application_count = _row_count(conn, "applications")
        if click_count != TOTAL_CLICKS or application_count != TOTAL_APPLICATIONS:
            return generate_database(db_path=db_path)
    except sqlite3.DatabaseError:
        return generate_database(db_path=db_path)

    return db_path


def load_dataset(db_path: Path = DATABASE_PATH) -> tuple[pd.DataFrame, pd.DataFrame]:
    with sqlite3.connect(db_path) as conn:
        clicks = pd.read_sql_query("SELECT * FROM clicks", conn, parse_dates=["click_timestamp"])
        applications = pd.read_sql_query(
            "SELECT * FROM applications",
            conn,
            parse_dates=["application_started_at", "application_completed_at"],
        )

    bool_columns = ["application_completed", "qualified_candidate", "hire_flag"]
    for column in bool_columns:
        applications[column] = applications[column].astype(bool)

    return clicks, applications

