import pandas as pd
import numpy as np
import re


def normalize_resource_name(name):
    if name is None:
        return "Unknown"
    name = str(name).strip()
    name = re.sub(r"\s+", " ", name)
    if name.lower() in ["", "nan", "none", "null", "nat"]:
        return "Unknown"
    return name


def detect_resource_log_columns(log_df):
    """
    Detect normalized columns or PM4Py/XES columns.
    """

    if "case_id" in log_df.columns:
        case_col = "case_id"
    elif "case:concept:name" in log_df.columns:
        case_col = "case:concept:name"
    else:
        raise KeyError("No case column found. Expected 'case_id' or 'case:concept:name'.")

    if "activity" in log_df.columns:
        activity_col = "activity"
    elif "concept:name" in log_df.columns:
        activity_col = "concept:name"
    else:
        raise KeyError("No activity column found. Expected 'activity' or 'concept:name'.")

    if "timestamp" in log_df.columns:
        timestamp_col = "timestamp"
    elif "time:timestamp" in log_df.columns:
        timestamp_col = "time:timestamp"
    else:
        raise KeyError("No timestamp column found. Expected 'timestamp' or 'time:timestamp'.")

    if "resource" in log_df.columns:
        resource_col = "resource"
    elif "org:resource" in log_df.columns:
        resource_col = "org:resource"
    else:
        raise KeyError("No resource column found. Expected 'resource' or 'org:resource'.")

    return case_col, activity_col, timestamp_col, resource_col


def normalize_available_time(available_time, duration_unit="hours"):
    """
    Convert available_time into a dict:
        {resource_name: available_time_in_selected_unit}

    accepted input:
    1. dict:
        {
            "Resource A": 40,
            "Resource B": 35
        }

    2. list of dicts:
        [
            {"resource": "Resource A", "available_time": 40},
            {"resource": "Resource B", "available_time": 35}
        ]
    """

    if available_time is None:
        return {}

    if isinstance(available_time, dict):
        return {
            normalize_resource_name(resource): float(value)
            for resource, value in available_time.items()
        }

    if isinstance(available_time, list):
        result = {}

        for item in available_time:
            resource = normalize_resource_name(item["resource"])
            value = float(item["available_time"])
            result[resource] = value

        return result

    raise TypeError(
        "available_time must be a dict or a list of dicts."
    )


def evaluate_resource_utilization(
    log_df,
    available_time=None,
    requirements=None
):
    """
    Evaluate resource workload, busy time, available time, and utilization rate.

    Parameters
    ----------
    log_df : pandas.DataFrame
        Event log. Supports:
        - case_id, activity, timestamp, resource
        - case:concept:name, concept:name, time:timestamp, org:resource

    available_time : dict or list of dicts
        Resource available time in the selected duration unit.

        Example dict:
        {
            "Billing Clerk": 40,
            "Booking Manager": 40,
            "Operations Agent": 40
        }

        Example list:
        [
            {"resource": "Billing Clerk", "available_time": 40},
            {"resource": "Booking Manager", "available_time": 40}
        ]

    requirements : dict, optional
        Example:
        {
            "duration_unit": "hours",
            "overload_threshold": 0.85,
            "idle_threshold": 0.2,
            "max_resource_missing_ratio": 0.05
        }

    Returns
    -------
    dict
        {
            "topic": "Resource utilization rate",
            "suitable": bool,
            "score": int,
            "resource_table": DataFrame,
            "activity_resource_table": DataFrame,
            "event_level_table": DataFrame,
            "attributes": dict,
            "issues": list,
            "warnings": list
        }
    """

    if requirements is None:
        requirements = {}

    issues = []
    warnings = []
    attributes = {}
    score = 100

    duration_unit = requirements.get("duration_unit", "hours")
    overload_threshold = requirements.get("overload_threshold", 0.85)
    idle_threshold = requirements.get("idle_threshold", 0.2)
    max_resource_missing_ratio = requirements.get("max_resource_missing_ratio", 0.05)

    df = log_df.copy()

    # --------------------------------------------------
    # Step 1: Detect columns
    # --------------------------------------------------
    try:
        case_col, activity_col, timestamp_col, resource_col = detect_resource_log_columns(df)
    except KeyError as e:
        return {
            "topic": "Resource utilization rate",
            "suitable": False,
            "score": 0,
            "resource_table": None,
            "activity_resource_table": None,
            "event_level_table": None,
            "attributes": {},
            "issues": [str(e)],
            "warnings": []
        }

    # --------------------------------------------------
    # Step 2: Normalize timestamp and resource
    # --------------------------------------------------
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
    df[resource_col] = df[resource_col].apply(normalize_resource_name)

    resource_missing_ratio = (df[resource_col] == "Unknown").mean()
    attributes["resource_missing_ratio"] = float(round(resource_missing_ratio, 4))

    if resource_missing_ratio > max_resource_missing_ratio:
        issues.append(
            f"Resource missing ratio is {resource_missing_ratio:.2%}, "
            f"which exceeds the allowed threshold of {max_resource_missing_ratio:.2%}."
        )
        score -= 30
    elif resource_missing_ratio > 0:
        warnings.append(
            f"Resource missing ratio is {resource_missing_ratio:.2%}. "
            "Missing resources are represented as 'Unknown'."
        )
        score -= 10

    df = df.dropna(subset=[case_col, activity_col, timestamp_col]).copy()

    if df.empty:
        return {
            "topic": "Resource utilization rate",
            "suitable": False,
            "score": 0,
            "resource_table": None,
            "activity_resource_table": None,
            "event_level_table": None,
            "attributes": attributes,
            "issues": ["No usable events remain after removing rows with missing case, activity, or timestamp."],
            "warnings": warnings
        }

    # --------------------------------------------------
    # Step 3: Sort events and calculate event-level busy time
    # --------------------------------------------------
    df = df.sort_values(
        by=[case_col, timestamp_col],
        kind="stable"
    ).reset_index(drop=True)

    df["next_timestamp"] = df.groupby(case_col)[timestamp_col].shift(-1)

    df["busy_time_seconds"] = (
        df["next_timestamp"] - df[timestamp_col]
    ).dt.total_seconds()

    # Negative durations are invalid
    negative_duration_count = int((df["busy_time_seconds"] < 0).sum())

    if negative_duration_count > 0:
        warnings.append(
            f"{negative_duration_count} events have negative busy time and are set to NaN."
        )
        df.loc[df["busy_time_seconds"] < 0, "busy_time_seconds"] = np.nan
        score -= 10

    # Last event in each case has no next timestamp
    df["busy_time_seconds"] = df["busy_time_seconds"].fillna(0)

    if duration_unit == "seconds":
        df[f"busy_time_{duration_unit}"] = df["busy_time_seconds"]
    elif duration_unit == "minutes":
        df[f"busy_time_{duration_unit}"] = df["busy_time_seconds"] / 60
    elif duration_unit == "hours":
        df[f"busy_time_{duration_unit}"] = df["busy_time_seconds"] / 3600
    elif duration_unit == "days":
        df[f"busy_time_{duration_unit}"] = df["busy_time_seconds"] / (3600 * 24)
    else:
        raise ValueError("duration_unit must be one of: seconds, minutes, hours, days")

    busy_col = f"busy_time_{duration_unit}"

    # --------------------------------------------------
    # Step 4: Resource event counts and busy time
    # --------------------------------------------------
    total_events = len(df)

    resource_table = (
        df.groupby(resource_col)
        .agg(
            event_count=(activity_col, "count"),
            total_busy_time=(busy_col, "sum"),
            mean_busy_time_per_event=(busy_col, "mean"),
            median_busy_time_per_event=(busy_col, "median"),
            min_busy_time_per_event=(busy_col, "min"),
            max_busy_time_per_event=(busy_col, "max"),
            distinct_activities=(activity_col, "nunique"),
            distinct_cases=(case_col, "nunique")
        )
        .reset_index()
        .rename(columns={resource_col: "resource"})
    )

    resource_table["workload_share"] = (
        resource_table["event_count"] / total_events
    )

    # --------------------------------------------------
    # Step 5: Add available time and utilization rate
    # --------------------------------------------------
    available_time_dict = normalize_available_time(
        available_time,
        duration_unit=duration_unit
    )

    resource_table["available_time"] = resource_table["resource"].map(
        available_time_dict
    )

    resource_table["utilization_rate"] = (
        resource_table["total_busy_time"] / resource_table["available_time"]
    )

    # If available time is missing, utilization cannot be calculated
    resource_table.loc[
        resource_table["available_time"].isna(),
        "utilization_rate"
    ] = np.nan

    resource_table["idle_time"] = (
        resource_table["available_time"] - resource_table["total_busy_time"]
    )

    resource_table["idle_rate"] = (
        1 - resource_table["utilization_rate"]
    )

    # --------------------------------------------------
    # Step 6: Classify resource status
    # --------------------------------------------------
    def classify_resource(row):
        if pd.isna(row["available_time"]):
            return "unknown_available_time"
        if row["utilization_rate"] > overload_threshold:
            return "overloaded"
        if row["utilization_rate"] < idle_threshold:
            return "idle"
        return "normal"

    resource_table["status"] = resource_table.apply(classify_resource, axis=1)

    # Clean numeric output
    numeric_cols = [
        "total_busy_time",
        "mean_busy_time_per_event",
        "median_busy_time_per_event",
        "min_busy_time_per_event",
        "max_busy_time_per_event",
        "workload_share",
        "available_time",
        "utilization_rate",
        "idle_time",
        "idle_rate"
    ]

    for col in numeric_cols:
        if col in resource_table.columns:
            resource_table[col] = resource_table[col].round(4)

    resource_table = resource_table.sort_values(
        "total_busy_time",
        ascending=False
    ).reset_index(drop=True)

    # --------------------------------------------------
    # Step 7: Activity-resource workload table
    # --------------------------------------------------
    activity_resource_table = (
        df.groupby([activity_col, resource_col])
        .agg(
            event_count=(case_col, "count"),
            total_busy_time=(busy_col, "sum"),
            mean_busy_time=(busy_col, "mean")
        )
        .reset_index()
        .rename(columns={
            activity_col: "activity",
            resource_col: "resource"
        })
        .sort_values(["activity", "event_count"], ascending=[True, False])
        .reset_index(drop=True)
    )

    activity_resource_table["total_busy_time"] = activity_resource_table["total_busy_time"].round(4)
    activity_resource_table["mean_busy_time"] = activity_resource_table["mean_busy_time"].round(4)

    # --------------------------------------------------
    # Step 8: Summary attributes
    # --------------------------------------------------
    attributes["num_resources"] = int(resource_table["resource"].nunique())
    attributes["total_events"] = int(total_events)
    attributes["total_busy_time"] = float(round(resource_table["total_busy_time"].sum(), 4))
    attributes["duration_unit"] = duration_unit
    attributes["overloaded_resources"] = resource_table.loc[
        resource_table["status"] == "overloaded",
        "resource"
    ].tolist()
    attributes["idle_resources"] = resource_table.loc[
        resource_table["status"] == "idle",
        "resource"
    ].tolist()
    attributes["resources_missing_available_time"] = resource_table.loc[
        resource_table["status"] == "unknown_available_time",
        "resource"
    ].tolist()

    if attributes["resources_missing_available_time"]:
        warnings.append(
            "Available time is missing for some resources. "
            "Utilization rate cannot be calculated for them."
        )
        score -= 10

    if attributes["num_resources"] < 2:
        warnings.append(
            "The log contains fewer than two resources. "
            "Resource utilization comparison is limited."
        )
        score -= 15

    score = max(score, 0)

    return {
        "topic": "Resource utilization rate",
        #"suitable": score >= 70 and len(issues) == 0,
        #"score": score,
        "resource_table": resource_table,
        "activity_resource_table": activity_resource_table,
        "event_level_table": df,
        "attributes": attributes,
        #"issues": issues,
        #"warnings": warnings
    }