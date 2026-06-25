import pandas as pd
import re


def normalize_activity_name(name):
    if name is None:
        return ""
    name = str(name).strip()
    name = re.sub(r"\s+", " ", name)
    return name


def detect_event_log_columns(log_df):
    """
    Detect whether the event log uses normalized columns or PM4Py/XES columns.
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

    return case_col, activity_col, timestamp_col


def evaluate_throughput_time(log_df, requirements=None):
    """
    Evaluate throughput time of cases, activity duration approximation,
    and duration statistics by trace variant.

    Parameters
    ----------
    log_df : pandas.DataFrame
        Event log DataFrame. Supports both:
        - case_id, activity, timestamp
        - case:concept:name, concept:name, time:timestamp

    requirements : dict, optional
        Optional thresholds, for example:
        {
            "max_allowed_timestamp_missing_ratio": 0.05,
            "duration_unit": "hours"
        }

    Returns
    -------
    dict
        {
            "topic": "Throughput time of cases",
            "suitable": bool,
            "score": int,
            "case_duration_table": DataFrame,
            "case_duration_describe": dict,
            "activity_duration_table": DataFrame,
            "activity_duration_describe": DataFrame,
            "variant_duration_table": DataFrame,
            "variant_duration_describe": DataFrame,
            "issues": list,
            "warnings": list,
            "attributes": dict
        }
    """

    if requirements is None:
        requirements = {}

    issues = []
    warnings = []
    attributes = {}
    score = 100

    df = log_df.copy()

    # --------------------------------------------------
    # Step 1: Detect columns
    # --------------------------------------------------
    try:
        case_col, activity_col, timestamp_col = detect_event_log_columns(df)
    except KeyError as e:
        return {
            "topic": "Throughput time of cases",
            "suitable": False,
            "score": 0,
            "issues": [str(e)],
            "warnings": [],
            "attributes": {},
            "case_duration_table": None,
            "case_duration_describe": None,
            "activity_duration_table": None,
            "activity_duration_describe": None,
            "variant_duration_table": None,
            "variant_duration_describe": None,
        }

    # --------------------------------------------------
    # Step 2: Normalize timestamp and activity names
    # --------------------------------------------------
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
    df[activity_col] = df[activity_col].apply(normalize_activity_name)

    timestamp_missing_ratio = df[timestamp_col].isna().mean()
    attributes["timestamp_missing_ratio"] = round(timestamp_missing_ratio, 4)

    max_allowed_missing = requirements.get(
        "max_allowed_timestamp_missing_ratio",
        0.05
    )

    if timestamp_missing_ratio > max_allowed_missing:
        issues.append(
            f"Timestamp missing/invalid ratio is {timestamp_missing_ratio:.2%}, "
            f"which exceeds the allowed threshold of {max_allowed_missing:.2%}."
        )
        score -= 40
    elif timestamp_missing_ratio > 0:
        warnings.append(
            f"Timestamp missing/invalid ratio is {timestamp_missing_ratio:.2%}. "
            "Rows with invalid timestamps are excluded from time-based calculations."
        )
        score -= 10

    df = df.dropna(subset=[case_col, activity_col, timestamp_col]).copy()

    if df.empty:
        return {
            "topic": "Throughput time of cases",
            "suitable": False,
            "score": 0,
            "issues": ["No usable events remain after removing rows with missing case, activity, or timestamp."],
            "warnings": warnings,
            "attributes": attributes,
            "case_duration_table": None,
            "case_duration_describe": None,
            "activity_duration_table": None,
            "activity_duration_describe": None,
            "variant_duration_table": None,
            "variant_duration_describe": None,
        }

    # --------------------------------------------------
    # Step 3: Sort events within each case
    # --------------------------------------------------
    df = df.sort_values(
        by=[case_col, timestamp_col],
        kind="stable"
    ).reset_index(drop=True)

    duration_unit = requirements.get("duration_unit", "hours")

    def seconds_to_unit(seconds):
        if duration_unit == "seconds":
            return seconds
        if duration_unit == "minutes":
            return seconds / 60
        if duration_unit == "hours":
            return seconds / 3600
        if duration_unit == "days":
            return seconds / (3600 * 24)
        raise ValueError("duration_unit must be one of: seconds, minutes, hours, days")

    # --------------------------------------------------
    # Step 4: Case throughput time
    # --------------------------------------------------
    case_duration_table = (
        df.groupby(case_col)[timestamp_col]
        .agg(start_time="min", end_time="max", event_count="count")
        .reset_index()
    )

    case_duration_table["duration_seconds"] = (
        case_duration_table["end_time"] - case_duration_table["start_time"]
    ).dt.total_seconds()

    case_duration_table[f"duration_{duration_unit}"] = seconds_to_unit(
        case_duration_table["duration_seconds"]
    )

    duration_col = f"duration_{duration_unit}"

    case_duration_describe = (
        case_duration_table[duration_col]
        .describe()
        .to_dict()
    )

    attributes["num_cases_with_duration"] = int(case_duration_table[case_col].nunique())
    attributes["case_duration_mean"] = round(case_duration_table[duration_col].mean(), 4)
    attributes["case_duration_median"] = round(case_duration_table[duration_col].median(), 4)
    attributes["case_duration_min"] = round(case_duration_table[duration_col].min(), 4)
    attributes["case_duration_max"] = round(case_duration_table[duration_col].max(), 4)

    # --------------------------------------------------
    # Step 5: Approximate activity duration
    # --------------------------------------------------
    # If the log only has one timestamp per event, we approximate the duration
    # of an activity as the time until the next event within the same case.
    df["next_timestamp"] = df.groupby(case_col)[timestamp_col].shift(-1)

    df["activity_duration_seconds"] = (
        df["next_timestamp"] - df[timestamp_col]
    ).dt.total_seconds()

    df[f"activity_duration_{duration_unit}"] = seconds_to_unit(
        df["activity_duration_seconds"]
    )

    activity_duration_col = f"activity_duration_{duration_unit}"

    # Last event in each case has no next timestamp, so duration is NaN.
    activity_duration_table = df[
        [
            case_col,
            activity_col,
            timestamp_col,
            "next_timestamp",
            "activity_duration_seconds",
            activity_duration_col
        ]
    ].copy()

    activity_duration_describe = (
        activity_duration_table
        .dropna(subset=[activity_duration_col])
        .groupby(activity_col)[activity_duration_col]
        .describe()
        .reset_index()
    )

    # --------------------------------------------------
    # Step 6: Variant extraction per case
    # --------------------------------------------------
    case_to_variant = (
        df.groupby(case_col)[activity_col]
        .apply(lambda x: tuple(x.tolist()))
        .to_dict()
    )

    case_duration_table["variant"] = case_duration_table[case_col].map(case_to_variant)
    case_duration_table["variant_string"] = case_duration_table["variant"].apply(
        lambda v: " -> ".join(v) if isinstance(v, tuple) else ""
    )

    # --------------------------------------------------
    # Step 7: Duration statistics by variant
    # --------------------------------------------------
    variant_duration_describe = (
        case_duration_table
        .groupby("variant_string")[duration_col]
        .describe()
        .reset_index()
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )

    variant_case_counts = (
        case_duration_table["variant_string"]
        .value_counts()
        .reset_index()
    )
    variant_case_counts.columns = ["variant_string", "case_count"]

    variant_duration_table = variant_duration_describe.merge(
        variant_case_counts,
        on="variant_string",
        how="left"
    )

    total_cases = case_duration_table[case_col].nunique()
    variant_duration_table["case_percentage"] = (
        variant_duration_table["case_count"] / total_cases * 100
    ).round(2)

    variant_duration_table = variant_duration_table.sort_values(
        "case_count",
        ascending=False
    ).reset_index(drop=True)

    # --------------------------------------------------
    # Step 8: Suitability checks
    # --------------------------------------------------
    if case_duration_table[duration_col].nunique() <= 1:
        warnings.append(
            "All cases have the same throughput time. "
            "This may reduce the usefulness of the log for teaching temporal analysis."
        )
        score -= 15

    if activity_duration_describe.empty:
        issues.append(
            "No activity durations could be calculated. "
            "This may happen if each case contains only one event."
        )
        score -= 30

    if variant_duration_table.empty:
        issues.append("No variant-level duration statistics could be calculated.")
        score -= 20

    score = max(score, 0)

    return {
        "topic": "Throughput time of cases",
        #"suitable": score >= 70 and len(issues) == 0,
        #"score": score,
        "attributes": attributes,
        "case_duration_table": case_duration_table,
        "case_duration_describe": case_duration_describe,
        "activity_duration_table": activity_duration_table,
        "activity_duration_describe": activity_duration_describe,
        "variant_duration_table": variant_duration_table,
        "variant_duration_describe": variant_duration_describe,
        #"issues": issues,
        #"warnings": warnings
    }