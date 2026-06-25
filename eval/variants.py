import pandas as pd
from string import ascii_uppercase
import matplotlib.pyplot as plt


def extract_trace_variants(log_df):
    """
    Extract trace variants from an event log.

    Each variant is represented as a tuple of activities.
    Example:
        ("Receive Order", "Check Stock", "Ship Goods")

    Returns
    -------
    dict
        {
            variant_tuple: case_count
        }
    """

    variants = {}

    for case_id, group in log_df.groupby("case_id"):
        if "timestamp" in group.columns and group["timestamp"].notna().any():
            group = group.sort_values("timestamp", kind="stable")

        trace = tuple(group["activity"].tolist())

        variants[trace] = variants.get(trace, 0) + 1

    return variants

def remove_traces_with_missing_values(
    log_df,
    required_columns=None,
    treat_empty_string_as_missing=True
):
    """
    Remove complete traces/cases that contain missing values
    in selected required columns.

    Parameters
    ----------
    log_df : pandas.DataFrame
        Event log DataFrame.

    required_columns : list
        Columns to check for missing values.
        If None, use ["case_id", "activity", "timestamp", "resource"].

    treat_empty_string_as_missing : bool
        Whether "", " ", "null", "None", "nan", "NaT" should also be treated as missing.

    Returns
    -------
    cleaned_log_df : pandas.DataFrame
        Event log after removing cases with missing values.

    report : dict
        Information about removed traces.
    """



    if required_columns is None:
        required_columns = ["case_id", "activity", "timestamp", "resource"]

    # Only check columns that actually exist
    existing_columns = [
        col for col in required_columns
        if col in log_df.columns
    ]

    missing_required_columns = [
        col for col in required_columns
        if col not in log_df.columns
    ]

    df = log_df.copy()

    # Basic missing values: NaN, None, NaT
    missing_mask = df[existing_columns].isna()

    if treat_empty_string_as_missing:
        empty_like_mask = df[existing_columns].astype(str).apply(
            lambda col: col.str.strip().str.lower().isin(
                ["", "nan", "none", "null", "nat"]
            )
        )
        missing_mask = missing_mask | empty_like_mask

    # Rows/events that contain missing values in any checked column
    rows_with_missing = missing_mask.any(axis=1)

    # Case IDs of traces that contain at least one problematic event
    traces_to_remove = df.loc[rows_with_missing, "case_id"].dropna().unique()

    cleaned_log_df = df[~df["case_id"].isin(traces_to_remove)].copy()

    report = {
        "checked_columns": existing_columns,
        "missing_required_columns": missing_required_columns,
        "removed_trace_count": int(len(traces_to_remove)),
        "removed_traces": list(traces_to_remove),
        "original_trace_count": int(df["case_id"].nunique()),
        "remaining_trace_count": int(cleaned_log_df["case_id"].nunique()),
        "original_event_count": int(len(df)),
        "remaining_event_count": int(len(cleaned_log_df)),
    }

    return cleaned_log_df, report

def analyze_variant_distribution(log_df, coverage_threshold=0.8):
    """
    Analyze trace variant distribution and select variants that cover
    a given proportion of cases, e.g. 80%.

    Parameters
    ----------
    log_df : pandas.DataFrame
        Event log with normalized columns: case_id, activity, timestamp.

    coverage_threshold : float
        Coverage threshold, e.g. 0.8 means variants covering 80% of cases.

    Returns
    -------
    dict
        Variant distribution summary.
    """

    variants = extract_trace_variants(log_df)

    total_cases = sum(variants.values())

    sorted_variants = sorted(
        variants.items(),
        key=lambda x: x[1],
        reverse=True
    )

    variant_table = []
    cumulative_count = 0
    selected_variants = []

    for rank, (variant, count) in enumerate(sorted_variants, start=1):
        ratio = count / total_cases
        cumulative_count += count
        cumulative_ratio = cumulative_count / total_cases

        row = {
            "rank": rank,
            "variant": variant,
            "case_count": int(count),
            "case_ratio": round(ratio, 4),
            "cumulative_count": int(cumulative_count),
            "cumulative_ratio": round(cumulative_ratio, 4)
        }

        variant_table.append(row)

        if cumulative_ratio <= coverage_threshold:
            selected_variants.append(variant)
        elif not selected_variants or variant not in selected_variants:
            # include the variant that crosses the threshold
            selected_variants.append(variant)
            break

    selected_case_count = sum(
        variants[variant] for variant in selected_variants
    )

    return {
        "total_cases": int(total_cases),
        "num_variants": int(len(variants)),
        "coverage_threshold": coverage_threshold,
        "selected_variant_count": int(len(selected_variants)),
        "selected_case_count": int(selected_case_count),
        "selected_case_ratio": round(selected_case_count / total_cases, 4),
        "selected_variants": selected_variants,
        "variant_table": variant_table
    }

def filter_log_by_selected_variants(log_df, selected_variants):
    """
    Keep only cases whose trace variant is in selected_variants.
    """

    selected_variants = set(selected_variants)

    case_to_variant = {}

    for case_id, group in log_df.groupby("case_id"):
        if "timestamp" in group.columns and group["timestamp"].notna().any():
            group = group.sort_values("timestamp", kind="stable")

        trace = tuple(group["activity"].tolist())
        case_to_variant[case_id] = trace

    selected_case_ids = [
        case_id
        for case_id, variant in case_to_variant.items()
        if variant in selected_variants
    ]

    filtered_log_df = log_df[
        log_df["case_id"].isin(selected_case_ids)
    ].copy()

    return filtered_log_df, selected_case_ids

def generate_activity_codes(n):
    """
    Generate activity codes: A, B, ..., Z, AA, AB, ...
    """
    codes = []
    alphabet = ascii_uppercase

    for i in range(n):
        code = ""
        x = i

        while True:
            code = alphabet[x % 26] + code
            x = x // 26 - 1
            if x < 0:
                break

        codes.append(code)

    return codes

def map_trace_variants_to_codes(variants):
    """
    Map activity names in trace variants to short codes.

    Parameters
    ----------
    variants : dict
        Dictionary in the form:
        {
            ("Activity A", "Activity B"): count,
            ("Activity A", "Activity C"): count
        }

    Returns
    -------
    dict
        {
            "mapped_variants": list of dicts,
            "activity_mapping": dict,
            "variant_table": pandas.DataFrame,
            "mapping_table": Code - Activity (pandas.DataFrame) 
        }
    """

    # --------------------------------------------------
    # collect all unique activities in first-seen order
    # --------------------------------------------------
    unique_activities = []

    for trace in variants.keys():
        for activity in trace:
            if activity not in unique_activities:
                unique_activities.append(activity)

    # --------------------------------------------------
    # create mapping table
    # --------------------------------------------------
    codes = generate_activity_codes(len(unique_activities))

    activity_mapping = {
        activity: code
        for activity, code in zip(unique_activities, codes)
    }

    # --------------------------------------------------
    # replace activities in traces
    # --------------------------------------------------
    total_cases = sum(variants.values())

    mapped_variants = []

    for trace, count in variants.items():
        mapped_trace = tuple(activity_mapping[activity] for activity in trace)
        percentage = count / total_cases * 100 if total_cases > 0 else 0
        mapped_variants.append({
            "percentage": round(percentage, 2),
            "mapped_trace": mapped_trace,
            "mapped_trace_string": " ".join(mapped_trace),
            "original_trace": trace,
            "original_trace_string": " -> ".join(trace),
            "count": int(count)
        })

    # Sort by count descending
    mapped_variants = sorted(
        mapped_variants,
        key=lambda x: x["count"],
        reverse=True
    )

    cumulative_percentage = 0

    for row in mapped_variants:
        cumulative_percentage += row["percentage"]
        row["cumulative_percentage"] = round(cumulative_percentage, 2)

        
    # --------------------------------------------------
    # Step 4: create DataFrames
    # --------------------------------------------------
    variant_table = pd.DataFrame(mapped_variants)

    mapping_table = pd.DataFrame([
        {
            "code": code,
            "activity": activity
        }
        for activity, code in activity_mapping.items()
    ])

    

    return {
        "mapped_variants": mapped_variants,
        "activity_mapping": activity_mapping,
        "variant_table": variant_table,
        "mapping_table": mapping_table
    }

def plot_variant_pareto_chart(
    variant_table,
    top_n=20,
    label_col="mapped_trace_string",
    count_col="count",
    cumulative_col="cumulative_percentage",
    title="Pareto Chart of Trace Variants"
):
    """
    Plot a Pareto chart for trace variants.

    Parameters
    ----------
    variant_table : pandas.DataFrame
        Table containing variant counts and cumulative percentages.

    top_n : int
        Number of top variants to display.

    label_col : str
        Column used as x-axis labels.

    count_col : str
        Column containing case counts.

    cumulative_col : str
        Column containing cumulative percentage.

    title : str
        Chart title.
    """

    df = variant_table.copy()

    # Make sure variants are sorted by count
    df = df.sort_values(count_col, ascending=False).reset_index(drop=True)

    # Recalculate cumulative percentage to be safe
    total_count = df[count_col].sum()
    df["percentage"] = df[count_col] / total_count * 100
    df[cumulative_col] = df["percentage"].cumsum()

    # Keep only top N variants for readability
    plot_df = df.head(top_n)

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Bar chart: count
    ax1.bar(
        plot_df[label_col],
        plot_df[count_col]
    )
    ax1.set_xlabel("Trace variant")
    ax1.set_ylabel("Case count")
    ax1.tick_params(axis="x", rotation=45)

    # Line chart: cumulative percentage
    ax2 = ax1.twinx()
    ax2.plot(
        plot_df[label_col],
        plot_df[cumulative_col],
        marker="o"
    )
    ax2.set_ylabel("Cumulative percentage (%)")
    ax2.set_ylim(0, 105)

    # Optional 80% reference line
    ax2.axhline(80, linestyle="--")

    #plt.title(title)
    #plt.tight_layout()
    #plt.show()
    plt.close(fig)
    return df

def plot_variant_case_count_histogram(
    variant_table,
    top_n=20,
    count_col="count",
    label_col="mapped_trace_string",
    title="Histogram of Case Counts per Variant",
    bins=20
):
    """
    Plot a histogram showing the distribution of case counts across variants.

    Two sub-plots are produced:
    - Left:  histogram of raw case counts (all variants)
    - Right: bar chart of top-N variants by case count

    Parameters
    ----------
    variant_table : pandas.DataFrame
        Output from map_trace_variants_to_codes, contains count and label columns.

    top_n : int
        Number of top variants shown in the bar chart.

    count_col : str
        Column with case counts.

    label_col : str
        Column used as x-axis labels for the bar chart.

    title : str
        Figure title.

    bins : int
        Number of bins for the histogram.

    Returns
    -------
    pandas.DataFrame
        The (possibly trimmed) variant table used for plotting.
    """
    df = variant_table.copy().sort_values(count_col, ascending=False).reset_index(drop=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- left: histogram of all variant case counts ---
    axes[0].hist(df[count_col], bins=bins, edgecolor="white", linewidth=0.5)
    axes[0].set_xlabel("Case count per variant")
    axes[0].set_ylabel("Number of variants")
    axes[0].set_title("Distribution of case counts (all variants)")

    mean_val = df[count_col].mean()
    median_val = df[count_col].median()
    axes[0].axvline(mean_val, linestyle="--", label=f"Mean: {mean_val:.1f}")
    axes[0].axvline(median_val, linestyle=":", label=f"Median: {median_val:.1f}")
    axes[0].legend(fontsize=8)

    # --- right: bar chart of top-N variants ---
    plot_df = df.head(top_n)
    axes[1].bar(range(len(plot_df)), plot_df[count_col])
    axes[1].set_xticks(range(len(plot_df)))
    axes[1].set_xticklabels(plot_df[label_col], rotation=45, ha="right", fontsize=7)
    axes[1].set_xlabel("Trace variant")
    axes[1].set_ylabel("Case count")
    axes[1].set_title(f"Top {top_n} variants by case count")

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.close(fig)
    return df


def plot_variant_trace_length_boxplot(
    log_df,
    top_n=15,
    case_col="case_id",
    activity_col="activity",
    timestamp_col="timestamp",
    title="Trace Length Distribution by Variant"
):
    """
    Plot a box plot showing the distribution of trace lengths
    (number of events per case) grouped by trace variant.

    Only the top-N most frequent variants are shown to keep the chart readable.

    Parameters
    ----------
    log_df : pandas.DataFrame
        Event log with at least case_id, activity, and timestamp columns.

    top_n : int
        Number of most frequent variants to include.

    case_col : str
    activity_col : str
    timestamp_col : str

    title : str
        Chart title.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns: case_id, trace_length, variant_string.
    """
    df = log_df.copy()

    if timestamp_col in df.columns:
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
        df = df.sort_values([case_col, timestamp_col], kind="stable")

    # Build variant string per case
    case_variants = (
        df.groupby(case_col)[activity_col]
        .apply(lambda x: " -> ".join(x.tolist()))
        .reset_index()
        .rename(columns={activity_col: "variant_string"})
    )

    # Trace length per case
    trace_lengths = (
        df.groupby(case_col)
        .size()
        .reset_index(name="trace_length")
    )

    merged = trace_lengths.merge(case_variants, on=case_col)

    # Keep only top-N variants
    top_variants = (
        merged["variant_string"]
        .value_counts()
        .head(top_n)
        .index.tolist()
    )
    plot_df = merged[merged["variant_string"].isin(top_variants)].copy()

    # Shorten labels for readability (max 40 chars)
    plot_df["variant_label"] = plot_df["variant_string"].apply(
        lambda s: s[:37] + "..." if len(s) > 40 else s
    )

    # Sort variants by median trace length descending
    order = (
        plot_df.groupby("variant_label")["trace_length"]
        .median()
        .sort_values(ascending=False)
        .index.tolist()
    )

    fig, ax = plt.subplots(figsize=(12, max(5, len(order) * 0.55)))

    data_by_variant = [
        plot_df.loc[plot_df["variant_label"] == v, "trace_length"].values
        for v in order
    ]

    bp = ax.boxplot(
        data_by_variant,
        vert=False,
        patch_artist=True,
        medianprops={"linewidth": 2},
        flierprops={"marker": "o", "markersize": 3, "alpha": 0.5},
    )

    ax.set_yticks(range(1, len(order) + 1))
    ax.set_yticklabels(order, fontsize=8)
    ax.set_xlabel("Trace length (number of events)")
    ax.set_title(title, fontsize=13, fontweight="bold")

    plt.tight_layout()
    plt.close(fig)

    return merged[[case_col, "trace_length", "variant_string"]]


def create_variants_report(log_df):
    report = {}
    variants_list = extract_trace_variants(log_df)
    mapped_result = map_trace_variants_to_codes(variants_list)
    mapped_variants = mapped_result["mapped_variants"]
    activity_mapping = mapped_result["activity_mapping"]
    variant_table = mapped_result["variant_table"]
    num_variants = len(mapped_variants)
    report["variants number"] = num_variants
    report["mapping list"] = activity_mapping

    dict_list = []
    for row in mapped_variants:
        dict_list.append({
            "trace": row["mapped_trace_string"],
            "count": row["count"],
            "percentage": row["percentage"],
            "cumulative_percentage": row["cumulative_percentage"]
        })
    report["variants_list"] = dict_list

    # Pareto chart
    pareto_df = plot_variant_pareto_chart(
        variant_table,
        top_n=20,
        label_col="mapped_trace_string",
        title="Pareto Chart of Trace Variants"
    )
    report["pareto chart"] = pareto_df

    # Histogram: distribution of case counts across variants
    histogram_df = plot_variant_case_count_histogram(
        variant_table,
        top_n=20,
        count_col="count",
        label_col="mapped_trace_string",
        title="Histogram of Case Counts per Variant"
    )
    report["histogram df"] = histogram_df

    # Boxplot: trace length distribution by variant
    boxplot_df = plot_variant_trace_length_boxplot(
        log_df,
        top_n=15,
        title="Trace Length Distribution by Variant"
    )
    report["boxplot df"] = boxplot_df

    return report