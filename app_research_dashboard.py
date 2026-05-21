from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Research Dashboard",
    page_icon=":bar_chart:",
    layout="wide",
)

st.title("Research Dashboard")

HIGH_SIGNAL_ROLE_FAMILIES = {
    "AI evaluation role",
    "Data annotation role",
    "Data collection role",
    "Translation / localization role",
}

SOURCE_QUALITY = {
    "greenhouse": "High",
    "lever": "High",
    "workday": "High",
    "icims": "High",
    "teamtailor": "High",
    "direct api": "High",
    "direct parser": "High",
    "rss": "Medium",
    "json": "Medium",
    "serpapi": "Low",
    "search": "Low",
}

RELEVANCE_SCORE = {
    "High": 3,
    "Medium": 2,
    "Low": 1,
    "Not relevant": 0,
    "": 0,
}

CONFIDENCE_SCORE = {
    "High": 3,
    "Medium": 2,
    "Low": 1,
    "": 1,
}

QUALITY_SCORE = {
    "High": 3,
    "Medium": 2,
    "Low": 1,
    "Unknown": 1,
}

FRIENDLY_COLUMN_NAMES = {
    "recommendation": "Recommendation",
    "recommendation_score": "Score",
    "rows": "Job postings found",
    "competitors": "Competitors found",
    "specific_posts": "Confirmed job posts",
    "high_relevance": "Strong matches",
    "target_roles": "Priority role types",
    "low_or_not_relevant": "Low-fit posts",
    "signal_pct": "Strong match rate",
    "noise_pct": "Low-fit rate",
    "best_locales": "Top locales",
    "best_role_families": "Top role types",
    "source_quality": "Source quality",
    "job_board": "Job board",
    "competitor": "Competitor",
    "llm_locale": "Locale",
    "llm_role_family": "Role type",
    "llm_language": "Language",
    "next_action": "Suggested next step",
    "decision_score": "Priority score",
    "llm_relevance": "Match quality",
    "llm_confidence": "Classification confidence",
    "welodata_suitability": "Welodata fit",
    "recommended_use": "Recommended use",
    "llm_strategic_insight": "Why this matters",
    "llm_hiring_signal": "Hiring signal",
    "llm_target_talent_pool": "Likely talent pool",
    "llm_country": "Country",
    "llm_pay_text": "Pay details",
    "notes": "Notes",
    "source_file": "Source file",
    "evidence_url": "Evidence link",
    "job_title": "Job title",
}

ENRICHMENT_COLUMNS = [
    "llm_is_specific_job_post",
    "llm_role_family",
    "llm_seniority",
    "llm_employment_type",
    "llm_location_mode",
    "llm_country",
    "llm_locale",
    "llm_language",
    "llm_pay_text",
    "llm_pay_min",
    "llm_pay_max",
    "llm_pay_currency",
    "llm_pay_interval",
    "llm_relevance",
    "llm_confidence",
    "llm_strategic_insight",
    "llm_target_talent_pool",
    "llm_hiring_signal",
]


def read_tabular_file(source: str | Path | object) -> pd.DataFrame:
    name = str(getattr(source, "name", source)).lower()
    if hasattr(source, "seek"):
        source.seek(0)

    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(source, dtype=str).fillna("")

    try:
        frame = pd.read_csv(source, dtype=str).fillna("")
        if len(frame.columns) != 1 or "|" not in str(frame.columns[0]):
            return frame
    except pd.errors.ParserError:
        pass

    if hasattr(source, "seek"):
        source.seek(0)
    return pd.read_csv(source, sep="|", engine="python", dtype=str).fillna("")


def canonicalize_probe_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy().fillna("")
    aliases = {
        "job_board": "source",
        "competitor": "company",
        "job_title": "title",
        "employment_type": "job_type",
        "evidence_url": "url",
        "pay_min": "salary_min",
        "pay_max": "salary_max",
        "pay_currency": "currency",
    }
    for canonical, alias in aliases.items():
        if canonical not in result.columns and alias in result.columns:
            result[canonical] = result[alias]

    if "notes" not in result.columns:
        note_columns = [
            column
            for column in ["department", "location", "country", "locale", "language_requirements"]
            if column in result.columns
        ]
        result["notes"] = result[note_columns].agg(" | ".join, axis=1) if note_columns else ""

    for column in ["job_board", "competitor", "job_title", "evidence_url"]:
        if column not in result:
            result[column] = ""
    return result


def normalized_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def normalized_url(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    parsed = urlparse(text)
    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower().removeprefix("www.")
    path = parsed.path.rstrip("/")
    return f"{scheme}://{netloc}{path}" if scheme and netloc else text.rstrip("/")


def dedupe_probe_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    if frame.empty:
        return frame.copy(), 0
    result = frame.copy()
    for column in ["job_board", "competitor", "job_title", "llm_locale", "locale", "location", "evidence_url"]:
        if column not in result:
            result[column] = ""
    fallback_locale = result["llm_locale"].where(result["llm_locale"].astype(str).str.strip().ne(""), result["locale"])
    result["_dedupe_url"] = result["evidence_url"].map(normalized_url)
    result["_dedupe_fallback"] = (
        result["job_board"].map(normalized_text)
        + "|"
        + result["competitor"].map(normalized_text)
        + "|"
        + result["job_title"].map(normalized_text)
        + "|"
        + fallback_locale.map(normalized_text)
        + "|"
        + result["location"].map(normalized_text)
    )
    result["_dedupe_key"] = result["_dedupe_url"].where(result["_dedupe_url"].ne(""), result["_dedupe_fallback"])
    before = len(result)
    result = result.drop_duplicates(subset=["_dedupe_key"], keep="first")
    helper_columns = [column for column in result.columns if column.startswith("_dedupe_")]
    return result.drop(columns=helper_columns), before - len(result)


def infer_source_quality(row: pd.Series) -> str:
    haystack = " ".join(
        str(row.get(column, "")) for column in ["source_method", "source", "job_board"]
    ).lower()
    for token, quality in SOURCE_QUALITY.items():
        if token in haystack:
            return quality
    domain = urlparse(str(row.get("evidence_url", "") or row.get("url", ""))).netloc.lower()
    if any(domain_part in domain for domain_part in ["greenhouse.io", "lever.co", "workdayjobs.com"]):
        return "High"
    return "Unknown"


def enrich_display_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    fallback_map = {
        "llm_role_family": "role_family",
        "llm_employment_type": "employment_type",
        "llm_location_mode": "location_mode",
        "llm_country": "country",
        "llm_locale": "locale",
        "llm_language": "language",
        "llm_pay_text": "pay_text",
        "llm_pay_min": "pay_min",
        "llm_pay_max": "pay_max",
        "llm_pay_currency": "pay_currency",
        "llm_pay_interval": "pay_interval",
        "llm_relevance": "Medium",
        "llm_confidence": "confidence",
        "llm_strategic_insight": "strategic_insight",
        "llm_target_talent_pool": "target_talent_pool",
        "llm_hiring_signal": "hiring_signal",
    }
    for column in ENRICHMENT_COLUMNS:
        if column not in result:
            result[column] = ""
    for llm_column, fallback in fallback_map.items():
        empty_mask = result[llm_column].astype(str).eq("")
        if fallback in result:
            result.loc[empty_mask, llm_column] = result.loc[empty_mask, fallback].astype(str)
        elif fallback in {"Medium"}:
            result.loc[empty_mask, llm_column] = fallback

    result["is_specific_job_post"] = result["llm_is_specific_job_post"].astype(str).str.lower().eq("true")
    result["is_high_relevance"] = result["llm_relevance"].eq("High")
    result["is_low_relevance"] = result["llm_relevance"].isin(["Low", "Not relevant"])
    result["is_target_role"] = result["llm_role_family"].isin(HIGH_SIGNAL_ROLE_FAMILIES)
    result["source_quality"] = result.apply(infer_source_quality, axis=1)

    result["welodata_suitability"] = result.apply(
        lambda row: "High"
        if row["is_high_relevance"] and row["is_specific_job_post"]
        else "Medium"
        if row["llm_relevance"] in ["High", "Medium"] or row["is_specific_job_post"]
        else "Low",
        axis=1,
    )
    result["recommended_use"] = result["welodata_suitability"].map(
        {
            "High": "Test Posting",
            "Medium": "Monitor / Investigate",
            "Low": "Ignore",
        }
    )
    result["reason_to_test"] = result["llm_strategic_insight"]
    result["decision_score"] = (
        result["llm_relevance"].map(RELEVANCE_SCORE).fillna(0) * 10
        + result["llm_confidence"].map(CONFIDENCE_SCORE).fillna(1) * 3
        + result["source_quality"].map(QUALITY_SCORE).fillna(1) * 2
        + result["is_specific_job_post"].astype(int) * 12
        + result["is_target_role"].astype(int) * 6
    )
    return result


def build_board_recommendations(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    grouped = (
        frame.groupby("job_board")
        .agg(
            rows=("job_board", "size"),
            competitors=("competitor", pd.Series.nunique),
            specific_posts=("is_specific_job_post", "sum"),
            high_relevance=("is_high_relevance", "sum"),
            target_roles=("is_target_role", "sum"),
            low_or_not_relevant=("is_low_relevance", "sum"),
            avg_row_score=("decision_score", "mean"),
        )
        .reset_index()
    )
    grouped["signal_pct"] = ((grouped["high_relevance"] / grouped["rows"]).fillna(0) * 100).round(0)
    grouped["noise_pct"] = ((grouped["low_or_not_relevant"] / grouped["rows"]).fillna(0) * 100).round(0)
    grouped["best_locales"] = grouped["job_board"].map(
        frame[frame["llm_locale"].astype(str).str.strip().ne("")]
        .groupby("job_board")["llm_locale"]
        .apply(lambda s: ", ".join(s.value_counts().head(4).index.astype(str)))
    ).fillna("")
    grouped["best_role_families"] = grouped["job_board"].map(
        frame[frame["llm_role_family"].astype(str).str.strip().ne("")]
        .groupby("job_board")["llm_role_family"]
        .apply(lambda s: ", ".join(s.value_counts().head(3).index.astype(str)))
    ).fillna("")
    grouped["source_quality"] = grouped["job_board"].map(
        frame.groupby("job_board")["source_quality"].apply(lambda s: ", ".join(s.value_counts().head(2).index.astype(str)))
    ).fillna("Unknown")
    grouped["recommendation_score"] = (
        grouped["high_relevance"] * 12
        + grouped["specific_posts"] * 8
        + grouped["target_roles"] * 6
        + grouped["competitors"] * 5
        + grouped["avg_row_score"]
        - grouped["low_or_not_relevant"] * 3
    ).round(1)
    grouped["recommendation"] = grouped.apply(
        lambda row: "Test posting"
        if row["high_relevance"] >= 3 and row["competitors"] >= 2
        else "Investigate"
        if row["specific_posts"] >= 3 or row["target_roles"] >= 2
        else "Deprioritize"
        if row["low_or_not_relevant"] / row["rows"] >= 0.6 and row["rows"] >= 5
        else "Monitor",
        axis=1,
    )
    return grouped.sort_values(
        ["recommendation_score", "high_relevance", "specific_posts", "competitors"],
        ascending=False,
    )


def build_action_queue(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    result = frame.copy()
    result["next_action"] = result.apply(
        lambda row: "Use as posting evidence"
        if row["is_high_relevance"] and row["is_specific_job_post"] and row["is_target_role"]
        else "Review for board fit"
        if row["is_high_relevance"] and row["is_specific_job_post"]
        else "Verify search result"
        if row["source_quality"] == "Low" and not row["is_specific_job_post"]
        else "Ignore for now"
        if row["is_low_relevance"]
        else "Manual review",
        axis=1,
    )
    return result.sort_values(["decision_score", "llm_relevance"], ascending=False)


def count_matrix(frame: pd.DataFrame, index: str, columns: str, values: str | None = None) -> pd.DataFrame:
    if frame.empty or index not in frame or columns not in frame:
        return pd.DataFrame()
    base = frame.copy()
    base = base[base[index].astype(str).str.strip().ne("") & base[columns].astype(str).str.strip().ne("")]
    if base.empty:
        return pd.DataFrame()
    if values:
        return pd.crosstab(base[index], base[columns], values=base[values], aggfunc="sum").fillna(0).astype(int)
    return pd.crosstab(base[index], base[columns]).fillna(0).astype(int)


def friendly_table(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.rename(columns=FRIENDLY_COLUMN_NAMES)


def filter_options(frame: pd.DataFrame, column: str) -> list[str]:
    if column not in frame:
        return []
    values = frame[column].astype(str).str.strip()
    values = values[values.ne("") & values.str.lower().ne("nan")]
    return sorted(values.unique().tolist())


def filter_by_text_values(frame: pd.DataFrame, column: str, selected: list[str]) -> pd.DataFrame:
    if not selected or column not in frame:
        return frame
    return frame[frame[column].astype(str).str.strip().isin(selected)]


def top_text(values: pd.Series, limit: int = 3) -> str:
    cleaned = values.astype(str).str.strip()
    cleaned = cleaned[cleaned.ne("") & cleaned.str.lower().ne("nan")]
    if cleaned.empty:
        return "Not available"
    return ", ".join(cleaned.value_counts().head(limit).index.tolist())


def postings_by_group(frame: pd.DataFrame, group_column: str, label_column: str) -> pd.DataFrame:
    if frame.empty or group_column not in frame:
        return pd.DataFrame(columns=[label_column, "Job postings found"])
    grouped = (
        frame.groupby(group_column)
        .size()
        .rename("Job postings found")
        .reset_index()
        .rename(columns={group_column: label_column})
        .sort_values("Job postings found", ascending=False)
    )
    return grouped


def signal_by_group(frame: pd.DataFrame, group_column: str, label_column: str) -> pd.DataFrame:
    if frame.empty or group_column not in frame:
        return pd.DataFrame()
    grouped = (
        frame.groupby(group_column)
        .agg(
            **{
                "Confirmed job posts": ("is_specific_job_post", "sum"),
                "Strong matches": ("is_high_relevance", "sum"),
                "Priority role types": ("is_target_role", "sum"),
                "Low-fit posts": ("is_low_relevance", "sum"),
            }
        )
        .reset_index()
        .rename(columns={group_column: label_column})
    )
    grouped["Useful signals"] = (
        grouped["Confirmed job posts"] + grouped["Strong matches"] + grouped["Priority role types"]
    )
    return grouped.sort_values("Useful signals", ascending=False).drop(columns=["Useful signals"])


def dataframe_download(frame: pd.DataFrame, label: str, file_name: str) -> None:
    st.download_button(
        label=label,
        data=frame.to_csv(index=False).encode("utf-8"),
        file_name=file_name,
        mime="text/csv",
        disabled=frame.empty,
    )


st.caption(
    "Upload one or more enriched research CSV or Excel files to explore board coverage, "
    "competitor signals, and action recommendations."
)

raw_df: pd.DataFrame | None = None
source_label = ""
removed_duplicates = 0

uploaded_files = st.file_uploader(
    "Upload enriched research files",
    type=["csv", "xlsx", "xls"],
    accept_multiple_files=True,
)
upload_signature = tuple((uploaded_file.name, uploaded_file.size) for uploaded_file in uploaded_files or [])
if uploaded_files and upload_signature != st.session_state.get("dashboard_upload_signature"):
    frames = []
    for uploaded_file in uploaded_files:
        frame = canonicalize_probe_frame(read_tabular_file(uploaded_file))
        frame["source_file"] = uploaded_file.name
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    raw_df, removed_duplicates = dedupe_probe_frame(combined)
    source_label = "Uploaded files" if len(uploaded_files) > 1 else uploaded_files[0].name
    st.session_state["dashboard_raw_df"] = raw_df
    st.session_state["dashboard_removed_duplicates"] = removed_duplicates
    st.session_state["dashboard_source_label"] = source_label
    st.session_state["dashboard_upload_signature"] = upload_signature
elif "dashboard_raw_df" in st.session_state:
    raw_df = st.session_state["dashboard_raw_df"]
    removed_duplicates = int(st.session_state.get("dashboard_removed_duplicates", 0) or 0)
    source_label = str(st.session_state.get("dashboard_source_label", "Uploaded files"))

if raw_df is None or raw_df.empty:
    st.info("Upload a CSV or Excel export to load the dashboard.")
    st.stop()

base_dashboard_df = enrich_display_frame(canonicalize_probe_frame(raw_df))
dashboard_df = base_dashboard_df.copy()

st.subheader(source_label)
col_loaded, col_clean, col_sources = st.columns(3)
col_loaded.metric("Loaded rows", len(base_dashboard_df) + removed_duplicates)
col_clean.metric("Clean rows", len(base_dashboard_df))
col_sources.metric("Duplicates removed", removed_duplicates)

with st.sidebar:
    st.header("Filters")
    if st.button("Reset filters"):
        for key in [
            "filter_job_board",
            "filter_competitor",
            "filter_role_family",
            "filter_locale",
            "filter_relevance",
            "filter_specific_only",
        ]:
            st.session_state.pop(key, None)
        st.rerun()

    board_filter = st.multiselect(
        "Job board",
        filter_options(base_dashboard_df, "job_board"),
        key="filter_job_board",
    )
    competitor_filter = st.multiselect(
        "Competitor",
        filter_options(base_dashboard_df, "competitor"),
        key="filter_competitor",
    )
    role_filter = st.multiselect(
        "Role family",
        filter_options(base_dashboard_df, "llm_role_family"),
        key="filter_role_family",
    )
    locale_filter = st.multiselect(
        "Locale",
        filter_options(base_dashboard_df, "llm_locale"),
        key="filter_locale",
    )
    relevance_filter = st.multiselect(
        "Relevance",
        ["High", "Medium", "Low", "Not relevant"],
        key="filter_relevance",
    )
    specific_only = st.checkbox("Specific job posts only", value=False, key="filter_specific_only")

dashboard_df = filter_by_text_values(dashboard_df, "job_board", board_filter)
dashboard_df = filter_by_text_values(dashboard_df, "competitor", competitor_filter)
dashboard_df = filter_by_text_values(dashboard_df, "llm_role_family", role_filter)
dashboard_df = filter_by_text_values(dashboard_df, "llm_locale", locale_filter)
dashboard_df = filter_by_text_values(dashboard_df, "llm_relevance", relevance_filter)
if specific_only:
    dashboard_df = dashboard_df[dashboard_df["is_specific_job_post"]]

specific = dashboard_df[dashboard_df["is_specific_job_post"]]
high_relevance = dashboard_df[dashboard_df["is_high_relevance"]]
low_relevance = dashboard_df[dashboard_df["is_low_relevance"]]
target_roles = dashboard_df[dashboard_df["is_target_role"]]
recommendations = build_board_recommendations(dashboard_df)
action_queue = build_action_queue(dashboard_df)

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Job postings found", len(dashboard_df))
col2.metric("Confirmed job posts", len(specific))
col3.metric("Strong matches", len(high_relevance))
col4.metric("Priority role types", len(target_roles))
col5.metric("Low-fit posts", len(low_relevance))

st.caption(f"Showing {len(dashboard_df):,} of {len(base_dashboard_df):,} clean postings after filters.")

with st.expander("How to read these terms"):
    st.markdown(
        """
        - **Job postings found**: every deduplicated posting or search result loaded from the uploaded file.
        - **Confirmed job posts**: rows classified as real job openings, not general pages, events, or weak search results.
        - **Strong matches**: postings that look highly relevant for data, AI evaluation, localization, or similar talent research.
        - **Priority role types**: roles in the main families we care about: AI evaluation, data annotation, data collection, and translation/localization.
        - **Locale**: the market, country, language, or region signal attached to the posting when available.
        """
    )

if dashboard_df.empty:
    st.info("No rows match the selected filters.")
    st.stop()

st.subheader("Recommended Boards")
st.caption(
    "Use this table to quickly see which job boards are worth testing first. "
    "Higher scores mean the board has more competitor activity and more relevant postings."
)
recommendation_cols = [
    "recommendation",
    "job_board",
    "recommendation_score",
    "rows",
    "competitors",
    "specific_posts",
    "high_relevance",
    "target_roles",
    "signal_pct",
    "noise_pct",
    "best_locales",
    "best_role_families",
    "source_quality",
]
recommendations_display = friendly_table(recommendations[recommendation_cols])
st.dataframe(
    recommendations_display,
    hide_index=True,
    width="stretch",
    column_config={
        "Score": st.column_config.NumberColumn("Score", help="Composite score based on competitor activity, confirmed posts, relevance, and source quality.", format="%.1f"),
        "Strong match rate": st.column_config.ProgressColumn("Strong match rate", help="Share of posts on this board that are strong matches.", min_value=0, max_value=100, format="%.0f%%"),
        "Low-fit rate": st.column_config.ProgressColumn("Low-fit rate", help="Share of posts that look noisy or not relevant.", min_value=0, max_value=100, format="%.0f%%"),
    },
)

tab_strategy, tab_coverage, tab_matrix, tab_actions = st.tabs(
    ["Strategic Intelligence", "Coverage", "Matrices", "Action Queue"]
)

with tab_strategy:
    st.subheader("Best Boards to Explore First")
    st.caption(
        "Boards in this section have the strongest evidence that competitors are posting roles similar to the ones we care about."
    )
    suitability_summary = (
        dashboard_df[dashboard_df["welodata_suitability"] == "High"]
        .groupby("job_board")
        .agg(
            strong_board_signals=("welodata_suitability", "size"),
            competitors=("competitor", lambda s: ", ".join(s.unique())),
            top_role_types=("llm_role_family", top_text),
            top_locales=("llm_locale", top_text),
            example_insight=("llm_strategic_insight", lambda s: s.iloc[0] if not s.empty else ""),
        )
        .sort_values("strong_board_signals", ascending=False)
    )
    st.dataframe(
        suitability_summary.rename(
            columns={
                "strong_board_signals": "Strong board signals",
                "competitors": "Competitors seen",
                "top_role_types": "Top role types",
                "top_locales": "Top locales",
                "example_insight": "Example insight",
            }
        ),
        width="stretch",
    )

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Markets by Job Board")
        st.caption("Shows which boards have strong-match postings for each locale or market.")
        locale_matrix = count_matrix(dashboard_df, "job_board", "llm_locale", "is_high_relevance")
        if not locale_matrix.empty:
            st.dataframe(friendly_table(locale_matrix.reset_index()).set_index("Job board"), width="stretch")
        else:
            st.info("No locale data is available in the current selection.")
    with col_b:
        st.subheader("Competitor Market Summary")
        st.caption("Use this to see which role types and locales each competitor appears to be hiring for.")
        comp_strategy = dashboard_df.groupby("competitor").agg(
            job_postings_found=("competitor", "size"),
            primary_role_types=("llm_role_family", top_text),
            top_locales=("llm_locale", top_text),
            hiring_signals=("llm_hiring_signal", top_text),
        )
        st.dataframe(
            comp_strategy.rename(
                columns={
                    "job_postings_found": "Job postings found",
                    "primary_role_types": "Primary role types",
                    "top_locales": "Top locales",
                    "hiring_signals": "Hiring signals",
                }
            ),
            width="stretch",
        )

with tab_coverage:
    st.caption(
        "Coverage answers the main question: where are competitors posting jobs, and which boards or markets show useful signals?"
    )
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Where Competitors Post: Job Boards")
        st.caption("Total job postings found on each board after deduplication.")
        board_volume = postings_by_group(dashboard_df, "job_board", "Job board").head(25)
        if not board_volume.empty:
            st.bar_chart(board_volume.set_index("Job board"))
        else:
            st.info("No job board data is available in the current selection.")
    with col_b:
        st.subheader("Which Competitors Are Most Active")
        st.caption("Total job postings found for each competitor in the uploaded file.")
        competitor_volume = postings_by_group(dashboard_df, "competitor", "Competitor")
        if not competitor_volume.empty:
            st.bar_chart(competitor_volume.set_index("Competitor"))
        else:
            st.info("No competitor data is available in the current selection.")

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Useful Signals by Board")
        st.caption("Breaks out confirmed job posts, strong matches, priority role types, and low-fit posts.")
        board_signal = signal_by_group(dashboard_df, "job_board", "Job board").head(25)
        if not board_signal.empty:
            st.bar_chart(board_signal.set_index("Job board"))
        else:
            st.info("No board signal data is available in the current selection.")
    with col_b:
        st.subheader("Useful Signals by Competitor")
        st.caption("Shows which competitors have the most relevant or confirmed postings.")
        competitor_signal = signal_by_group(dashboard_df, "competitor", "Competitor")
        if not competitor_signal.empty:
            st.bar_chart(competitor_signal.set_index("Competitor"))
        else:
            st.info("No competitor signal data is available in the current selection.")

    st.subheader("Market and Locale Focus")
    st.caption("These views show where competitors are hiring by locale or market when that data is available.")
    col_a, col_b = st.columns(2)
    with col_a:
        locale_volume = postings_by_group(
            dashboard_df[dashboard_df["llm_locale"].astype(str).str.strip().ne("")],
            "llm_locale",
            "Locale",
        ).head(20)
        if not locale_volume.empty:
            st.bar_chart(locale_volume.set_index("Locale"))
        else:
            st.info("No locale data is available in the current selection.")
    with col_b:
        locale_competitor = count_matrix(dashboard_df, "llm_locale", "competitor")
        if not locale_competitor.empty:
            st.dataframe(friendly_table(locale_competitor.reset_index()).set_index("Locale"), width="stretch")
        else:
            st.info("No locale data is available in the current selection.")

with tab_matrix:
    st.subheader("Detailed Cross-Tabs")
    st.caption(
        "These tables are for deeper analysis. Each number is a count of strong-match postings in that board, competitor, locale, or role type."
    )

    st.subheader("Job Board x Competitor")
    board_competitor = count_matrix(dashboard_df, "job_board", "competitor", "is_high_relevance")
    st.dataframe(board_competitor, width="stretch")
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Job Board x Locale")
        board_locale = count_matrix(dashboard_df, "job_board", "llm_locale", "is_high_relevance")
        st.dataframe(board_locale, width="stretch")
    with col_b:
        st.subheader("Job Board x Role Type")
        board_role = count_matrix(dashboard_df, "job_board", "llm_role_family", "is_high_relevance")
        st.dataframe(board_role, width="stretch")

with tab_actions:
    st.caption(
        "This queue turns the research data into suggested follow-up steps for recruiting or market research."
    )
    action_cols = [
        "next_action",
        "decision_score",
        "llm_relevance",
        "llm_confidence",
        "welodata_suitability",
        "recommended_use",
        "llm_strategic_insight",
        "llm_hiring_signal",
        "llm_target_talent_pool",
        "job_board",
        "competitor",
        "job_title",
        "llm_role_family",
        "llm_locale",
        "llm_language",
        "llm_pay_text",
        "evidence_url",
    ]
    visible_cols = [column for column in action_cols if column in action_queue.columns]
    action_display = friendly_table(action_queue[visible_cols])
    st.dataframe(
        action_display,
        hide_index=True,
        width="stretch",
        column_config={
            "Priority score": st.column_config.NumberColumn(
                "Priority score",
                help="Higher means the posting is more relevant, better sourced, or tied to a priority role type.",
                format="%.0f",
            ),
            "Evidence link": st.column_config.LinkColumn("Evidence link"),
        },
    )

st.subheader("Filtered Rows")
st.caption("Clean row-level detail after upload, deduplication, and the active filters.")
all_cols = [
    "welodata_suitability",
    "recommended_use",
    "llm_relevance",
    "source_quality",
    "job_board",
    "competitor",
    "job_title",
    "llm_role_family",
    "llm_strategic_insight",
    "llm_hiring_signal",
    "llm_target_talent_pool",
    "llm_country",
    "llm_locale",
    "llm_language",
    "llm_pay_text",
    "notes",
    "source_file",
    "evidence_url",
]
visible_cols = [column for column in all_cols if column in dashboard_df.columns]
filtered_display = friendly_table(dashboard_df[visible_cols])
st.dataframe(
    filtered_display,
    hide_index=True,
    width="stretch",
    column_config={"Evidence link": st.column_config.LinkColumn("Evidence link")},
)

dataframe_download(dashboard_df, "Download filtered enriched CSV", "research_dashboard_enriched.csv")
dataframe_download(recommendations, "Download board recommendations", "research_dashboard_board_recommendations.csv")
