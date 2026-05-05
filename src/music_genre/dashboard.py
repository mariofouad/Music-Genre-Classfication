from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Streamlit Cloud installs deps from requirements.txt only; it does not run `poetry install`,
# so the `music_genre` package under `src/` is not on sys.path. Put `src/` first when needed.
_src_dir = Path(__file__).resolve().parents[1]
if _src_dir.name == "src" and (_src_dir / "music_genre").is_dir():
    src_str = str(_src_dir)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)

import joblib  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from music_genre.config import load_config  # noqa: E402
from music_genre.features import add_engineered_features, feature_columns  # noqa: E402

# Spotify-style numeric audio descriptors (display only; the model uses all `feature_columns`).
_AUDIO_FEATURE_COLUMNS: tuple[str, ...] = (
    "danceability",
    "energy",
    "key",
    "loudness",
    "mode",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "time_signature",
)


@st.cache_data(show_spinner=False)
def _load_test_split_cached(split_path_str: str) -> pd.DataFrame:
    path = Path(split_path_str)
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if not df.empty and "energy_acoustic_ratio" not in df.columns:
        df = add_engineered_features(df)
    return df


def load_dashboard_data(config_path: str | Path) -> tuple[dict, pd.DataFrame, dict]:
    config = load_config(config_path)
    data_path = Path(config["paths"]["processed_data"])
    if not data_path.exists():
        data_path = Path(config["paths"]["integrated_data"])
    data = pd.read_csv(data_path) if data_path.exists() else pd.DataFrame()
    if not data.empty and "energy_acoustic_ratio" not in data:
        data = add_engineered_features(data)
    metrics_path = Path(config["paths"]["metrics_path"])
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    return config, data, metrics


def _search_test_tracks(test_df: pd.DataFrame, query: str, *, limit: int = 40) -> pd.DataFrame:
    q = (query or "").strip().lower()
    if not q or test_df.empty:
        return pd.DataFrame()
    names = test_df["track_name"].fillna("").astype(str).str.lower()
    artists = test_df["artists"].fillna("").astype(str).str.lower()
    mask = names.str.contains(q, regex=False) | artists.str.contains(q, regex=False)
    return test_df.loc[mask].head(limit).reset_index(drop=True)


def _display_cell(v: object) -> str:
    """Arrow-friendly string for mixed-type feature columns in st.dataframe."""
    if pd.isna(v):
        return ""
    return str(v)


def _audio_feature_display_frame(row: pd.Series) -> pd.DataFrame:
    present = [c for c in _AUDIO_FEATURE_COLUMNS if c in row.index and pd.notna(row.get(c))]
    if not present:
        return pd.DataFrame()
    return pd.DataFrame(
        {"feature": present, "value": [_display_cell(row[c]) for c in present]}
    ).set_index("feature")


def _render_test_set_predictor(config: dict) -> None:
    st.divider()
    st.subheader("Test set: song search → audio features → predict → compare to label")
    st.caption(
        "Uses **`data/processed/splits/test.csv`** (held out from `fit`). "
        "Prediction uses **feature columns only**—the label is never passed to the model. "
        "Below, **audio features** are the core descriptors; other model inputs are in an expander."
    )

    model_path = Path(config["paths"]["model_dir"]) / "best_model.joblib"
    if not model_path.exists():
        st.warning("Train first so `models/best_model.joblib` exists (e.g. `poetry run music-genre-train`).")
        return

    split_path = Path(config["paths"]["split_dir"]) / "test.csv"
    test_df = _load_test_split_cached(str(split_path.resolve()))
    if test_df.empty:
        st.warning(f"No test split at `{split_path}`. Run the data pipeline to build splits.")
        return

    target = config["data"]["target"]
    feats = feature_columns(config)
    missing = [c for c in feats if c not in test_df.columns]
    if missing:
        st.error(f"Test split missing columns required by the model: {missing}")
        return
    if target not in test_df.columns:
        st.error(f"Test split has no target column `{target}`.")
        return

    model = joblib.load(model_path)

    query = st.text_input(
        "Song or artist (substring search in test set)",
        placeholder="e.g. song title or artist name",
    )
    c1, c2 = st.columns(2)
    with c1:
        search_clicked = st.button("Search test tracks", disabled=not bool((query or "").strip()))
    with c2:
        random_clicked = st.button("Pick random test track")

    if random_clicked:
        st.session_state["test_matches"] = test_df.sample(n=1, random_state=None).reset_index(drop=True)
        st.session_state["test_query_label"] = "(random)"
    if search_clicked:
        matches = _search_test_tracks(test_df, query.strip())
        st.session_state["test_matches"] = matches
        st.session_state["test_query_label"] = query.strip()
        if matches.empty:
            st.warning("No rows matched. Try different text.")

    matches: pd.DataFrame | None = st.session_state.get("test_matches")
    if matches is None or matches.empty:
        return

    if st.session_state.get("test_query_label"):
        st.caption(f"Current result set: `{st.session_state['test_query_label']}`")

    labels = [
        f"{row.get('track_name', '')} — {row.get('artists', '')}" for _, row in matches.iterrows()
    ]
    idx = st.selectbox("Choose a track", range(len(labels)), format_func=lambda i: labels[i])

    if st.button("Predict genre from features", type="primary"):
        row = matches.iloc[idx]
        x_row = row[feats].to_frame().T.reset_index(drop=True)

        st.markdown("#### Audio features (label not fed to the model)")
        audio_tbl = _audio_feature_display_frame(row)
        if audio_tbl.empty:
            st.info("No standard audio columns to display for this row.")
        else:
            st.dataframe(audio_tbl, width="stretch")

        with st.expander("Other features the model uses"):
            other = [c for c in feats if c not in _AUDIO_FEATURE_COLUMNS]
            if other:
                st.dataframe(
                    pd.DataFrame(
                        {"feature": other, "value": [_display_cell(row.get(c)) for c in other]}
                    ).set_index("feature"),
                    width="stretch",
                )

        try:
            pred = model.predict(x_row)[0]
        except Exception as exc:  # noqa: BLE001
            st.error(f"Prediction failed: {exc}")
            return

        actual = row[target]
        same = str(pred) == str(actual)

        st.markdown("#### Prediction vs hold-out label")
        pc1, pc2, pc3 = st.columns(3)
        pc1.metric("Predicted", str(pred))
        pc2.metric(f"Actual {target}", str(actual))
        pc3.metric("Match", "Yes" if same else "No")

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(x_row)[0]
            classes = list(model.classes_)
            top = sorted(range(len(proba)), key=lambda i: float(proba[i]), reverse=True)[
                : min(12, len(classes))
            ]
            st.dataframe(
                pd.DataFrame(
                    {target: [classes[i] for i in top], "probability": [float(proba[i]) for i in top]}
                ),
                width="stretch",
                hide_index=True,
            )


def render_dashboard(config_path: str | Path = "configs/project.yaml") -> None:
    import plotly.express as px

    config, data, metrics = load_dashboard_data(config_path)
    st.set_page_config(page_title="Music Genre Classification", layout="wide")
    st.title("Music Genre Classification")

    target = config["data"]["target"]
    target_label = "Genre families" if target == "genre_family" else "Genre labels"

    if data.empty:
        st.warning(
            "No processed dataset found. Run `poetry run music-genre-data` and `poetry run music-genre-train` "
            "for charts. You can still use the **test-set predictor** below if `test.csv` and `best_model.joblib` exist."
        )
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("Tracks", f"{len(data):,}")
        col2.metric(target_label, data[target].nunique())
        col3.metric("Sources", data["source_dataset"].nunique())

        source_filter = st.multiselect(
            "Source",
            sorted(data["source_dataset"].dropna().unique()),
            default=sorted(data["source_dataset"].dropna().unique()),
        )
        filtered = data[data["source_dataset"].isin(source_filter)]

        left, right = st.columns(2)
        with left:
            counts = filtered[target].value_counts().reset_index()
            counts.columns = [target, "tracks"]
            st.plotly_chart(
                px.bar(counts, x=target, y="tracks", title="Class Distribution"),
                width="stretch",
            )
        with right:
            st.plotly_chart(
                px.scatter(
                    filtered,
                    x="energy",
                    y="acousticness",
                    color=target,
                    hover_data=["track_name", "artists"],
                    title="Energy vs Acousticness",
                ),
                width="stretch",
            )

        st.plotly_chart(
            px.box(
                filtered,
                x=target,
                y="danceability",
                color=target,
                title=f"Danceability by {target_label}",
            ),
            width="stretch",
        )

        if metrics:
            st.subheader("Best Model Evaluation")
            metric_rows = []
            for split_name, payload in metrics.items():
                for metric_name, value in payload.get("metrics", {}).items():
                    metric_rows.append({"split": split_name, "metric": metric_name, "value": value})
            st.dataframe(pd.DataFrame(metric_rows), width="stretch")

    _render_test_set_predictor(config)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Streamlit dashboard.")
    parser.add_argument("--config", default="configs/project.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    render_dashboard(args.config)


if __name__ == "__main__":
    main()
