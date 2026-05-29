from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


DATA_FILE = Path("data/f1_data.csv")
MODEL_FILE = Path("artifacts/model.pkl")

FEATURE_COLUMNS = [
    "race_name",
    "driver_name",
    "constructor",
    "qualifying_position",
]
TARGET_COLUMN = "points"


@st.cache_data
def load_data():
    """Load the prepared F1 data."""
    return pd.read_csv(DATA_FILE)


@st.cache_resource
def load_model():
    """Load the trained scikit-learn model."""
    return joblib.load(MODEL_FILE)


def sorted_options(df, column_name):
    """Return clean dropdown options for a column."""
    return sorted(df[column_name].dropna().astype(str).unique())


def get_result_rows(df):
    """Keep rows that have real race points, not qualifying-only rows."""
    df = df.copy()
    df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
    return df.dropna(subset=[TARGET_COLUMN])


def show_recent_driver_rows(df, driver_name):
    """Show a small table of recent rows for the selected driver."""
    driver_rows = df[df["driver_name"].astype(str) == driver_name].copy()

    if driver_rows.empty:
        st.info("No recent rows found for this driver.")
        return

    if "season" in driver_rows.columns:
        driver_rows = driver_rows.sort_values("season", ascending=False)

    columns_to_show = [
        column
        for column in [
            "season",
            "race_name",
            "driver_name",
            "constructor",
            "qualifying_position",
            "points",
            "session_type",
        ]
        if column in driver_rows.columns
    ]

    st.subheader(f"Recent rows for {driver_name}")
    st.dataframe(driver_rows[columns_to_show].head(10), use_container_width=True)


def main():
    st.set_page_config(page_title="F1 Points Predictor")

    st.title("F1 Points Predictor")
    st.write(
        "Choose a race, driver, team, and qualifying position to estimate "
        "how many points the driver might score."
    )

    if not DATA_FILE.exists():
        st.error(f"Could not find {DATA_FILE}. Run prepare_data.py first.")
        return

    if not MODEL_FILE.exists():
        st.error(f"Could not find {MODEL_FILE}. Run train_model.py first.")
        return

    df = load_data()
    model = load_model()

    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        st.error("Missing required columns: " + ", ".join(missing_columns))
        return

    results_df = get_result_rows(df)
    if results_df.empty:
        st.error("No rows with points found in data/f1_data.csv.")
        return

    race_name = st.selectbox("Race", sorted_options(results_df, "race_name"))
    driver_name = st.selectbox("Driver", sorted_options(results_df, "driver_name"))

    driver_rows = results_df[results_df["driver_name"].astype(str) == driver_name]
    constructor_options = sorted_options(driver_rows, "constructor")
    if not constructor_options:
        constructor_options = sorted_options(results_df, "constructor")

    constructor = st.selectbox("Constructor / Team", constructor_options)
    qualifying_position = st.number_input(
        "Qualifying position",
        min_value=1,
        max_value=30,
        value=10,
        step=1,
    )

    if st.button("Predict points"):
        input_data = pd.DataFrame(
            [
                {
                    "race_name": race_name,
                    "driver_name": driver_name,
                    "constructor": constructor,
                    "qualifying_position": qualifying_position,
                }
            ],
            columns=FEATURE_COLUMNS,
        )

        predicted_points = model.predict(input_data)[0]
        st.success(f"Predicted F1 points: {predicted_points:.2f}")
        st.metric("Predicted points", f"{predicted_points:.2f}")

    show_recent_driver_rows(results_df, driver_name)


if __name__ == "__main__":
    main()
