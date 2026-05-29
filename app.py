from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="F1 Points Predictor",
    page_icon="🏎️",
    layout="wide",
)

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
    return pd.read_csv(DATA_FILE)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)


def sorted_options(df, column_name):
    if column_name not in df.columns:
        return []
    return sorted(df[column_name].dropna().astype(str).unique())


def get_result_rows(df):
    clean_df = df.copy()
    clean_df[TARGET_COLUMN] = pd.to_numeric(clean_df[TARGET_COLUMN], errors="coerce")
    clean_df["qualifying_position"] = pd.to_numeric(
        clean_df["qualifying_position"], errors="coerce"
    )
    clean_df = clean_df.dropna(subset=[TARGET_COLUMN, "race_name", "driver_name", "constructor"])
    return clean_df


def show_recent_driver_rows(df, driver_name):
    driver_rows = df[df["driver_name"].astype(str) == driver_name].copy()

    if driver_rows.empty:
        st.info("No recent race rows found for this driver.")
        return

    if "season" in driver_rows.columns:
        driver_rows["season"] = pd.to_numeric(driver_rows["season"], errors="coerce")

    sort_columns = [col for col in ["season", "race_name"] if col in driver_rows.columns]
    if sort_columns:
        driver_rows = driver_rows.sort_values(sort_columns, ascending=False)

    columns_to_show = [
        col
        for col in [
            "season",
            "race_name",
            "driver_name",
            "constructor",
            "qualifying_position",
            "points",
        ]
        if col in driver_rows.columns
    ]

    st.subheader("Recent Race Results")
    st.dataframe(
        driver_rows[columns_to_show].head(8),
        use_container_width=True,
        hide_index=True,
    )


def prediction_label(predicted_points):
    if predicted_points >= 15:
        return "Podium-level outcome likely."
    if predicted_points >= 8:
        return "Strong points finish likely."
    if predicted_points >= 3:
        return "Moderate points potential."
    return "Low expected points outcome."


def main():
    st.title("🏎️ F1 Points Predictor")
    st.caption(
        "A baseline machine learning app that estimates a driver's expected race points "
        "from race, driver, constructor, and qualifying position."
    )

    if not DATA_FILE.exists():
        st.error("Could not find data/f1_data.csv. Run prepare_data.py first.")
        return

    if not MODEL_FILE.exists():
        st.error("Could not find artifacts/model.pkl. Run train_model.py first.")
        return

    df = load_data()
    model = load_model()

    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        st.error("Missing required columns: " + ", ".join(missing_columns))
        return

    results_df = get_result_rows(df)
    if results_df.empty:
        st.error("No race-result rows with points were found in data/f1_data.csv.")
        return

    top_col1, top_col2, top_col3 = st.columns(3)
    with top_col1:
        st.metric("Model Type", "Baseline Regressor")
    with top_col2:
        st.metric("Training Rows", "21,673")
    with top_col3:
        st.metric("RMSE", "2.535")

    st.divider()

    race_options = sorted_options(results_df, "race_name")
    driver_options = sorted_options(results_df, "driver_name")

    with st.form("prediction_form"):
        left_col, right_col = st.columns(2)

        with left_col:
            race_name = st.selectbox("Race", race_options)
            driver_name = st.selectbox("Driver", driver_options)

        driver_filtered_df = results_df[results_df["driver_name"].astype(str) == driver_name]
        constructor_options = sorted_options(driver_filtered_df, "constructor")
        if not constructor_options:
            constructor_options = sorted_options(results_df, "constructor")

        with right_col:
            constructor = st.selectbox("Constructor / Team", constructor_options)
            qualifying_position = st.number_input(
                "Qualifying Position",
                min_value=1,
                max_value=30,
                value=10,
                step=1,
            )

        submitted = st.form_submit_button("Predict Points")

    if submitted:
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

        predicted_points = float(model.predict(input_data)[0])

        st.divider()
        result_col1, result_col2 = st.columns([1, 1])

        with result_col1:
            st.metric("Predicted Points", f"{predicted_points:.1f}")

        with result_col2:
            label = prediction_label(predicted_points)
            if predicted_points >= 8:
                st.success(label)
            elif predicted_points >= 3:
                st.info(label)
            else:
                st.warning(label)

        st.caption(
            "This is a baseline ML estimate based on race, driver, constructor, and qualifying position."
        )

    st.divider()
    show_recent_driver_rows(results_df, driver_name)


if __name__ == "__main__":
    main()