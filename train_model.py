from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_FILE = Path("data/f1_data.csv")
ARTIFACTS_DIR = Path("artifacts")
MODEL_FILE = ARTIFACTS_DIR / "model.pkl"

TARGET_COLUMN = "points"
FEATURE_COLUMNS = [
    "race_name",
    "driver_name",
    "constructor",
    "qualifying_position",
]
CATEGORICAL_FEATURES = ["race_name", "driver_name", "constructor"]
NUMERIC_FEATURES = ["qualifying_position"]


def load_data():
    """Load the prepared F1 data."""
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Could not find {DATA_FILE}. Run prepare_data.py first."
        )

    return pd.read_csv(DATA_FILE)


def check_required_columns(df):
    """Make sure the CSV has the columns this script needs."""
    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_columns = [
        column for column in required_columns if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns: " + ", ".join(missing_columns)
        )


def clean_data(df):
    """Keep usable training rows without being too strict."""
    rows_before = len(df)
    df = df.copy()

    df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
    df["qualifying_position"] = pd.to_numeric(
        df["qualifying_position"],
        errors="coerce",
    )

    # Keep only rows where we know the result we want to predict.
    df = df.dropna(subset=[TARGET_COLUMN])

    # These text fields are important identifiers, so rows missing them are not useful.
    df = df.dropna(subset=CATEGORICAL_FEATURES)

    for column in CATEGORICAL_FEATURES:
        df[column] = df[column].astype("string").str.strip()
        df = df[df[column] != ""]

    # Qualifying position is useful, but many race rows may not have it.
    # Fill missing values with the median instead of dropping those rows.
    median_qualifying_position = df["qualifying_position"].median()
    if pd.isna(median_qualifying_position):
        median_qualifying_position = 0

    df["qualifying_position"] = df["qualifying_position"].fillna(
        median_qualifying_position
    )

    rows_after = len(df)
    print(f"Rows before cleaning: {rows_before}")
    print(f"Rows after cleaning: {rows_after}")

    return df


def build_model():
    """Build a simple preprocessing and regression pipeline."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", Ridge(alpha=1.0)),
        ]
    )


def rmse(y_true, y_pred):
    """Calculate root mean squared error."""
    return mean_squared_error(y_true, y_pred) ** 0.5


def main():
    df = load_data()
    check_required_columns(df)
    df = clean_data(df)

    if df.empty:
        raise ValueError("No rows left after cleaning.")

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    model = build_model()

    if len(df) < 2:
        model.fit(X, y)
        print("Not enough rows for a train/test split, so the model used all rows.")
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
        )

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        print(f"MAE: {mean_absolute_error(y_test, predictions):.3f}")
        print(f"RMSE: {rmse(y_test, predictions):.3f}")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_FILE)

    print(f"Features used: {', '.join(FEATURE_COLUMNS)}")
    print(f"Saved model to: {MODEL_FILE}")


if __name__ == "__main__":
    main()
