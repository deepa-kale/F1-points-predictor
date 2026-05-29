from pathlib import Path

import pandas as pd


DATA_DIR = Path("data")
OUTPUT_FILE = DATA_DIR / "f1_data.csv"

OUTPUT_COLUMNS = [
    "season",
    "race_name",
    "driver_name",
    "constructor",
    "points",
    "qualifying_position",
    "session_type",
    "source_file",
]

COLUMN_ALIASES = {
    "race_name": ["race", "race name", "grand prix", "grand_prix", "gp", "event"],
    "driver_name": ["driver", "driver name", "driver_name", "name", "full name"],
    "constructor": ["constructor", "constructor name", "team", "team name", "car"],
    "points": ["points", "pts", "point"],
    "qualifying_position": [
        "qualifying position",
        "qualifying_position",
        "qualifying pos",
        "grid",
        "grid position",
        "position",
        "pos",
    ],
}


def normalize_column_name(column_name):
    """Normalize a column name so small spelling differences are easier to match."""
    return (
        str(column_name)
        .strip()
        .lower()
        .replace("_", " ")
        .replace("-", " ")
    )


def find_matching_column(df, possible_names):
    """Find the real CSV column for one of our standard output columns."""
    normalized_columns = {
        normalize_column_name(column): column
        for column in df.columns
    }

    for name in possible_names:
        normalized_name = normalize_column_name(name)
        if normalized_name in normalized_columns:
            return normalized_columns[normalized_name]

    return None


def find_season_from_path(file_path):
    """Find a four-digit season folder such as 2021 or 2022 in the file path."""
    for part in file_path.parts:
        if part.isdigit() and len(part) == 4:
            return int(part)

    return None


def find_session_type(file_path):
    """Return race or qualifying if the CSV lives in the right kind of folder."""
    folder_names = [part.lower() for part in file_path.parts]

    if "race results" in folder_names:
        return "race"

    if "qualifying results" in folder_names:
        return "qualifying"

    return None


def find_result_csv_files():
    """Recursively find CSVs under data/ inside Race Results or Qualifying Results."""
    if not DATA_DIR.exists():
        print(f"Data folder not found: {DATA_DIR}")
        return []

    result_files = []
    for file_path in DATA_DIR.rglob("*.csv"):
        if find_session_type(file_path) is not None:
            result_files.append(file_path)

    return sorted(result_files)


def clean_text(series):
    """Strip whitespace from text columns and convert blank strings to missing values."""
    return series.astype("string").str.strip().replace("", pd.NA)


def read_result_file(file_path):
    """Read one race or qualifying CSV and return a cleaned DataFrame."""
    season = find_season_from_path(file_path)
    session_type = find_session_type(file_path)

    if season is None or session_type is None:
        print(f"Skipped: {file_path}")
        return None

    try:
        raw_df = pd.read_csv(file_path)
    except Exception as error:
        print(f"Skipped: {file_path} ({error})")
        return None

    if raw_df.empty:
        print(f"Skipped: {file_path} (empty file)")
        return None

    clean_df = pd.DataFrame()
    clean_df["season"] = season
    clean_df["session_type"] = session_type
    clean_df["source_file"] = str(file_path)

    matched_columns = {}
    for output_column, possible_names in COLUMN_ALIASES.items():
        source_column = find_matching_column(raw_df, possible_names)
        matched_columns[output_column] = source_column

        if source_column is None:
            clean_df[output_column] = pd.NA
        else:
            clean_df[output_column] = raw_df[source_column]

    if matched_columns["driver_name"] is None:
        print(f"Skipped: {file_path} (no driver column found)")
        return None

    if matched_columns["race_name"] is None:
        clean_df["race_name"] = file_path.stem

    for column in ["race_name", "driver_name", "constructor"]:
        clean_df[column] = clean_text(clean_df[column])

    clean_df["points"] = pd.to_numeric(clean_df["points"], errors="coerce")
    clean_df["qualifying_position"] = pd.to_numeric(
        clean_df["qualifying_position"],
        errors="coerce",
    )

    clean_df = clean_df.dropna(subset=["driver_name"])
    if clean_df.empty:
        print(f"Skipped: {file_path} (no usable driver rows)")
        return None

    print(f"Loaded: {file_path}")
    return clean_df[OUTPUT_COLUMNS]


def main():
    all_frames = []
    loaded_files = 0

    for file_path in find_result_csv_files():
        result_df = read_result_file(file_path)
        if result_df is not None:
            all_frames.append(result_df)
            loaded_files += 1

    if not all_frames:
        print("No matching race or qualifying CSV files were loaded.")
        return

    final_df = pd.concat(all_frames, ignore_index=True)
    final_df = final_df.drop_duplicates()
    final_df = final_df.sort_values(
        ["season", "race_name", "session_type", "driver_name"],
        na_position="last",
    )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(OUTPUT_FILE, index=False)

    print()
    print(f"Files loaded: {loaded_files}")
    print(f"Saved output to: {OUTPUT_FILE}")
    print(f"Rows saved: {len(final_df)}")


if __name__ == "__main__":
    main()
