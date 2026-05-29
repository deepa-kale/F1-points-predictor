# F1 Points Predictor

A portfolio-style Streamlit app that predicts a Formula 1 driver's expected race points from race, driver, constructor, and qualifying position. It uses historical F1 race and qualifying data, a baseline scikit-learn model, and a lightweight interactive dashboard for local prediction.

## Overview

This project was built as a fast MVP to demonstrate:
- data preparation from messy historical F1 files
- baseline machine learning model training
- model serialization with `joblib`
- an interactive Streamlit interface for predictions

The app lets a user:
- choose a race
- choose a driver
- choose a constructor/team
- enter a qualifying position
- predict expected F1 points
- view recent race rows for the selected driver

## Features

- Streamlit dashboard with a simple two-column layout
- Prediction form for race, driver, team, and qualifying position
- Baseline regression model for expected points
- Recent-results table for selected driver
- Local model loading from `artifacts/model.pkl`

## Dataset

The project uses historical Formula 1 race and qualifying data downloaded from Kaggle and processed into a single flat CSV for modeling.

Prepared dataset file:
- `data/f1_data.csv`

Main columns used:
- `race_name`
- `driver_name`
- `constructor`
- `qualifying_position`
- `points`

Other available columns:
- `season`
- `session_type`
- `source_file`

## Model

The baseline model predicts `points` using these features:
- `race_name`
- `driver_name`
- `constructor`
- `qualifying_position`

The training workflow:
1. Load the cleaned CSV
2. Keep rows with non-null target values
3. Clean and filter usable feature columns
4. Train a baseline scikit-learn regression model
5. Save the trained model to `artifacts/model.pkl`

The saved model is included in this repository so the app can run without retraining first.[web:159][web:162][web:165]

## Results

Current baseline performance:

- Training rows before cleaning: 37,311
- Training rows after cleaning: 21,673
- MAE: 1.648
- RMSE: 2.535

These results come from the current baseline version and are meant as a starting point, not a final production-grade F1 prediction system.

## Project structure

```text
F1-points-predictor/
├── app.py
├── train_model.py
├── prepare_data.py
├── requirements.txt
├── README.md
├── data/
│   └── f1_data.csv
├── artifacts/
│   └── model.pkl
└── .streamlit/
    └── config.toml
```

GitHub-flavored Markdown supports fenced code blocks like the project tree above, which is a clean way to show repo structure in a README.[web:153][web:155][web:154]

## How to run locally

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/F1-points-predictor.git
cd F1-points-predictor
```

### 2. Create and activate a virtual environment

Mac/Linux:

```bash
python -m venv venv
source venv/bin/activate
```

Windows:

```bash
venv\Scripts\activate
```

Creating a virtual environment with `venv` is the standard Python workflow and is also recommended in Streamlit’s setup guidance.[web:167]

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run app.py
```

`streamlit run app.py` is the standard command for launching a local Streamlit app.[web:158][web:164][web:167]

### 5. Open the local URL

After running the command, Streamlit will show a local URL in the terminal, usually something like:

```text
http://localhost:8501
```

## Training the model again

If you want to retrain the model locally:

```bash
python train_model.py
```

This will retrain the baseline model and overwrite the saved model artifact used by the app.

## Preparing data again

If you want to rebuild the combined dataset:

```bash
python prepare_data.py
```

This script scans the historical race and qualifying folders and rebuilds `data/f1_data.csv`.

## Limitations

- This is a baseline MVP, not a production-grade forecasting system
- The current model uses a relatively small feature set
- Historical F1 files required aggressive cleaning because file formats and column names were inconsistent
- The app is designed for demonstration and portfolio use, not betting or official sports analytics decisions
- The prediction quality depends heavily on the quality of the prepared dataset

## Next steps

Planned improvements:
- cleaner filtering between race and qualifying data
- better feature engineering
- improved driver/team/race dependency logic in the UI
- model comparison across multiple algorithms
- public deployment
- better visuals and charts for race context

## Why this project matters

This project shows an end-to-end workflow:
- raw sports data preparation
- machine learning model training
- model persistence
- interactive product-style demo
