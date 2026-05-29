# F1 Points Predictor 🏎️

A simple Streamlit app that predicts a Formula 1 driver's expected points outcome based on race, driver, constructor, and qualifying position. This project uses historical F1 race and qualifying data, a baseline scikit-learn model, and a lightweight interactive interface for quick predictions.

## Features

- Select a race, driver, and constructor
- Enter qualifying position
- Predict expected F1 points
- View recent rows for the selected driver
- Run locally with Streamlit

## Dataset

This project uses historical Formula 1 race and qualifying data downloaded from Kaggle and processed into a single CSV for modeling. The cleaned file used for training is:

- `data/f1_data.csv`

The prepared dataset includes these columns:

- `season`
- `race_name`
- `driver_name`
- `constructor`
- `points`
- `qualifying_position`
- `session_type`
- `source_file`

## Model

The baseline model was trained using scikit-learn to predict `points`.

Features used:
- `race_name`
- `driver_name`
- `constructor`
- `qualifying_position`

The training pipeline:
- Removed rows with missing target values
- Kept the most useful available features for a fast MVP
- Applied preprocessing for categorical and numeric data
- Trained a baseline regression model
- Saved the model to `artifacts/model.pkl`

## Results

Current baseline performance:

- MAE: 1.648
- RMSE: 2.535

Rows before cleaning: 37,311  
Rows after cleaning: 21,673

## How to run

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/F1-points-predictor.git
cd F1-points-predictor
```

### 2. Create and activate a virtual environment

On Mac/Linux:

```bash
python -m venv venv
source venv/bin/activate
```

On Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal.

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
└── artifacts/
    └── model.pkl
```

## Limitations

- This is a baseline MVP, not a final production model
- The current model uses a small feature set
- Some source data required aggressive cleaning because race and qualifying files were inconsistent
- The UI can still be improved by filtering dropdown options more intelligently

