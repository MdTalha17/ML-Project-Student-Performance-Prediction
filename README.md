## Student Performance Prediction — End‑to‑End ML Project

### Overview
Predict students' math scores using demographic and study-related features. This repo includes data ingestion, preprocessing, model training (with model selection), and a Flask web app for inference.

### Tech Stack
- **Language**: Python 3.11
- **ML**: scikit‑learn, CatBoost, XGBoost
- **Web**: Flask, Jinja2

### Project Structure (key files)
- `src/components/data_ingestion.py`: Reads raw CSV, splits train/test, writes to `artifacts/` and can drive the full training flow when run directly.
- `src/components/data_transformation.py`: Builds preprocessing pipeline and persists `artifacts/preprocessor.pkl`.
- `src/components/model_trainer.py`: Trains multiple models with hyperparams, selects best by R², saves `artifacts/model.pkl`.
- `src/pipeline/predict_pipeline.py`: Loads preprocessor and model to produce predictions.
- `app.py` / `application.py`: Flask app with forms UI.
- `artifacts/`: Outputs — `data.csv`, `train.csv`, `test.csv`, `preprocessor.pkl`, `model.pkl`.
- `notebook/`: EDA and experimentation.

---

### Quickstart (Windows PowerShell)
1) Clone and enter the project directory
```powershell
git clone <your-repo-url>
cd ML-Project-Student-Performance-Prediction
```

2) Create and activate a virtual environment (Python 3.11)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3) Install dependencies
```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

---

### Train the Model
This will:
- Read `notebook/data/stud.csv`
- Split into train/test and write to `artifacts/`
- Fit and persist `artifacts/preprocessor.pkl` and the best `artifacts/model.pkl`

Run:
```powershell
python .\src\components\data_ingestion.py
```

Outputs land in the `artifacts/` folder. Check console logs for R² on the test split.

---

### Run the Web App (Inference)
Start Flask locally:
```powershell
python .\app.py
```

Then open `http://127.0.0.1:5000/` in your browser.

Routes:
- `/` — Landing page
- `/predictdata` — Form to input features and see predicted math score

Required form fields (as used by the app):
- `gender` (e.g., male/female)
- `ethnicity` (mapped to `race_ethnicity`)
- `parental_level_of_education`
- `lunch` (e.g., standard/free/reduced)
- `test_preparation_course` (e.g., none/completed)
- `reading_score` (integer)
- `writing_score` (integer)

Note: The app loads `artifacts/preprocessor.pkl` and `artifacts/model.pkl`. Ensure you have trained first or provide these files.

---

### Programmatic Inference
```python
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

sample = CustomData(
    gender="female",
    race_ethnicity="group B",
    parental_level_of_education="bachelor's degree",
    lunch="standard",
    test_preparation_course="none",
    reading_score=72,
    writing_score=74,
)

df = sample.get_data_as_dataframe()
pred = PredictPipeline().predict(df)
print(float(pred[0]))
```

---

### Data
- Source CSV expected at `notebook/data/stud.csv`.
- Artifacts saved to `artifacts/` after training.

### Troubleshooting
- If Flask cannot find models, verify `artifacts/model.pkl` and `artifacts/preprocessor.pkl` exist (train first).
- Windows paths: This repo uses backslashes in some paths; run commands from the project root as shown.
- For port conflicts, set `FLASK_RUN_PORT` or stop other processes using port 5000.

### License
Add your preferred license here.