# 📊 Student Performance Prediction System

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.3+-green.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A comprehensive **end-to-end machine learning system** that predicts student mathematics scores based on demographic, academic, and background features. This production-ready application demonstrates the complete ML lifecycle from data ingestion to web deployment.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Architecture](#project-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Details](#model-details)
- [API Documentation](#api-documentation)
- [Development](#development)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project implements a **regression-based machine learning system** that predicts student math scores (0-100) using supervised learning algorithms. The system includes:

- **Automated ML pipeline** with data preprocessing, feature engineering, and model selection
- **8 machine learning algorithms** evaluated and tuned automatically
- **Production-ready web application** with input validation and error handling
- **Model persistence** for consistent inference
- **Comprehensive logging** and exception handling

### Problem Statement

Predict student mathematics performance to help educators:
- Identify students who may need additional support
- Understand factors affecting academic performance
- Make data-driven decisions for curriculum planning

## ✨ Features

- **Multi-Model Training**: Evaluates 8 different algorithms (Random Forest, XGBoost, CatBoost, etc.)
- **Hyperparameter Tuning**: Automatic grid search with cross-validation
- **Automated Model Selection**: Chooses best performing model based on R² score
- **Feature Engineering**: 
  - Categorical encoding (One-Hot Encoding)
  - Numerical scaling (StandardScaler)
  - Missing value imputation
- **Web Interface**: User-friendly Flask application with Bootstrap UI
- **Input Validation**: Comprehensive client and server-side validation
- **Error Handling**: Custom exception handling with detailed logging
- **Model Persistence**: Saves trained models and preprocessors for production use
- **Reproducible Results**: Fixed random seeds and version-controlled dependencies

## 🛠 Tech Stack

### Core Technologies
- **Python 3.11** - Programming language
- **Flask 2.3+** - Web framework
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing

### Machine Learning
- **scikit-learn 1.2+** - Machine learning algorithms and preprocessing
- **XGBoost 2.0+** - Gradient boosting framework
- **CatBoost 1.2+** - Gradient boosting with categorical support

### Utilities
- **Dill** - Advanced serialization for Python objects
- **Bootstrap 4** - Frontend framework
- **Jinja2** - Template engine

## 🏗 Project Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Data Ingestion                       │
│  • CSV Reading                                          │
│  • Train/Test Split (80/20)                             │
│  • Data Persistence                                     │
└──────────────────┬──────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────┐
│                 Data Transformation                      │
│  • Numerical Pipeline (Imputation + Scaling)           │
│  • Categorical Pipeline (Imputation + OneHot)            │
│  • ColumnTransformer Integration                        │
│  • Preprocessor Persistence                             │
└──────────────────┬──────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────┐
│                  Model Training                          │
│  • 8 ML Algorithms                                       │
│  • GridSearchCV Hyperparameter Tuning                    │
│  • Model Evaluation (R² Score)                           │
│  • Best Model Selection & Persistence                    │
└──────────────────┬──────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────┐
│              Prediction Pipeline                        │
│  • Load Preprocessor & Model                            │
│  • Transform Input Data                                 │
│  • Generate Predictions                                 │
└──────────────────┬──────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────┐
│                  Flask Web App                          │
│  • RESTful API Endpoints                                │
│  • Input Validation                                     │
│  • Error Handling                                       │
│  • Response Rendering                                   │
└─────────────────────────────────────────────────────────┘
```

## 📦 Installation

### Prerequisites
- Python 3.11 or higher
- pip package manager
- Git (optional, for cloning)

### Step 1: Clone the Repository

```bash
git clone <your-repo-url>
cd ML-Project-Student-Performance-Prediction
```

### Step 2: Create Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python -c "import flask, sklearn, xgboost, catboost; print('All dependencies installed successfully!')"
```

## 🚀 Usage

### Quickstart

```bash
# 1) Create and activate venv, then install deps
python -m venv venv && .\venv\Scripts\Activate.ps1  # Windows
pip install -r requirements.txt

# 2) Train the model (saves artifacts to artifacts/)
python src/components/data_ingestion.py

# 3) Run the web app (serves on http://127.0.0.1:8080)
python app.py
```

### Training the Model

Train the model by running the complete pipeline:

```bash
python src/components/data_ingestion.py
```

This script will:
1. Load data from `notebook/data/stud.csv`
2. Split into training (80%) and test (20%) sets
3. Preprocess features (imputation, encoding, scaling)
4. Train 8 different models with hyperparameter tuning
5. Select the best model based on R² score
6. Save model and preprocessor to `artifacts/`

**Output:**
- Console displays the final R² score
- Models saved to `artifacts/model.pkl`
- Preprocessor saved to `artifacts/preprocessor.pkl`
- Logs written to `logs/` directory

### Running the Web Application

Start the Flask development server:

```bash
python app.py
```

The application will be available at:
- **Local:** `http://127.0.0.1:8080`
- **Network:** `http://0.0.0.0:8080` (accessible from other devices)

**Routes:**
- `GET /` - Landing page with project information
- `GET /predictdata` - Display prediction form
- `POST /predictdata` - Submit form and receive prediction

### Environment Configuration

Set environment variables for production:

```bash
# Enable debug mode (development only)
export FLASK_DEBUG=true
```

The server listens on port 8080 by default. To change it, set `PORT` in the environment and update `app.run` if needed.

**Windows PowerShell:**
```powershell
$env:FLASK_DEBUG="true"
```

### Docker (Optional)

Build and run with Docker:

```bash
# Build image
docker build -t student-perf:latest .

# Run container mapping port 8080
docker run --rm -p 8080:8080 --name student-perf student-perf:latest
```

After the container starts, open `http://127.0.0.1:8080`.

### Programmatic Usage

Use the prediction pipeline in your Python code:

```python
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# Create input data
sample = CustomData(
    gender="female",
    race_ethnicity="group B",
    parental_level_of_education="bachelor's degree",
    lunch="standard",
    test_preparation_course="none",
    reading_score=72,
    writing_score=74
)

# Get prediction
df = sample.get_data_as_dataframe()
pipeline = PredictPipeline()
prediction = pipeline.predict(df)

print(f"Predicted Math Score: {prediction[0]:.2f}")
```

## 📁 Project Structure

```
ML-Project-Student-Performance-Prediction/
│
├── app.py                          # Flask application entry point
├── setup.py                        # Package installation configuration
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── .gitignore                      # Git ignore rules
│
├── src/                            # Source code directory
│   ├── __init__.py
│   ├── exception.py                # Custom exception classes
│   ├── logger.py                   # Logging configuration
│   ├── utils.py                    # Utility functions (save/load objects)
│   │
│   ├── components/                 # ML pipeline components
│   │   ├── __init__.py
│   │   ├── data_ingestion.py      # Data loading and splitting
│   │   ├── data_transformation.py  # Feature preprocessing
│   │   └── model_trainer.py        # Model training and selection
│   │
│   └── pipeline/                   # Inference pipeline
│       ├── __init__.py
│       └── predict_pipeline.py     # Prediction pipeline and data classes
│
├── templates/                      # HTML templates
│   ├── index.html                  # Landing page
│   └── home.html                   # Prediction form page
│
├── artifacts/                      # Generated files (gitignored)
│   ├── data.csv                    # Raw dataset
│   ├── train.csv                   # Training dataset
│   ├── test.csv                    # Test dataset
│   ├── preprocessor.pkl             # Saved preprocessing pipeline
│   └── model.pkl                   # Trained model
│
├── logs/                          # Application logs (gitignored)
│   └── [timestamp].log           # Rotating log files
│
└── notebook/                      # Jupyter notebooks
    ├── 1. EDA STUDENT PERFORMANCE.ipynb
    ├── 2. MODEL TRAINING.ipynb
    └── data/
        └── stud.csv                # Source dataset
```

## 🤖 Model Details

### Algorithms Evaluated

The system evaluates the following machine learning algorithms:

1. **Linear Regression** - Baseline linear model
2. **Decision Tree** - Non-linear tree-based model
3. **Random Forest** - Ensemble of decision trees
4. **Gradient Boosting** - Sequential ensemble method
5. **K-Neighbors Regressor** - Instance-based learning
6. **XGBoost** - Optimized gradient boosting
7. **CatBoost** - Gradient boosting with categorical support
8. **AdaBoost** - Adaptive boosting ensemble

### Model Selection Process

1. **Hyperparameter Tuning**: Each algorithm undergoes grid search with 3-fold cross-validation
2. **Evaluation Metric**: R² (Coefficient of Determination) on test set
3. **Selection Criteria**: Model with highest R² score is selected
4. **Quality Threshold**: Minimum R² of 0.6 required (configurable)
5. **Final Training**: Selected model is retrained with best hyperparameters on full training set

### Input Features

**Categorical Features:**
- `gender`: Male/Female
- `race_ethnicity`: Group A/B/C/D/E
- `parental_level_of_education`: Associate's/Bachelor's/High school/Master's/Some college/Some high school
- `lunch`: Free/reduced/Standard
- `test_preparation_course`: None/Completed

**Numerical Features:**
- `reading_score`: Integer (0-100)
- `writing_score`: Integer (0-100)

**Target Variable:**
- `math_score`: Integer (0-100)

### Preprocessing Pipeline

**Numerical Features:**
- Missing value imputation (Median strategy)
- StandardScaler (with_mean=False)

**Categorical Features:**
- Missing value imputation (Most frequent strategy)
- One-Hot Encoding

## 📚 API Documentation

### Web API Endpoints

#### `GET /`
Landing page displaying project information and navigation.

**Response:** HTML page

---

#### `GET /predictdata`
Display prediction form.

**Response:** HTML form page

---

#### `POST /predictdata`
Submit prediction request with student data.

**Request Body (Form Data):**
```
gender: string (required) - "male" or "female"
ethnicity: string (required) - "group A" | "group B" | "group C" | "group D" | "group E"
parental_level_of_education: string (required)
lunch: string (required) - "free/reduced" or "standard"
test_preparation_course: string (required) - "none" or "completed"
reading_score: integer (required, 0-100)
writing_score: integer (required, 0-100)
```

**Success Response:**
```html
<!-- HTML page with prediction result -->
Predicted Math Score: 75.23
```

**Error Response:**
```html
<!-- HTML page with error message -->
Error: [error description]
```

---

### Python API

#### `CustomData`

Data container class for prediction inputs.

```python
CustomData(
    gender: str,
    race_ethnicity: str,
    parental_level_of_education: str,
    lunch: str,
    test_preparation_course: str,
    reading_score: int,
    writing_score: int
)
```

**Methods:**
- `get_data_as_dataframe() -> pd.DataFrame`: Converts input to DataFrame

---

#### `PredictPipeline`

Prediction pipeline for generating math score predictions.

```python
pipeline = PredictPipeline()
predictions = pipeline.predict(features: pd.DataFrame) -> np.ndarray
```

**Parameters:**
- `features`: Pandas DataFrame with required columns

**Returns:**
- NumPy array of predictions

## 🔧 Development

### Running Tests

Currently, the project uses console output and logging for validation. To add unit tests:

```bash
pytest tests/
```

### Code Style

Follow PEP 8 Python style guide. Consider using:
- **Black** for code formatting
- **Flake8** for linting
- **Pylint** for code quality

### Logging

Logs are automatically generated in the `logs/` directory with:
- Timestamp-based file names
- Rotation at 5MB with 5 backup files
- INFO level logging by default

View logs:
```bash
tail -f logs/[latest-log-file].log
```

## ❗ Troubleshooting

### Common Issues

**Issue: Model files not found**
```
FileNotFoundError: artifacts/model.pkl
```
**Solution:** Run the training script first:
```bash
python src/components/data_ingestion.py
```

---

**Issue: Port already in use**
```
OSError: [Errno 48] Address already in use
```
**Solution:** Change port or stop conflicting process:
```bash
# Option 1: Use different port mapping in Docker (recommended when using Docker)
docker run -p 5001:8080 student-perf:latest

# Option 2: Kill process on port 8080 (Linux/macOS)
lsof -ti:8080 | xargs kill
```

---

**Issue: Import errors**
```
ModuleNotFoundError: No module named 'src'
```
**Solution:** Ensure you're running from project root:
```bash
cd ML-Project-Student-Performance-Prediction
python app.py
```

---

**Issue: Path errors on Windows**
```
FileNotFoundError: Invalid path
```
**Solution:** The project uses `pathlib.Path` and `os.path.join` for cross-platform compatibility. Ensure you're using the latest version of the code.

---

**Issue: Low model accuracy**
```
CustomException: No best model found
```
**Solution:** Check data quality:
- Verify `notebook/data/stud.csv` exists and is valid
- Ensure sufficient training data
- Review preprocessing steps in logs

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Make your changes** following code style guidelines
4. **Commit with clear messages**
   ```bash
   git commit -m "Add amazing feature"
   ```
5. **Push to your branch**
   ```bash
   git push origin feature/amazing-feature
   ```
6. **Open a Pull Request**

### Contribution Ideas

- Add unit tests
- Implement additional ML algorithms
- Improve frontend UI/UX
- Add API documentation (Swagger/OpenAPI)
- Implement model versioning
- Add data visualization dashboard
- Support for batch predictions

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👤 Author

**Mohd Talha**
- Email: talhamoh017@gmail.com

## 🙏 Acknowledgments

- **scikit-learn** team for excellent ML tools
- **Flask** community for web framework
- **Bootstrap** for UI components
- Dataset contributors and educational research community

---

<div align="center">

**Made with ❤️ using Python and Machine Learning**

[⬆ Back to Top](#-student-performance-prediction-system)

</div>
