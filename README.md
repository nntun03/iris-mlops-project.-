# iris-mlops-project.-
a complete, end-to-end MLOps cycle for a simple model. a system that you can run on your own machine (or cloud) that embodies the core principles of MLOps: reproducibility, automation, testing, deployment, and monitoring. Classify iris flowers into one of three species based on sepal and petal measurements.

## What this adds:
- Complete end-to-end MLOps pipeline for Iris flower classification
- FastAPI backend with live prediction endpoints
- Data validation and model training scripts
- Automated testing and CI/CD workflow setup

## Features:
- ✅ Data validation and preprocessing
- ✅ Model training with RandomForest
- ✅ REST API with FastAPI
- ✅ Interactive documentation at `/docs`
- ✅ Proper project structure with virtual environment

## Project Structure

IRIS-MLOPS-PROJECT/
├── src/
│   ├── __init__.py
│   ├── data_validation.py
│   ├── train.py
│   └── predict.py
├── tests/
│   └── test_basics.py
├── models/                 # (ignored by git)
├── venv/                  # (ignored by git)
├── requirements.txt
└── .gitignore

## Key URLs
🌐 Local API Docs: http://127.0.0.1:8000/docs

🐙 GitHub Repository: https://github.com/nntun03/iris-mlops-project.-/

## 1. Setup Environment
bash
# Create virtual environment
python -m venv venv

# Activate (Windows PowerShell)
.\venv\Scripts\activate.ps1

# Install dependencies
pip install -r requirements.txt

## 2. Train Model
bash
python src/train.py
Output: Creates models/iris_rf_model.joblib

## 3. Run API Server
bash
uvicorn src.predict:app --reload --host 0.0.0.0 --port 8000

## 4. Test API - Open in Browser
text
📊 Interactive Documentation: http://127.0.0.1:8000/docs
📝 Alternative Docs:         http://127.0.0.1:8000/redoc
🏠 Root Endpoint:            http://127.0.0.1:8000/

## 5. Git Commands
bash
# Stage changes
git add .

# Commit
git commit -m "feat: description of changes"

# Push to branch
git push origin feat2

## API Testing Examples
Test in http://127.0.0.1:8000/docs using these JSON inputs:

Setosa
json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
Versicolor
json
{
  "sepal_length": 6.4,
  "sepal_width": 3.2,
  "petal_length": 4.5,
  "petal_width": 1.5
}
Virginica
json
{
  "sepal_length": 6.3,
  "sepal_width": 3.3,
  "petal_length": 6.0,
  "petal_width": 2.5
}

## 
Troubleshooting Commands
Fix Virtual Environment
bash
# Delete corrupted environment
deactivate
Remove-Item -Recurse -Force venv

# Create fresh environment
python -m venv venv
.\venv\Scripts\activate.ps1
pip install -r requirements.txt
Check Python Path
bash
python --version
where python
Test Import Issues
bash
# Test core modules
python -c "import joblib, fastapi, sklearn; print('All packages installed!')"

# Test data validation
python -c "from src.data_validation import get_iris_data, validate_data; df = get_iris_data(); validate_data(df); print('✅ Data validation passed!')"

