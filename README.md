# Fraud Detection API 🔍

A machine learning REST API that detects fraudulent credit card transactions in real-time.

## Tech Stack
- **Model**: XGBoost (trained on 284,807 transactions)
- **API**: FastAPI + Uvicorn
- **Validation**: Pydantic
- **Imbalance Handling**: SMOTE (0.4% fraud ratio)

## Model Performance
| Metric | Score |
|--------|-------|
| Precision | 69% |
| Recall | 87% |
| F1 Score | 0.77 |

> Recall prioritized over Precision — missing a fraud is worse than a false alarm.

## API Endpoints

### GET /
Health check
```json
{ "message": "Fraud Detection API is running ✅" }
```

### POST /predict
Send transaction data, get fraud prediction.

**Request:** 30 features (Time, V1–V28, Amount)  
**Response:**
```json
{
  "prediction": "fraud",
  "confidence": 0.9134,
  "amount": 150.0
}
```

## Run Locally
```bash
python -m venv env
source env/bin/activate
pip install fastapi uvicorn joblib xgboost scikit-learn
python -m uvicorn main:app --reload
```

Open http://127.0.0.1:8000/docs for interactive Swagger UI.
