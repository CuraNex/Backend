# CuraNex — Backend API & ML Engine

This directory contains the core intelligence of the **CuraNex** hybrid ensemble demand forecasting system. It powers the data processing pipelines, machine learning models, API endpoints, and the interactive forecasting dashboard.

## 🛠️ Technology Stack
- **API Framework:** FastAPI, Uvicorn
- **Machine Learning:** PyTorch, LightGBM, XGBoost, Scikit-Learn
- **Data Handling:** Pandas, Polars
- **Analytics Dashboard:** Streamlit
- **Experiment Tracking:** MLflow

## ⚙️ Prerequisites
- **Python 3.10–3.12** is recommended to ensure compatibility with PyTorch and our MLOps stack.

## 🚀 Setup & Installation

1. **Create and Activate a Virtual Environment:**
   ```bash
   python -m venv .venv
   
   # On Windows:
   .\.venv\Scripts\Activate.ps1
   
   # On macOS/Linux:
   source .venv/bin/activate
   ```

2. **Install Dependencies:**
   Make sure you have sufficient disk space, as installing ML libraries (like PyTorch) can take some time.
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

## 🏃‍♂️ Running the API Server

Start the FastAPI application using Uvicorn:
```bash
python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```
*(Alternatively, you can just run `python api/main.py`)*

Once running, you can access the interactive API documentation (Swagger UI) at:
👉 **http://localhost:8000/docs**

### Data Handling Note
The server reads initial context from `Backend/data/synthetic/` and writes evaluation outputs to `Backend/evaluation/`. Sample synthetic data is included out-of-the-box.

## 📊 Running the Streamlit Dashboard (Optional)

We use Streamlit to provide deep insights into model metrics, predictions, and quantile forecasts. To run the analytics dashboard:

```bash
streamlit run dashboard/app.py
```
*(Streamlit will automatically open the dashboard in your browser on an available port.)*
