# 🚀 AI-Powered Startup Success Predictor & Advisor

*Data-driven insights and Generative AI to evaluate, predict, and scale the next big unicorn.*

[![Live Demo](https://img.shields.io/badge/Demo-Live_App-success?style=for-the-badge&logo=streamlit)](https://startup-success-predictor-d5u63hesntzh5ayhsm64ds.streamlit.app/) 
[![Python](https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python)](#)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?style=for-the-badge&logo=fastapi)](#)

---

##  The Problem Statement
Over **90% of startups fail** within their first few years. Founders often lack data-backed validation for their ideas and struggle to find actionable, contextual advice during critical phases like fundraising and MVP building. 

**The Solution:** This project bridges the gap between raw statistical data and strategic execution. It provides founders with a realistic success probability based on historical metrics and pairs it with an intelligent AI consultant to guide their next steps.

## 🌟 Key Features

1. **📈 Real-Time Predictive Engine**: 
   - Evaluates startups dynamically using a trained Random Forest classifier.
   - Considers critical business dimensions: Funding Capital, Team Size, Founders' Experience, and Market Scale.
   - Generates actionable, threshold-based suggestions based on the prediction score.
2. **🤖 Context-Aware AI Advisor**: 
   - Integrated with **Google Gemini 2.0 Flash** for high-speed, strategic business consulting.
   - Features a custom, highly responsive ChatGPT-style chat interface with built-in fallback mechanisms for high availability.
3. **📊 Interactive Analytics Dashboard**: 
   - Plotly-powered visual insights into market trends, funding vs. success correlations, and industry risk analysis.
4. **⚡ Headless API Architecture**: 
   - Includes a standalone FastAPI backend, allowing the prediction model to be consumed programmatically by mobile apps or other web services.

## 💎 What Makes This Project Unique?
Unlike standard ML projects that stop at a binary "Pass/Fail" prediction, this application combines **Predictive AI** (Scikit-Learn) with **Generative AI** (Google Gemini). It doesn't just tell founders *if* they will succeed—it tells them *why* and advises them on *how* to improve their odds. It is built with a product-first mindset, featuring custom UI/UX and robust error handling.

## ⚙️ How It Works (System Flow)
1. **Data Ingestion:** User inputs core startup metrics via the Streamlit UI or REST API.
2. **ML Inference:** The system loads a pre-trained `RandomForestClassifier` (`startup_model.pkl`) to calculate a precise success probability.
3. **Generative Consultation:** Users interact with the Gemini-powered AI advisor to discuss roadblocks, fundraising strategies, and product-market fit.
4. **Visualization:** The analytics engine processes the data and generates real-time performance graphs and industry comparisons.

## 🛠️ Tech Stack & Architecture

- **Frontend:** Streamlit, Custom HTML/CSS, Plotly
- **Backend & API:** FastAPI, Uvicorn, Python
- **Machine Learning:** Scikit-Learn, Pandas, NumPy, Joblib
- **Generative AI:** Google GenAI SDK (Gemini 2.0 Flash)

## 🧠 Model Details & Performance
- **Algorithm:** Random Forest Classifier (`n_estimators=10`)
- **Input Features:** `Funding ($)`, `Team Size`, `Founders Experience (Years)`, `Market Size (Ordinal: 0=Small, 1=Medium, 2=Large)`
- **Current Limitations:** The model is currently trained on a lightweight synthetic dataset to demonstrate end-to-end integration. (See *Future Scope* for scaling plans).

## 📸 Demo

*(Add screenshots of your UI here)*

> **[Live Streamlit App: Startup Success Predictor](https://startup-success-predictor-d5u63hesntzh5ayhsm64ds.streamlit.app/)**

## ⚙️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/avnish1505/Predicting-startup-success-using-AI.git
cd "Predicting-startup-success-using-AI"
```

### 2. Install Dependencies
It is recommended to use a virtual environment.
```bash
pip install -r requirements.txt
```

### 3. Setup API Keys
Create a `.streamlit/secrets.toml` file in the root directory and add your Google Gemini API key:
```toml
GEMINI_API_KEY = "your_actual_api_key_here"
```

### 4. Train the Machine Learning Model
Before running the app, train the model to generate the `startup_model.pkl` file:
```bash
python train_model.py
```

### 5. Run the Application
Start the Streamlit web application:
```bash
streamlit run app.py
```

*(Optional)* To run the FastAPI backend:
```bash
uvicorn api:app --reload
```

## 📁 Project Structure

- `app.py`: Main Streamlit application with customized UI.
- `predictor.py`: Core logic for loading the ML model and generating predictions.
- `advisor_ai.py`: Gemini AI integration with fallback mechanisms.
- `analytics.py`: Data visualization and dashboard metrics.
- `train_model.py`: Script to train and save the RandomForest model.
- `api.py`: FastAPI implementation for the predictor.
- `requirements.txt`: Python dependencies.

## 🚀 Future Scope

- Integrate a real-world, large-scale dataset (e.g., Crunchbase or Kaggle Startup datasets) for higher prediction accuracy.
- Add authentication/login for personalized user dashboards.
- Expand API functionality to include batch predictions.

---
*Built with ❤️ by [Avnish Singh](https://github.com/avnish1505)*