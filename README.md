# Player Retention Analysis

A complete **Streamlit + Machine Learning + SHAP + NLP** project to analyze, predict, and improve player retention in games. This project simulates a game environment, models churn behavior, and provides insights using SHAP, NLP word clouds, and strategy simulators.

---

## Features

- Interactive EDA Dashboard  
- Churn Prediction using Random Forest  
- SHAP Explainability for Feature Importance  
- NLP Word Cloud from Player Reviews  
- AI-based Retention Strategy Simulator  
- Dynamic Churn Predictor UI (via Streamlit)

---

## Project Structure

```
player-retention-analysis/
├── app.py
├── requirements.txt
├── data/
│   └── processed/
│       └── reviews.csv
│   └── raw/
│       └── player_data_enhanced.csv
├── src/
│   ├── data_processing.py
│   ├── model.py
│   ├── explainability.py
│   ├── visualizations.py
│   ├── nlp_analysis.py
│   ├── strategy_simulator.py
│   └── utils.py
├── tests/
│   ├── test_data_processing.py
│   └── test_model.py
└── README.md
```

---

## Installation

1. **Clone the repository:**

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

---

## Run the App

```bash
streamlit run app.py
```

This will launch the interactive dashboard in your browser.

---

## Running Tests

Run unit tests using:

```bash
pytest tests/
```

---

## Tech Stack

- **Python 3.10+**
- Streamlit
- scikit-learn
- SHAP
- Seaborn / Matplotlib
- WordCloud / TextBlob
- NLTK
- pandas / NumPy

---

## Screenshots

![image](https://github.com/user-attachments/assets/6362a63f-1106-45b9-8a0a-de0fa7a8192a)
![image](https://github.com/user-attachments/assets/063681c3-c6ed-4531-872a-0b0aefd481d6)
![image](https://github.com/user-attachments/assets/f0b1232c-75f5-4f66-953c-f4c9183b4380)
![image](https://github.com/user-attachments/assets/2211a685-3b35-4924-b2a7-bfc4aa55da7d)
![image](https://github.com/user-attachments/assets/3302158a-dafc-402d-b3c8-3d6f76fb4ac7)



---
