import streamlit as st
from src import data_processing, model, explainability, nlp_analysis, strategy_simulator
import os

# File paths (update paths as needed)
RAW_DATA_PATH = 'data/raw/player_data_enhanced.csv'
PROCESSED_DATA_PATH = 'data/processed/player_data_processed.csv'
REVIEWS_PATH = 'data/processed/reviews.csv'
MODEL_PATH = 'models/rf_model.joblib'

def main():
    st.title("🎮 Player Retention Analysis Major Project")

    menu = ["Data Processing & EDA", "Model Training & Evaluation", "Explainability (SHAP)",
            "NLP Review Analysis", "Retention Strategy Simulator"]
    choice = st.sidebar.selectbox("Choose a Module", menu)

    if choice == "Data Processing & EDA":
        st.header("Data Processing & Exploratory Data Analysis (EDA)")
        if st.button("Run Data Processing Pipeline"):
            df = data_processing.load_and_process(RAW_DATA_PATH, PROCESSED_DATA_PATH)
            st.success("Data processed and saved!")
        if os.path.exists(PROCESSED_DATA_PATH):
            df = data_processing.load_raw_data(PROCESSED_DATA_PATH)
            st.write(df.head())
            data_processing.show_eda(df)
        else:
            st.warning("Processed data not found. Run pipeline above.")

    elif choice == "Model Training & Evaluation":
        st.header("Train and Evaluate Churn Prediction Model")
        if os.path.exists(PROCESSED_DATA_PATH):
            df = data_processing.load_raw_data(PROCESSED_DATA_PATH)
            if st.button("Train Model"):
                model_, X_test = model.train_and_evaluate(df, use_grid_search=False, model_path=MODEL_PATH)
                st.success("Model trained and saved.")
                st.write("Test data sample:")
                st.write(X_test.head())
        else:
            st.warning("Processed data not found. Please process data first.")

    elif choice == "Explainability (SHAP)":
        st.header("Model Explainability with SHAP")
        if os.path.exists(MODEL_PATH) and os.path.exists(PROCESSED_DATA_PATH):
            model_ = model.load_model(MODEL_PATH)
            df = data_processing.load_raw_data(PROCESSED_DATA_PATH)
            X = df.drop(columns=['player_id', 'churn'])
            explainability.shap_summary_plot(model_, X)
        else:
            st.warning("Model or processed data missing. Train model first.")

    elif choice == "NLP Review Analysis":
        st.header("Player Review NLP Analysis")
        if os.path.exists(REVIEWS_PATH):
            nlp_analysis.run_nlp_dashboard(REVIEWS_PATH)
        else:
            st.warning("Reviews dataset not found.")

    elif choice == "Retention Strategy Simulator":
        strategy_simulator.run_simulator()

if __name__ == "__main__":
    main()
