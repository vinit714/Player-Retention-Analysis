import shap
import streamlit as st
import matplotlib.pyplot as plt

def shap_summary_plot(model, X):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    st.subheader("SHAP Summary Plot")

    # For binary classification (list of 2 arrays)
    if isinstance(shap_values, list) and len(shap_values) == 2:
        shap_to_use = shap_values[1]  # class 1
    else:
        shap_to_use = shap_values  # regression or multiclass fallback

    fig, ax = plt.subplots()
    shap.summary_plot(shap_to_use, X, show=False)
    st.pyplot(fig)


def shap_force_plot(model, X, index=0):
    """
    Display SHAP force plot for a single prediction instance.
    Args:
        model: Trained model.
        X (pd.DataFrame): Feature data.
        index (int): Index of the sample to explain.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    st.subheader(f"SHAP Force Plot for Sample #{index}")
    shap.initjs()
    force_plot = shap.force_plot(explainer.expected_value[1], shap_values[1][index,:], X.iloc[index,:], matplotlib=True)
    
    fig = plt.gcf()
    st.pyplot(fig)
