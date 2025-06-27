import streamlit as st

def retention_uplift(strategy: str, inactivity_days: int) -> int:
    """
    Simple rule-based model to estimate retention uplift percentage
    based on strategy and player inactivity days.
    """
    if strategy == "Bonus after inactivity" and inactivity_days >= 5:
        return 12
    elif strategy == "Special mission unlocked" and inactivity_days >= 3:
        return 9
    elif strategy == "Limited-time skin offer" and inactivity_days >= 10:
        return 7
    else:
        return 2  # Minimal effect

def run_simulator():
    st.title("⚙️ Player Retention Strategy Simulator")

    strategy = st.selectbox(
        "Select Retention Strategy:",
        ["Bonus after inactivity", "Special mission unlocked", "Limited-time skin offer"]
    )

    inactivity_days = st.slider("Player inactivity (days):", min_value=0, max_value=30, value=7)

    uplift = retention_uplift(strategy, inactivity_days)

    if uplift > 5:
        st.success(f"Estimated retention uplift: +{uplift}%")
    else:
        st.warning(f"Minimal effect expected. Try targeting more inactive players or a different strategy.")

    st.markdown("""
    **How this simulator works:**  
    It uses simple rules based on inactivity duration and the selected strategy to estimate how much player retention might improve.
    """)

