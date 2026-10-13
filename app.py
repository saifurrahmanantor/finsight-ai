
import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(
    page_title="FinSight AI",
    page_icon="💰",
    layout="wide"
)

st.title("💰 FinSight AI")
st.subheader("Financial decision assistant for university students")
st.caption("Plan your spending. Understand your financial risk.")

# User inputs
st.sidebar.header("Your Financial Information")

balance = st.sidebar.number_input(
    "Current balance (BDT)",
    min_value=0, value=12000, step=500
)

income_min = st.sidebar.number_input(
    "Minimum monthly income (BDT)",
    min_value=0, value=8000, step=500
)

income_max = st.sidebar.number_input(
    "Maximum monthly income (BDT)",
    min_value=0, value=14000, step=500
)

expenses = st.sidebar.number_input(
    "Monthly essential expenses (BDT)",
    min_value=0, value=7000, step=500
)

semester_fee = st.sidebar.number_input(
    "Semester fee (BDT)",
    min_value=0, value=5000, step=1000
)

phone_cost = st.sidebar.number_input(
    "Phone cost (BDT)",
    min_value=0, value=20000, step=500
)

purchase_week = st.sidebar.selectbox(
    "When will you buy the phone?",
    ["Do not buy", "Week 1", "Week 4"]
)

minimum_reserve = st.sidebar.number_input(
    "Minimum safety reserve (BDT)",
    min_value=0, value=2000, step=500
)

fee_week = 6
weeks = 8
simulations = 10000

if income_min > income_max:
    st.error("Maximum income must be at least minimum income.")
    st.stop()

phone_week = {
    "Do not buy": None,
    "Week 1": 1,
    "Week 4": 4
}[purchase_week]

# Monte Carlo engine
rng = np.random.default_rng(42)

weekly_income_min = income_min * 12 / 52
weekly_income_max = income_max * 12 / 52
weekly_expenses = expenses * 12 / 52

balances = np.full(simulations, float(balance))
history = []

for week in range(1, weeks + 1):
    income = rng.uniform(
        weekly_income_min,
        weekly_income_max,
        simulations
    )

    unexpected = np.where(
        rng.random(simulations) < 0.15,
        rng.uniform(500, 2500, simulations),
        0
    )

    balances += income - weekly_expenses - unexpected

    if week == fee_week:
        balances -= semester_fee

    if week == phone_week:
        balances -= phone_cost

    history.append(balances.copy())

history = np.array(history).T
final_balances = history[:, -1]

shortfall_probability = np.any(
    history < minimum_reserve, axis=1
).mean()

avg_balance = final_balances.mean()
p05_balance = np.percentile(final_balances, 5)

# Dashboard
st.markdown("### Your financial outlook")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Shortfall probability",
    f"{shortfall_probability:.1%}"
)

col2.metric(
    "Average final balance",
    f"৳{avg_balance:,.0f}"
)

col3.metric(
    "5th percentile balance",
    f"৳{p05_balance:,.0f}"
)

if shortfall_probability >= 0.30:
    st.error("High modeled risk — review your spending decision.")
elif shortfall_probability >= 0.10:
    st.warning("Moderate modeled risk — protect your safety reserve.")
else:
    st.success("Lower modeled risk — keep an emergency reserve.")

st.markdown("### Possible balances after 8 weeks")

chart_data = pd.DataFrame({
    "Final balance (BDT)": final_balances
})

st.hist_chart(chart_data, x="Final balance (BDT)")

st.markdown("### What does this mean?")
st.write(
    f"In this scenario, approximately "
    f"{shortfall_probability:.1%} of simulated paths "
    f"fell below your chosen safety reserve at least once."
)

st.caption(
    "Prototype only. Income variability and unexpected-expense "
    "assumptions are illustrative, not calibrated to survey data. "
    "Results are estimates, not financial guarantees."
)
