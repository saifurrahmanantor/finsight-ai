import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="FinSight AI | Financial Resilience", page_icon="💰", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 2rem; padding-bottom: 2rem;}
.hero {padding: 1.5rem; border-radius: 16px; background: linear-gradient(120deg,#10243A,#176B68); color: white; margin-bottom: 1.5rem;}
.hero h1 {color: white;}
.hero p {color: #E4F3F1; font-size: 1.02rem;}
div[data-testid="stMetric"] {border: 1px solid rgba(128,128,128,.25); padding: 1rem; border-radius: 12px;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
<h1>💰 FinSight AI</h1>
<p>Make financial decisions with greater confidence.</p>
<p>Risk-aware financial planning for university students in Bangladesh.</p>
</div>
""", unsafe_allow_html=True)

# User inputs
st.sidebar.header("Your Financial Information")
balance = st.sidebar.number_input("Current balance (BDT)", min_value=0, value=12000, step=500)
income_min = st.sidebar.number_input("Minimum monthly income (BDT)", min_value=0, value=8000, step=500)
income_max = st.sidebar.number_input("Maximum monthly income (BDT)", min_value=0, value=14000, step=500)
expenses = st.sidebar.number_input("Monthly essential expenses (BDT)", min_value=0, value=7000, step=500)
semester_fee = st.sidebar.number_input("Upcoming semester fee (BDT)", min_value=0, value=5000, step=1000)
item_name = st.sidebar.text_input("What do you want to buy?", value="Laptop").strip()
if not item_name:
    item_name = "Item"
item_cost = st.sidebar.number_input(f"{item_name} cost (BDT)", min_value=0, value=20000, step=500)
purchase_choice = st.sidebar.selectbox(
    f"When are you considering buying the {item_name.lower()}?",
    ["Do not buy", "Buy now (Week 1)", "Buy later (Week 4)"]
)
minimum_reserve = st.sidebar.number_input("Minimum safety reserve (BDT)", min_value=0, value=2000, step=500)

WEEKS = 8
FEE_WEEK = 6
SIMULATIONS = 10000
SEED = 42

if income_min > income_max:
    st.error("Maximum monthly income must be at least minimum monthly income.")
    st.stop()

selected_purchase_week = {
    "Do not buy": None,
    "Buy now (Week 1)": 1,
    "Buy later (Week 4)": 4,
}[purchase_choice]

# Monte Carlo simulation: shared random conditions across all decisions
rng = np.random.default_rng(SEED)
weekly_income_min = float(income_min) * 12 / 52
weekly_income_max = float(income_max) * 12 / 52
weekly_expenses = float(expenses) * 12 / 52

income_draws = rng.uniform(weekly_income_min, weekly_income_max, size=(SIMULATIONS, WEEKS))
unexpected_mask = rng.random((SIMULATIONS, WEEKS)) < 0.15
unexpected_amounts = rng.uniform(500, 2500, size=(SIMULATIONS, WEEKS))
unexpected_draws = np.where(unexpected_mask, unexpected_amounts, 0.0)

scenarios = {"Do not buy": None, "Buy now (Week 1)": 1, "Buy later (Week 4)": 4}
results = {}

for scenario_name, scenario_purchase_week in scenarios.items():
    balances = np.full(SIMULATIONS, float(balance), dtype=float)
    history = np.zeros((SIMULATIONS, WEEKS), dtype=float)

    for week in range(1, WEEKS + 1):
        balances += income_draws[:, week - 1] - weekly_expenses - unexpected_draws[:, week - 1]
        if week == FEE_WEEK:
            balances -= float(semester_fee)
        if scenario_purchase_week is not None and week == scenario_purchase_week:
            balances -= float(item_cost)
        history[:, week - 1] = balances

    final_balances = history[:, -1].copy()
    fee_balances = history[:, FEE_WEEK - 1].copy()
    results[scenario_name] = {
        "history": history,
        "final_balances": final_balances,
        "shortfall_probability": float(np.mean(np.any(history < minimum_reserve, axis=1))),
        "average_final_balance": float(np.mean(final_balances)),
        "median_final_balance": float(np.median(final_balances)),
        "p05_balance": float(np.percentile(final_balances, 5)),
        "average_balance_after_fee": float(np.mean(fee_balances)),
        "reserve_after_fee_probability": float(np.mean(fee_balances >= minimum_reserve)),
    }
