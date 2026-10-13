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

# Compare scenarios
st.markdown("### Compare your financial decisions")
st.write("All options use the same simulated income and unexpected-expense paths for a fairer comparison.")
rows = []
for name, result in results.items():
    risk = result["shortfall_probability"]
    label = "High" if risk >= 0.30 else ("Moderate" if risk >= 0.10 else "Lower")
    rows.append({
        "Decision": name,
        "Shortfall Risk (%)": round(risk * 100, 2),
        "Average Final Balance (BDT)": round(result["average_final_balance"]),
        "5th Percentile Final Balance (BDT)": round(result["p05_balance"]),
        "Average Balance After Fee (BDT)": round(result["average_balance_after_fee"]),
        "Reserve After Fee (%)": round(result["reserve_after_fee_probability"] * 100, 2),
        "Risk Category": label,
    })
comparison = pd.DataFrame(rows)
st.dataframe(comparison, use_container_width=True, hide_index=True)

# Basic arithmetic validation
st.markdown("### Model validation")
no_buy_avg = results["Do not buy"]["average_final_balance"]
buy_now_avg = results["Buy now (Week 1)"]["average_final_balance"]
buy_later_avg = results["Buy later (Week 4)"]["average_final_balance"]
now_difference = no_buy_avg - buy_now_avg
later_difference = no_buy_avg - buy_later_avg

st.write(f"Expected {item_name.lower()} cost difference: ৳{float(item_cost):,.2f}")
st.write(f"Observed buy-now difference: ৳{now_difference:,.2f}")
st.write(f"Observed buy-later difference: ৳{later_difference:,.2f}")

valid = (
    len(comparison) == 3
    and comparison["Decision"].is_unique
    and comparison["Shortfall Risk (%)"].between(0, 100).all()
    and np.isclose(now_difference, float(item_cost), atol=0.01)
    and np.isclose(later_difference, float(item_cost), atol=0.01)
)
if valid:
    st.success("Validation passed: scenario count, risk range, and item-cost arithmetic checks are OK.")
else:
    st.error("A validation check failed. Please review the calculations.")

# Affordability check: keep reserve at purchase and after fee if fee is still due
def affordability_probability(purchase_week):
    if purchase_week is None:
        return np.nan
    if purchase_week == 1:
        before_purchase = np.full(SIMULATIONS, float(balance), dtype=float)
    else:
        # Use the no-purchase path at the end of the week before buying.
        before_purchase = results["Do not buy"]["history"][:, purchase_week - 2].copy()

    after_purchase = before_purchase - float(item_cost)
    safe_at_purchase = after_purchase >= float(minimum_reserve)

    if purchase_week < FEE_WEEK:
        remaining_income = income_draws[:, purchase_week:FEE_WEEK].sum(axis=1)
        remaining_costs = weekly_expenses * (FEE_WEEK - purchase_week)
        remaining_shocks = unexpected_draws[:, purchase_week:FEE_WEEK].sum(axis=1)
        after_fee = after_purchase + remaining_income - remaining_costs - remaining_shocks - float(semester_fee)
        return float(np.mean(safe_at_purchase & (after_fee >= float(minimum_reserve))))
    return float(np.mean(safe_at_purchase))

afford_now = affordability_probability(1)
afford_later = affordability_probability(4)
risk_now = results["Buy now (Week 1)"]["shortfall_probability"]
risk_later = results["Buy later (Week 4)"]["shortfall_probability"]
risk_no_buy = results["Do not buy"]["shortfall_probability"]

st.markdown("### FinSight Recommendation")
st.write(f"Estimated affordability if bought now: **{afford_now:.1%}**")
st.write(f"Estimated affordability if bought in Week 4: **{afford_later:.1%}**")

AFFORDABILITY_LIMIT = 0.90
RISK_LIMIT = 0.10
now_ok = afford_now >= AFFORDABILITY_LIMIT and risk_now < RISK_LIMIT
later_ok = afford_later >= AFFORDABILITY_LIMIT and risk_later < RISK_LIMIT

if now_ok:
    recommendation = "Buy now (Week 1)"
    reason = "This option meets the prototype affordability and shortfall-risk thresholds."
elif later_ok:
    recommendation = "Buy later (Week 4)"
    reason = "Buying now fails the prototype thresholds, but buying in Week 4 meets them."
else:
    recommendation = "Do not buy yet"
    reason = "Neither purchase timing meets both prototype thresholds. Consider waiting or choosing a lower-cost alternative."

rec1, rec2 = st.columns(2)
rec1.metric("Suggested decision", recommendation)
rec2.metric("Shortfall risk if no purchase", f"{risk_no_buy:.1%}")
st.write(reason)
st.caption("The 90% affordability and 10% risk cutoffs are preliminary design choices, not validated standards.")

# Selected scenario
selected_name = {
    None: "Do not buy",
    1: "Buy now (Week 1)",
    4: "Buy later (Week 4)",
}[selected_purchase_week]
selected = results[selected_name]

st.markdown("### Your selected scenario")
c1, c2, c3 = st.columns(3)
c1.metric("Shortfall probability", f"{selected['shortfall_probability']:.1%}")
c2.metric("Average final balance", f"৳{selected['average_final_balance']:,.0f}")
c3.metric("5th percentile final balance", f"৳{selected['p05_balance']:,.0f}")

risk = selected["shortfall_probability"]
if risk >= 0.30:
    st.error("High modeled risk. Review your spending plan.")
elif risk >= 0.10:
    st.warning("Moderate modeled risk. Consider protecting your reserve.")
else:
    st.success("Lower modeled risk under the current assumptions.")

# Charts
st.markdown("### Shortfall risk by decision")
st.bar_chart(comparison.set_index("Decision")["Shortfall Risk (%)"])

st.markdown("### Possible balances after 8 weeks")
fig, ax = plt.subplots(figsize=(10, 4))
for name, result in results.items():
    ax.hist(result["final_balances"], bins=40, alpha=0.35, label=name)
ax.axvline(float(minimum_reserve), linestyle="--", label="Minimum reserve")
ax.set_xlabel("Final balance (BDT)")
ax.set_ylabel("Number of simulations")
ax.legend()
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.markdown("### Cash-flow trajectory")
trajectory = pd.DataFrame({
    name: np.mean(result["history"], axis=0)
    for name, result in results.items()
}, index=[f"Week {i}" for i in range(1, WEEKS + 1)])
st.line_chart(trajectory)

with st.expander("Model assumptions and limitations"):
    st.markdown("""
    - Forecast horizon: 8 weeks; semester fee is deducted in Week 6.
    - Monthly income is converted to a weekly range and sampled uniformly.
    - Essential expenses are treated as a fixed weekly average.
    - Each week has a 15% chance of an unexpected expense between ৳500 and ৳2,500.
    - Delayed or lost income is not explicitly modeled yet.
    - Shared random conditions make comparisons fairer but do not establish real-world predictive accuracy.
    - The risk thresholds are preliminary prototype settings, not validated standards.
    """)

st.caption("FinSight AI is a prototype. Outputs are scenario estimates, not guarantees or professional financial advice.")
