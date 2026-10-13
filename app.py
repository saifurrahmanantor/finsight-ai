
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="FinSight AI | Financial Resilience",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}
.hero {
    padding: 1.5rem;
    border-radius: 16px;
    background: linear-gradient(120deg, #10243A, #176B68);
    color: white;
    margin-bottom: 1.5rem;
}
.hero h1 {
    color: white;
    margin-bottom: 0.4rem;
}
.hero p {
    color: #E4F3F1;
    font-size: 1.05rem;
}
div[data-testid="stMetric"] {
    border: 1px solid rgba(128,128,128,0.25);
    padding: 1rem;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1>💰 FinSight AI</h1>
    <p>Make financial decisions with greater confidence.</p>
    <p>Risk-aware financial planning for university students in Bangladesh.</p>
</div>
""", unsafe_allow_html=True)


# =========================================================
# 2. FINANCIAL INPUTS
# =========================================================

st.sidebar.header("Your Financial Information")

balance = st.sidebar.number_input(
    "Current balance (BDT)",
    min_value=0,
    value=12000,
    step=500
)

income_min = st.sidebar.number_input(
    "Minimum monthly income (BDT)",
    min_value=0,
    value=8000,
    step=500
)

income_max = st.sidebar.number_input(
    "Maximum monthly income (BDT)",
    min_value=0,
    value=14000,
    step=500
)

expenses = st.sidebar.number_input(
    "Monthly essential expenses (BDT)",
    min_value=0,
    value=7000,
    step=500
)

semester_fee = st.sidebar.number_input(
    "Semester fee (BDT)",
    min_value=0,
    value=5000,
    step=1000
)

cost = st.sidebar.number_input(
    "Cost of things you want to buy (BDT)",
    min_value=0,
    value=20000,
    step=500
)

purchase_choice = st.sidebar.selectbox(
    "When are you consider to buy?",
    ["Do not buy", "Buy now", "Buy later"]
)

minimum_reserve = st.sidebar.number_input(
    "Minimum safety reserve (BDT)",
    min_value=0,
    value=2000,
    step=500
)

if income_min > income_max:
    st.error("Maximum monthly income must be at least minimum monthly income.")
    st.stop()


# =========================================================
# 3. MODEL SETTINGS
# =========================================================

WEEKS = 8
FEE_WEEK = 6
SIMULATIONS = 10000

purchase_weeks = {
    "Do not buy": None,
    "Buy now (Week 1)": 1,
    "Buy later (Week 4)": 4
}

selected_name = purchase_choice
selected_phone_week = purchase_weeks[selected_name]


# =========================================================
# 4. MONTE CARLO SIMULATION
# =========================================================

# Same random conditions for all scenarios make the comparison fairer.
rng = np.random.default_rng(42)

weekly_income_min = income_min * 12 / 52
weekly_income_max = income_max * 12 / 52
weekly_expenses = expenses * 12 / 52

income_draws = rng.uniform(
    weekly_income_min,
    weekly_income_max,
    size=(SIMULATIONS, WEEKS)
)

unexpected_draws = np.where(
    rng.random((SIMULATIONS, WEEKS)) < 0.15,
    rng.uniform(500, 2500, size=(SIMULATIONS, WEEKS)),
    0
)

scenario_results = {}

for scenario_name, phone_week in purchase_weeks.items():

    path_balances = np.full(
        SIMULATIONS,
        float(balance)
    )

    history = []

    for week in range(1, WEEKS + 1):

        path_balances += (
            income_draws[:, week - 1]
            - weekly_expenses
            - unexpected_draws[:, week - 1]
        )

        if week == FEE_WEEK:
            path_balances -= semester_fee

        if week == phone_week:
            path_balances -= phone_cost

        history.append(path_balances.copy())

    history = np.array(history).T
    final_balances = history[:, -1]
    fee_balances = history[:, FEE_WEEK - 1]

    # Any balance below the reserve during the forecast counts as a shortfall.
    shortfall_probability = np.mean(
        np.any(history < minimum_reserve, axis=1)
    )

    # Check whether the fee can be paid while retaining the reserve.
    # This uses the simulated balance after the fee payment in Week 6.
    fee_reserve_probability = np.mean(
        fee_balances >= minimum_reserve
    )

    # Balance immediately before the phone purchase:
    # initial balance plus cash flows through the preceding week.
    if phone_week is None:
        pre_purchase_balances = None
    elif phone_week == 1:
        pre_purchase_balances = np.full(
            SIMULATIONS, float(balance)
        )
    else:
        pre_purchase_balances = history[:, phone_week - 2].copy()

    # Affordability: can buy the phone and retain the reserve
    # at the purchase moment? Also protect the fee if it is still due.
    if phone_week is None:
        affordability_probability = np.nan
    else:
        projected_after_purchase = (
            pre_purchase_balances - phone_cost
        )

        if phone_week < FEE_WEEK:
            # Reserve must remain after buying and after paying the fee.
            projected_after_fee = (
                projected_after_purchase
                + np.sum(
                    income_draws[:, phone_week:FEE_WEEK],
                    axis=1
                )
                - weekly_expenses * (FEE_WEEK - phone_week)
                - np.sum(
                    unexpected_draws[:, phone_week:FEE_WEEK],
                    axis=1
                )
                - semester_fee
            )

            affordability_probability = np.mean(
                (projected_after_purchase >= minimum_reserve)
                & (projected_after_fee >= minimum_reserve)
            )
        else:
            affordability_probability = np.mean(
                projected_after_purchase >= minimum_reserve
            )

    scenario_results[scenario_name] = {
        "history": history,
        "final_balances": final_balances,
        "shortfall_probability": shortfall_probability,
        "average_final_balance": final_balances.mean(),
        "median_final_balance": np.median(final_balances),
        "p05_balance": np.percentile(final_balances, 5),
        "average_balance_after_fee": fee_balances.mean(),
        "fee_reserve_probability": fee_reserve_probability,
        "affordability_probability": affordability_probability
    }


# =========================================================
# 5. COMPARE SCENARIOS
# =========================================================

st.markdown("## Compare Your Financial Decisions")

rows = []

for name, result in scenario_results.items():

    risk = result["shortfall_probability"]

    if risk >= 0.30:
        risk_label = "High"
    elif risk >= 0.10:
        risk_label = "Moderate"
    else:
        risk_label = "Lower"

    affordability = result["affordability_probability"]

    rows.append({
        "Decision": name,
        "Shortfall Risk (%)": round(risk * 100, 2),
        "Average Final Balance (BDT)": round(
            result["average_final_balance"], 0
        ),
        "5th Percentile Balance (BDT)": round(
            result["p05_balance"], 0
        ),
        "Reserve After Fee (%)": round(
            result["fee_reserve_probability"] * 100, 2
        ),
        "Purchase Affordability (%)": (
            "Not applicable"
            if np.isnan(affordability)
            else round(affordability * 100, 2)
        ),
        "Risk Category": risk_label
    })

comparison = pd.DataFrame(rows)

st.dataframe(
    comparison,
    use_container_width=True,
    hide_index=True
)


# =========================================================
# 6. VALIDATION
# =========================================================

no_buy_avg = scenario_results["Do not buy"]["average_final_balance"]
buy_now_avg = scenario_results["Buy now (Week 1)"]["average_final_balance"]
buy_later_avg = scenario_results["Buy later (Week 4)"]["average_final_balance"]

assert len(comparison) == 3
assert comparison["Decision"].is_unique

assert comparison["Shortfall Risk (%)"].between(0, 100).all()

assert abs((no_buy_avg - buy_now_avg) - phone_cost) < 0.01
assert abs((no_buy_avg - buy_later_avg) - phone_cost) < 0.01

with st.expander("Model validation"):
    st.write("Expected phone-cost difference:", f"৳{phone_cost:,.0f}")
    st.write("Buy-now balance difference:", f"৳{no_buy_avg - buy_now_avg:,.2f}")
    st.write("Buy-later balance difference:", f"৳{no_buy_avg - buy_later_avg:,.2f}")
    st.success("Phone-cost validation tests passed.")


# =========================================================
# 7. SELECTED SCENARIO
# =========================================================

selected = scenario_results[selected_name]

st.markdown("## Your Selected Scenario")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Shortfall Probability",
    f"{selected['shortfall_probability']:.1%}"
)

col2.metric(
    "Average Final Balance",
    f"৳{selected['average_final_balance']:,.0f}"
)

col3.metric(
    "5th Percentile Balance",
    f"৳{selected['p05_balance']:,.0f}"
)

risk = selected["shortfall_probability"]

if risk >= 0.30:
    st.error("High modeled risk. Review your spending plan.")
elif risk >= 0.10:
    st.warning("Moderate modeled risk. Consider protecting your reserve.")
else:
    st.success("Lower modeled risk under the current assumptions.")


# =========================================================
# 8. IMPROVED RECOMMENDATION ENGINE
# =========================================================

st.markdown("## FinSight Recommendation")

no_buy = scenario_results["Do not buy"]
buy_now = scenario_results["Buy now (Week 1)"]
buy_later = scenario_results["Buy later (Week 4)"]

# A purchase is considered relatively affordable only if:
# 1. The probability of retaining the reserve after purchase
#    and protecting the upcoming fee is at least 90%.
# 2. The shortfall risk is below 10%.
#
# These are prototype thresholds, not universal financial rules.

AFFORDABILITY_THRESHOLD = 0.90
MAX_ACCEPTABLE_RISK = 0.10

def purchase_is_affordable(result):
    affordability = result["affordability_probability"]

    return (
        not np.isnan(affordability)
        and affordability >= AFFORDABILITY_THRESHOLD
        and result["shortfall_probability"] < MAX_ACCEPTABLE_RISK
    )

now_affordable = purchase_is_affordable(buy_now)
later_affordable = purchase_is_affordable(buy_later)

if now_affordable:
    recommendation = "Buy now (Week 1)"
    explanation = (
        "Under the modeled assumptions, buying now meets the prototype "
        "affordability and shortfall-risk thresholds. Confirm your real "
        "fee deadline and essential expenses before purchasing."
    )

elif later_affordable:
    recommendation = "Buy later (Week 4)"
    explanation = (
        "Buying now does not meet the prototype safety thresholds, "
        "but buying in Week 4 does. Recheck your actual balance and "
        "upcoming obligations before making the purchase."
    )

else:
    recommendation = "Do not buy yet"
    explanation = (
        "Neither purchase option meets the prototype safety thresholds "
        "under the current assumptions. Consider saving more, choosing "
        "a cheaper phone, or reassessing the purchase after the semester fee."
    )

col1, col2 = st.columns(2)

with col1:
    st.metric("Recommended Option", recommendation)

with col2:
    if recommendation == "Buy now (Week 1)":
        recommendation_risk = buy_now["shortfall_probability"]
    elif recommendation == "Buy later (Week 4)":
        recommendation_risk = buy_later["shortfall_probability"]
    else:
        recommendation_risk = no_buy["shortfall_probability"]

    st.metric(
        "Estimated Shortfall Risk",
        f"{recommendation_risk:.1%}"
    )

st.write(explanation)

st.caption(
    "Prototype thresholds: at least 90% modeled affordability and "
    "less than 10% shortfall risk for a purchase. These thresholds are "
    "design choices, not validated financial advice. The model does not "
    "include all real-world obligations or personal preferences."
)


# =========================================================
# 9. SHORTFALL RISK CHART
# =========================================================

st.markdown("## Shortfall Risk by Decision")

risk_chart = comparison.set_index("Decision")["Shortfall Risk (%)"]

st.bar_chart(risk_chart)


# =========================================================
# 10. CASH-FLOW TRAJECTORY
# =========================================================

st.markdown("## 8-Week Cash-Flow Trajectory")

selected_history = selected["history"]

weekly_median = np.median(selected_history, axis=0)
weekly_p05 = np.percentile(selected_history, 5, axis=0)
weekly_p95 = np.percentile(selected_history, 95, axis=0)

cashflow_df = pd.DataFrame({
    "Median balance (BDT)": weekly_median,
    "5th percentile (BDT)": weekly_p05,
    "95th percentile (BDT)": weekly_p95
}, index=[f"Week {i}" for i in range(1, WEEKS + 1)])

st.line_chart(cashflow_df)

st.caption(
    "The median is the middle simulated outcome. The 5th percentile "
    "is a downside outcome, not the absolute worst case."
)


# =========================================================
# 11. FINAL BALANCE DISTRIBUTION
# =========================================================

st.markdown("## Possible Balances After 8 Weeks")

fig, ax = plt.subplots(figsize=(10, 4))

for name, result in scenario_results.items():
    ax.hist(
        result["final_balances"],
        bins=40,
        alpha=0.35,
        label=name
    )

ax.axvline(
    minimum_reserve,
    linestyle="--",
    label="Minimum reserve"
)

ax.set_xlabel("Final balance (BDT)")
ax.set_ylabel("Number of simulations")
ax.legend()

fig.tight_layout()
st.pyplot(fig)
plt.close(fig)


# =========================================================
# 12. DISCLAIMER
# =========================================================

st.caption(
    "FinSight AI is a prototype, not a guarantee of financial safety. "
    "The model assumes uniform income within the selected range, a 15% "
    "weekly probability of an unexpected expense between ৳500 and ৳2,500, "
    "and a semester-fee payment in Week 6. Income timing, debt, inflation, "
    "and other obligations are not modeled. Results depend on assumptions."
)
