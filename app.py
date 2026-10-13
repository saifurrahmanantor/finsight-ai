
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
# 2. USER INPUTS
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

phone_cost = st.sidebar.number_input(
    "Phone cost (BDT)",
    min_value=0,
    value=20000,
    step=500
)

purchase_week = st.sidebar.selectbox(
    "When will you buy the phone?",
    ["Do not buy", "Week 1", "Week 4"]
)

minimum_reserve = st.sidebar.number_input(
    "Minimum safety reserve (BDT)",
    min_value=0,
    value=2000,
    step=500
)


# =========================================================
# 3. VALIDATE INPUTS
# =========================================================

if income_min > income_max:
    st.error("Maximum monthly income must be at least minimum monthly income.")
    st.stop()

fee_week = 6
weeks = 8
simulations = 10000

phone_week = {
    "Do not buy": None,
    "Week 1": 1,
    "Week 4": 4
}[purchase_week]


# =========================================================
# 4. MONTE CARLO SIMULATION
# =========================================================

# A fixed seed makes results reproducible for the same inputs.
rng = np.random.default_rng(42)

weekly_income_min = income_min * 12 / 52
weekly_income_max = income_max * 12 / 52
weekly_expenses = expenses * 12 / 52

# Generate the same random income and unexpected expenses
# for all three scenarios, making comparisons fairer.
income_draws = rng.uniform(
    weekly_income_min,
    weekly_income_max,
    size=(simulations, weeks)
)

unexpected_draws = np.where(
    rng.random((simulations, weeks)) < 0.15,
    rng.uniform(500, 2500, size=(simulations, weeks)),
    0
)

scenarios = {
    "Do not buy": None,
    "Buy now (Week 1)": 1,
    "Buy later (Week 4)": 4
}

scenario_results = {}

for scenario_name, scenario_phone_week in scenarios.items():

    path_balances = np.full(
        simulations,
        float(balance)
    )

    history = []

    for week in range(1, weeks + 1):

        # Add income and subtract essential and unexpected expenses.
        path_balances += (
            income_draws[:, week - 1]
            - weekly_expenses
            - unexpected_draws[:, week - 1]
        )

        # Pay semester fee in Week 6.
        if week == fee_week:
            path_balances -= semester_fee

        # Subtract phone cost in the selected purchase week.
        if week == scenario_phone_week:
            path_balances -= phone_cost

        history.append(path_balances.copy())

    history = np.array(history).T

    final_balances = history[:, -1]
    fee_week_balances = history[:, fee_week - 1]

    # A shortfall occurs if the balance falls below the
    # minimum reserve at any point during the 8-week period.
    shortfall_probability = np.any(
        history < minimum_reserve,
        axis=1
    ).mean()

    scenario_results[scenario_name] = {
        "history": history,
        "final_balances": final_balances,
        "shortfall_probability": shortfall_probability,
        "average_final_balance": final_balances.mean(),
        "median_final_balance": np.median(final_balances),
        "p05_balance": np.percentile(final_balances, 5),
        "average_balance_after_fee": fee_week_balances.mean(),
        "reserve_after_fee_probability": np.mean(
            fee_week_balances >= minimum_reserve
        )
    }


# =========================================================
# 5. COMPARE ALL THREE DECISIONS
# =========================================================

st.markdown("### Compare your financial decisions")

rows = []

for name, result in scenario_results.items():

    risk = result["shortfall_probability"]

    if risk >= 0.30:
        risk_label = "High"
    elif risk >= 0.10:
        risk_label = "Moderate"
    else:
        risk_label = "Lower"

    rows.append({
        "Decision": name,
        "Shortfall Risk (%)": round(risk * 100, 2),
        "Average Final Balance (BDT)": round(
            result["average_final_balance"], 0
        ),
        "5th Percentile Balance (BDT)": round(
            result["p05_balance"], 0
        ),
        "Average Balance After Fee (BDT)": round(
            result["average_balance_after_fee"], 0
        ),
        "Reserve After Fee (%)": round(
            result["reserve_after_fee_probability"] * 100, 2
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
# 6. MODEL VALIDATION
# =========================================================

st.markdown("### Model validation")

no_buy_avg = scenario_results["Do not buy"]["average_final_balance"]
buy_now_avg = scenario_results["Buy now (Week 1)"]["average_final_balance"]
buy_later_avg = scenario_results["Buy later (Week 4)"]["average_final_balance"]

buy_now_difference = no_buy_avg - buy_now_avg
buy_later_difference = no_buy_avg - buy_later_avg

st.write("Expected phone-cost difference:", f"৳{phone_cost:,.0f}")
st.write("Observed buy-now difference:", f"৳{buy_now_difference:,.2f}")
st.write("Observed buy-later difference:", f"৳{buy_later_difference:,.2f}")

# Buying a phone in either scenario must reduce the Week 8
# balance by exactly the phone cost, given shared random draws.
assert len(comparison) == 3
assert comparison["Decision"].is_unique
assert comparison["Shortfall Risk (%)"].between(0, 100).all()

assert abs(buy_now_difference - phone_cost) < 0.01
assert abs(buy_later_difference - phone_cost) < 0.01

st.success("All model validation tests passed.")


# =========================================================
# 7. SELECTED SCENARIO
# =========================================================

selected_name = {
    "Do not buy": "Do not buy",
    "Week 1": "Buy now (Week 1)",
    "Week 4": "Buy later (Week 4)"
}[purchase_week]

# Use the same results dictionary for every displayed metric.
selected = scenario_results[selected_name]

st.markdown("### Your selected scenario")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Shortfall probability",
    f"{selected['shortfall_probability']:.1%}"
)

col2.metric(
    "Average final balance",
    f"৳{selected['average_final_balance']:,.0f}"
)

col3.metric(
    "5th percentile balance",
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
# 8. RECOMMENDATION ENGINE
# =========================================================

st.markdown("### FinSight Recommendation")

# Rank by lowest shortfall risk, then highest average final balance.
ranked_scenarios = sorted(
    scenario_results.items(),
    key=lambda item: (
        item[1]["shortfall_probability"],
        -item[1]["average_final_balance"]
    )
)

best_name, best_result = ranked_scenarios[0]

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Recommended option",
        best_name
    )

with col2:
    st.metric(
        "Estimated shortfall risk",
        f"{best_result['shortfall_probability']:.1%}"
    )

st.write(
    f"Based on the current assumptions, **{best_name}** "
    "has the lowest estimated shortfall risk. If multiple options "
    "have the same risk, the model prefers the higher average final balance."
)

st.caption(
    "This recommendation compares only the modeled scenarios. "
    "It does not account for every personal need or the urgency of replacing a phone."
)


# =========================================================
# 9. SHORTFALL RISK CHART
# =========================================================

st.markdown("### Shortfall risk by decision")

risk_chart = comparison.set_index("Decision")["Shortfall Risk (%)"]

st.bar_chart(risk_chart)


# =========================================================
# 10. FINAL BALANCE DISTRIBUTION
# =========================================================

st.markdown("### Possible balances after 8 weeks")

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
# 11. DEBUG TABLE — TEMPORARY
# =========================================================

with st.expander("Debug — Inspect inputs and simulation results"):

    st.write("Current inputs:")

    st.write({
        "Current balance": balance,
        "Minimum monthly income": income_min,
        "Maximum monthly income": income_max,
        "Monthly expenses": expenses,
        "Semester fee": semester_fee,
        "Phone cost": phone_cost,
        "Minimum reserve": minimum_reserve,
        "Selected scenario": selected_name
    })

    debug_rows = []

    for name, result in scenario_results.items():
        debug_rows.append({
            "Scenario": name,
            "Shortfall risk (%)": round(
                result["shortfall_probability"] * 100, 2
            ),
            "Average final balance (BDT)": round(
                result["average_final_balance"], 2
            ),
            "Minimum simulated balance (BDT)": round(
                result["history"].min(), 2
            ),
            "5th percentile final balance (BDT)": round(
                result["p05_balance"], 2
            )
        })

    st.dataframe(
        pd.DataFrame(debug_rows),
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# 12. DISCLAIMER
# =========================================================

st.caption(
    "FinSight AI is a prototype. Results depend on illustrative assumptions "
    "and are not empirical estimates of real-world financial risk. "
    "The current model assumes uniform income within the selected range, "
    "a 15% weekly chance of an unexpected expense, and a fixed semester-fee "
    "payment in Week 6. Risk thresholds are preliminary."
)
