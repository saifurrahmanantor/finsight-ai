
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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


# Monte Carlo engine: fair scenario comparison
rng = np.random.default_rng(42)

weekly_income_min = income_min * 12 / 52
weekly_income_max = income_max * 12 / 52
weekly_expenses = expenses * 12 / 52

# Generate the same random conditions for every decision
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

for scenario_name, phone_week in scenarios.items():
    path_balances = np.full(
        simulations, float(balance)
    )
    history = []

    for week in range(1, weeks + 1):
        path_balances += (
            income_draws[:, week - 1]
            - weekly_expenses
            - unexpected_draws[:, week - 1]
        )

        if week == fee_week:
            path_balances -= semester_fee

        if week == phone_week:
            path_balances -= phone_cost

        history.append(path_balances.copy())

    history = np.array(history).T
    final_balances = history[:, -1]

    # Risk means reserve breached at any point
    shortfall_probability = np.any(
        history < minimum_reserve, axis=1
    ).mean()

  
    fee_week_balances = history[:, fee_week - 1]

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

# Show all three decisions
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

# Basic validation checks
assert len(comparison) == 3
assert comparison["Decision"].is_unique
assert comparison["Shortfall Risk (%)"].between(0, 100).all()

# Buying later or now should cost the same amount by Week 8
no_buy_avg = scenario_results["Do not buy"][
    "average_final_balance"
]
buy_now_avg = scenario_results["Buy now (Week 1)"][
    "average_final_balance"
]
buy_later_avg = scenario_results["Buy later (Week 4)"][
    "average_final_balance"
]

print("Decision count:", len(comparison))
print("No-buy average:", round(no_buy_avg, 2))
print("Buy-now average:", round(buy_now_avg, 2))
print("Buy-later average:", round(buy_later_avg, 2))

print(
    "Buy-now balance difference:",
    round(no_buy_avg - buy_now_avg, 2)
)
print(
    "Buy-later balance difference:",
    round(no_buy_avg - buy_later_avg, 2)
)

st.dataframe(comparison, use_container_width=True, hide_index=True)

st.markdown("### Model validation")

no_buy_avg = scenario_results["Do not buy"][
    "average_final_balance"
]
buy_now_avg = scenario_results["Buy now (Week 1)"][
    "average_final_balance"
]
buy_later_avg = scenario_results["Buy later (Week 4)"][
    "average_final_balance"
]

st.write(
    "Expected phone-cost difference:",
    f"৳{phone_cost:,.0f}"
)

st.write(
    "Observed buy-now difference:",
    f"৳{no_buy_avg - buy_now_avg:,.2f}"
)

st.write(
    "Observed buy-later difference:",
    f"৳{no_buy_avg - buy_later_avg:,.2f}"
)

assert abs(
    (no_buy_avg - buy_now_avg) - phone_cost
) < 0.01

assert abs(
    (no_buy_avg - buy_later_avg) - phone_cost
) < 0.01

st.success("Both phone-cost validation tests passed.")


# Selected scenario
selected_name = {
    None: "Do not buy",
    1: "Buy now (Week 1)",
    4: "Buy later (Week 4)"
}[phone_week]

selected = scenario_results[selected_name]

# --- Decision Recommendation Engine ---

st.subheader(" FinSight Recommendation")

# Lower risk is better; use final balance as a tie-breaker.
ranked_scenarios = sorted(
    scenario_results.items(),
    key=lambda item: (
        item[1]["shortfall_probability"],
        -item[1]["average_final_balance"]
    )
)

best_name, best_result = ranked_scenarios[0]

st.write(
    f"Based on your current assumptions, **{best_name}** "
    f"has the lowest estimated probability of falling below "
    f"your minimum reserve."
)

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

st.caption(
    "This recommendation compares only the modeled scenarios. "
    "It is not a guarantee; review the fee deadline, essential expenses, "
    "and your actual financial needs before deciding."
)

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

# Visual comparison
st.markdown("### Shortfall risk by decision")

risk_chart = comparison.set_index("Decision")[
    "Shortfall Risk (%)"
]
st.bar_chart(risk_chart)

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

st.caption(
    "Prototype estimates use illustrative assumptions. "
    "Shared random scenarios make the comparison fairer, "
    "but do not establish real-world financial risk. "
    "Risk categories are preliminary design thresholds."
)

