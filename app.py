import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. PAGE SETUP
# ---------------------------------------------------------
st.set_page_config(page_title="FinSight AI", page_icon="💰", layout="wide")
st.title("💰 FinSight AI")
st.caption("Risk-aware financial planning for university students in Bangladesh.")

# ---------------------------------------------------------
# 2. SETTINGS
# ---------------------------------------------------------
WEEKS = 8
FEE_WEEK = 6
SIMULATIONS = 10000
WEEKS_PER_MONTH = 4  # 2 months = 8 weeks, so the model stays exact

# ---------------------------------------------------------
# 3. INPUTS
# ---------------------------------------------------------
st.sidebar.header("Your Financial Information")
balance = st.sidebar.number_input("Current balance (BDT)", 0, value=12000, step=500)
income_min = st.sidebar.number_input("Minimum monthly income (BDT)", 0, value=8000, step=500)
income_max = st.sidebar.number_input("Maximum monthly income (BDT)", 0, value=14000, step=500)
expenses = st.sidebar.number_input("Monthly essential expenses (BDT)", 0, value=7000, step=500)
semester_fee = st.sidebar.number_input("Semester fee (BDT)", 0, value=5000, step=1000)
minimum_reserve = st.sidebar.number_input("Minimum safety reserve (BDT)", 0, value=2000, step=500)

st.sidebar.header("Planned Purchase")
item_name = st.sidebar.text_input("What do you want to buy?", value="Phone")
item_cost = st.sidebar.number_input(f"{item_name} cost (BDT)", 0, value=20000, step=500)
later_week = st.sidebar.slider("If buying later, which week?", 2, WEEKS, 4)

# Scenario names (they use the item name and chosen week)
NO_BUY = "Do not buy"
BUY_NOW = "Buy now (Week 1)"
BUY_LATER = f"Buy later (Week {later_week})"
PURCHASE_WEEKS = {NO_BUY: None, BUY_NOW: 1, BUY_LATER: later_week}

purchase_choice = st.sidebar.selectbox(
    f"When are you considering buying the {item_name.lower()}?",
    list(PURCHASE_WEEKS.keys()),
)

if income_min > income_max:
    st.error("Maximum monthly income must be at least minimum monthly income.")
    st.stop()

# ---------------------------------------------------------
# 4. SIMULATION (cached)
# ---------------------------------------------------------
@st.cache_data
def simulate(balance, income_min, income_max, expenses, semester_fee,
             item_cost, minimum_reserve, later_week):
    purchase_weeks = {
        "Do not buy": None,
        "Buy now (Week 1)": 1,
        f"Buy later (Week {later_week})": later_week,
    }

    rng = np.random.default_rng(42)
    weekly_expenses = expenses / WEEKS_PER_MONTH

    # One income level per month, repeated for each week of that month
    monthly = rng.uniform(income_min, income_max, size=(SIMULATIONS, 2))
    weekly_income = np.repeat(monthly, WEEKS_PER_MONTH, axis=1) / WEEKS_PER_MONTH

    # Unexpected expenses: 15% chance per week, 500-2500 BDT
    shocks = np.where(
        rng.random((SIMULATIONS, WEEKS)) < 0.15,
        rng.uniform(500, 2500, size=(SIMULATIONS, WEEKS)),
        0,
    )

    results = {}
    for name, buy_week in purchase_weeks.items():
        bal = np.full(SIMULATIONS, float(balance))
        history = []

        for week in range(1, WEEKS + 1):
            bal += weekly_income[:, week - 1] - weekly_expenses - shocks[:, week - 1]
            if week == FEE_WEEK:
                bal -= semester_fee
            if week == buy_week:
                bal -= item_cost
            history.append(bal.copy())

        history = np.array(history).T  # shape: (simulations, weeks)
        final = history[:, -1]

        shortfall = np.mean(
            (balance < minimum_reserve) | np.any(history < minimum_reserve, axis=1)
        )

        if buy_week is None:
            affordability = np.nan
        else:
            end_week = max(buy_week, FEE_WEEK)
            affordability = np.mean(
                np.all(history[:, buy_week - 1:end_week] >= minimum_reserve, axis=1)
            )

        results[name] = {
            "history": history,
            "final": final,
            "shortfall": shortfall,
            "avg_final": final.mean(),
            "p05_final": np.percentile(final, 5),
            "fee_reserve": np.mean(history[:, FEE_WEEK - 1] >= minimum_reserve),
            "affordability": affordability,
        }
    return results


results = simulate(balance, income_min, income_max, expenses,
                   semester_fee, item_cost, minimum_reserve, later_week)

# ---------------------------------------------------------
# 5. COMPARISON TABLE
# ---------------------------------------------------------
def risk_label(r):
    return "High" if r >= 0.30 else "Moderate" if r >= 0.10 else "Lower"

rows = []
for name, r in results.items():
    rows.append({
        "Decision": name,
        "Shortfall Risk (%)": round(r["shortfall"] * 100, 2),
        "Average Final Balance (BDT)": round(r["avg_final"]),
        "5th Percentile Balance (BDT)": round(r["p05_final"]),
        "Reserve After Fee (%)": round(r["fee_reserve"] * 100, 2),
        "Purchase Affordability (%)": (
            np.nan if np.isnan(r["affordability"]) else round(r["affordability"] * 100, 2)
        ),
        "Risk Category": risk_label(r["shortfall"]),
    })
comparison = pd.DataFrame(rows)

st.markdown(f"## Compare Your Decisions: {item_name}")
st.dataframe(comparison, hide_index=True, width="stretch")

# ---------------------------------------------------------
# 6. VALIDATION
# ---------------------------------------------------------
no_buy_avg = results[NO_BUY]["avg_final"]
now_diff = no_buy_avg - results[BUY_NOW]["avg_final"]
later_diff = no_buy_avg - results[BUY_LATER]["avg_final"]

with st.expander("Model validation"):
    st.write("Expected cost difference:", f"৳{item_cost:,.0f}")
    st.write("Buy-now difference:", f"৳{now_diff:,.2f}")
    st.write("Buy-later difference:", f"৳{later_diff:,.2f}")
    if np.isclose(now_diff, item_cost) and np.isclose(later_diff, item_cost):
        st.success("Purchase-cost validation passed.")
    else:
        st.error("Validation failed: check the simulation logic.")

# ---------------------------------------------------------
# 7. SELECTED SCENARIO
# ---------------------------------------------------------
selected = results[purchase_choice]
st.markdown("## Your Selected Scenario")
c1, c2, c3 = st.columns(3)
c1.metric("Shortfall Probability", f"{selected['shortfall']:.1%}")
c2.metric("Average Final Balance", f"৳{selected['avg_final']:,.0f}")
c3.metric("5th Percentile Balance", f"৳{selected['p05_final']:,.0f}")

if selected["shortfall"] >= 0.30:
    st.error("High modeled risk. Review your spending plan.")
elif selected["shortfall"] >= 0.10:
    st.warning("Moderate modeled risk. Consider protecting your reserve.")
else:
    st.success("Lower modeled risk under the current assumptions.")

# ---------------------------------------------------------
# 8. RECOMMENDATION
# ---------------------------------------------------------
AFFORDABILITY_THRESHOLD = 0.90
MAX_ACCEPTABLE_RISK = 0.10

def purchase_is_affordable(r):
    return (
        not np.isnan(r["affordability"])
        and r["affordability"] >= AFFORDABILITY_THRESHOLD
        and r["shortfall"] < MAX_ACCEPTABLE_RISK
    )

no_buy = results[NO_BUY]
buy_now = results[BUY_NOW]
buy_later = results[BUY_LATER]

st.markdown("## FinSight Recommendation")

if no_buy["shortfall"] >= MAX_ACCEPTABLE_RISK:
    recommendation, rec_risk = f"Do not buy the {item_name.lower()} yet", no_buy["shortfall"]
    explanation = (
        f"Even without the {item_name.lower()}, your budget shows a shortfall risk of "
        f"{no_buy['shortfall']:.1%}. Improve your baseline (raise income, cut expenses, "
        "or build your reserve) before any purchase."
    )
elif purchase_is_affordable(buy_now):
    recommendation, rec_risk = BUY_NOW, buy_now["shortfall"]
    explanation = (
        f"Buying the {item_name.lower()} now meets the prototype thresholds. "
        "Confirm your real fee deadline and expenses first."
    )
elif purchase_is_affordable(buy_later):
    recommendation, rec_risk = BUY_LATER, buy_later["shortfall"]
    explanation = (
        f"Buying now does not meet the thresholds, but waiting until Week {later_week} does. "
        "Recheck your balance and obligations before purchasing."
    )
else:
    recommendation, rec_risk = f"Do not buy the {item_name.lower()} yet", no_buy["shortfall"]
    explanation = (
        "Your baseline is stable, but neither purchase timing meets the thresholds. "
        "Consider saving more or choosing a cheaper option."
    )

c1, c2 = st.columns(2)
c1.metric("Recommended Option", recommendation)
c2.metric("Estimated Shortfall Risk", f"{rec_risk:.1%}")
st.write(explanation)
st.caption(
    "Prototype thresholds: at least 90% modeled affordability and less than 10% "
    "shortfall risk. These are design choices, not validated financial advice."
)

# ---------------------------------------------------------
# 9. CHARTS
# ---------------------------------------------------------
st.markdown("## Shortfall Risk by Decision")
st.bar_chart(comparison.set_index("Decision")["Shortfall Risk (%)"])

st.markdown("## 8-Week Cash-Flow Trajectory")
h = selected["history"]
cashflow_df = pd.DataFrame(
    {
        "Median balance (BDT)": np.median(h, axis=0),
        "5th percentile (BDT)": np.percentile(h, 5, axis=0),
        "95th percentile (BDT)": np.percentile(h, 95, axis=0),
    },
    index=[f"Week {i}" for i in range(1, WEEKS + 1)],
)
st.line_chart(cashflow_df)

st.markdown("## Possible Balances After 8 Weeks")
fig, ax = plt.subplots(figsize=(10, 4))
for name, r in results.items():
    ax.hist(r["final"], bins=40, alpha=0.35, label=name)
ax.axvline(minimum_reserve, linestyle="--", color="black", label="Minimum reserve")
ax.set_xlabel("Final balance (BDT)")
ax.set_ylabel("Number of simulations")
ax.legend()
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

# ---------------------------------------------------------
# 10. DISCLAIMER
# ---------------------------------------------------------
st.caption(
    "FinSight AI is a prototype, not a guarantee of financial safety. The model draws "
    "one income level per month (uniform within your range), a 15% weekly chance of an "
    "unexpected expense of ৳500-2,500, and a semester fee in Week 6. Debt, inflation, "
    "and other obligations are not modeled."
)
