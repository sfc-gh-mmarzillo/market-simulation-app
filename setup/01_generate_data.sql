-- Voltara Energy what-if demo: synthetic data generator (fictional battery manufacturer)
CREATE OR REPLACE PROCEDURE WHATIF_DEMO.SIM.GENERATE_DATA(SEED INT)
RETURNS STRING
LANGUAGE PYTHON
RUNTIME_VERSION = '3.11'
PACKAGES = ('snowflake-snowpark-python', 'numpy', 'pandas')
HANDLER = 'run'
AS
$$
import numpy as np
import pandas as pd

def run(session, seed):
    rng = np.random.default_rng(seed)

    # --- Customers (true elasticities are hidden; the models must recover them) ---
    customers = pd.DataFrame([
        # id, name, channel, base weekly units, true own elasticity, loyalty (lower = churns easier)
        ("C01", "AutoZone",          "Aftermarket Retail", 42000, -1.35, 0.55),
        ("C02", "Walmart",           "Mass Merchant",      55000, -1.85, 0.35),
        ("C03", "O'Reilly",          "Aftermarket Retail", 38000, -1.25, 0.60),
        ("C04", "Advance Auto",      "Aftermarket Retail", 30000, -1.40, 0.50),
        ("C05", "NAPA",              "Distributor",        22000, -0.95, 0.75),
        ("C06", "Costco",            "Club",               26000, -2.05, 0.30),
        ("C07", "Interstate Dealer Network", "Distributor", 18000, -0.85, 0.80),
        ("C08", "OEM Assembly (Auto Co.)",   "OEM",         48000, -0.55, 0.90),
    ], columns=["CUSTOMER_ID","CUSTOMER_NAME","CHANNEL","BASE_WEEKLY_UNITS","TRUE_ELASTICITY","LOYALTY"])

    skus = pd.DataFrame([
        # group, list price $, lead kg per unit, non-lead unit cost $, volume mix
        ("FLOODED_STD", "Flooded Standard", 115.0, 10.5, 38.0, 0.45),
        ("EFB",         "Enhanced Flooded (EFB)", 150.0, 11.5, 48.0, 0.25),
        ("AGM",         "AGM Premium",       215.0, 12.5, 72.0, 0.22),
        ("HEAVY_DUTY",  "Heavy Duty / Commercial", 260.0, 22.0, 85.0, 0.08),
    ], columns=["SKU_GROUP","SKU_DESC","LIST_PRICE","LEAD_KG_PER_UNIT","OTHER_UNIT_COST","VOLUME_MIX"])

    # --- LME lead daily (GBM with mild mean reversion), 3 years ---
    days = pd.bdate_range("2023-11-13", "2026-11-13")
    p = [2150.0]
    for _ in range(1, len(days)):
        drift = 0.02 * (np.log(2100) - np.log(p[-1])) / 252
        p.append(p[-1] * np.exp(drift + 0.22 / np.sqrt(252) * rng.standard_normal()))
    lme = pd.DataFrame({"PRICE_DATE": days.date, "LEAD_USD_PER_T": np.round(p, 2)})
    lme_week = pd.Series(p, index=days).resample("W-MON").mean()

    # --- Weekly sales history: our price, competitor price, units ---
    weeks = pd.date_range("2023-11-13", "2026-11-09", freq="W-MON")
    rows = []
    for _, c in customers.iterrows():
        for _, s in skus.iterrows():
            cust_disc = rng.uniform(0.82, 0.95)
            base_price = s.LIST_PRICE * cust_disc
            cross = rng.uniform(0.5, 0.9) * abs(c.TRUE_ELASTICITY)
            our_p = base_price; comp_p = base_price * rng.uniform(0.95, 1.03)
            pending = []          # scheduled competitor reactions: (week_index, new_comp_price)
            shift_until, shift_mult = -1, 1.0
            for i, w in enumerate(weeks):
                # occasional price moves by us (gives the regression variation)
                if i > 4 and rng.random() < 0.07:
                    new_p = base_price * rng.uniform(0.9, 1.12)
                    move = new_p / our_p - 1
                    our_p = new_p
                    # competitor reacts to our move with a lag (same rules as COMPETITOR_RESPONSES)
                    if abs(move) >= 0.015:
                        u = rng.random()
                        p_match, p_under = (0.45 - 1.5 * move, 0.15 + 2.0 * move) if move > 0 else (0.55, 0.30)
                        lag = int(rng.integers(2, 13))
                        if u < p_match:
                            pending.append((i + lag, comp_p * (1 + move * rng.uniform(0.7, 1.0))))
                        elif u < p_match + p_under:
                            cut = abs(rng.normal(0.03, 0.015)) if move > 0 else abs(move) + abs(rng.normal(0.02, 0.01))
                            pending.append((i + lag, comp_p * (1 - cut)))
                        # share shift at less loyal accounts after an increase
                        gap = our_p / comp_p - 1
                        p_shift = 1 / (1 + np.exp(-(-2.6 + 22.0 * max(gap, 0) * (1 - c.LOYALTY) + 1.2 * (0.5 - c.LOYALTY))))
                        if move > 0.02 and rng.random() < p_shift:
                            shift_until, shift_mult = i + int(rng.integers(13, 27)), 1 - rng.uniform(0.15, 0.35)
                for wk, cp in [x for x in pending if x[0] == i]:
                    comp_p = cp
                pending = [x for x in pending if x[0] > i]
                if rng.random() < 0.045: comp_p = base_price * rng.uniform(0.88, 1.10)  # unprovoked competitor moves
                season = 1 + 0.22 * np.cos(2 * np.pi * (w.dayofyear - 15) / 365) + 0.10 * np.cos(2 * np.pi * (w.dayofyear - 200) / 365)  # winter + summer-heat peaks
                units = (c.BASE_WEEKLY_UNITS * s.VOLUME_MIX * season
                         * (our_p / base_price) ** c.TRUE_ELASTICITY
                         * (comp_p / base_price) ** cross
                         * (shift_mult if i <= shift_until else 1.0)
                         * np.exp(0.06 * rng.standard_normal()))
                lead = float(lme_week.asof(w)) if w >= lme_week.index[0] else p[0]
                rows.append((c.CUSTOMER_ID, s.SKU_GROUP, w.date(), round(our_p, 2), round(comp_p, 2),
                             int(units), round(lead, 2), round(base_price, 2)))
    sales = pd.DataFrame(rows, columns=["CUSTOMER_ID","SKU_GROUP","WEEK_START","OUR_PRICE","COMP_PRICE",
                                        "UNITS","LEAD_USD_PER_T","CONTRACT_BASE_PRICE"])

    # --- Quarterly account reviews: did the customer shift share to a competitor? (for churn model) ---
    ev = []
    for _, c in customers.iterrows():
        for q in range(60):  # synthetic multi-year, multi-category review history
            gap = rng.normal(0.0, 0.07)  # our price vs competitor (+ = we are pricier)
            logit = -2.6 + 22.0 * gap * (1 - c.LOYALTY) + 1.2 * (0.5 - c.LOYALTY)
            switched = int(rng.random() < 1 / (1 + np.exp(-logit)))
            ev.append((c.CUSTOMER_ID, q, round(gap, 4), switched,
                       round(rng.uniform(0.15, 0.35), 3) if switched else 0.0))
    events = pd.DataFrame(ev, columns=["CUSTOMER_ID","REVIEW_SEQ","PRICE_GAP_PCT","SHARE_SHIFTED","SHARE_LOST"])

    # --- Competitor price response history (feeds rule-based response model) ---
    cr = []
    for _ in range(400):
        our_move = rng.choice([-0.06,-0.04,-0.02,0.02,0.04,0.06,0.08,0.10])
        u = rng.random()
        if our_move > 0:
            p_match = 0.45 - 1.5 * our_move; p_under = 0.15 + 2.0 * our_move
        else:
            p_match = 0.55; p_under = 0.30
        resp = "MATCH" if u < p_match else ("UNDERCUT" if u < p_match + p_under else "HOLD")
        comp_move = {"MATCH": our_move * rng.uniform(0.7, 1.0),
                     "UNDERCUT": -abs(rng.normal(0.03, 0.015)) if our_move > 0 else our_move - abs(rng.normal(0.02, 0.01)),
                     "HOLD": 0.0}[resp]
        cr.append((float(our_move), resp, round(float(comp_move), 4), int(rng.integers(2, 12))))
    comp = pd.DataFrame(cr, columns=["OUR_PRICE_MOVE","RESPONSE","COMP_PRICE_MOVE","LAG_WEEKS"])

    for name, df in [("CUSTOMERS", customers.drop(columns=["TRUE_ELASTICITY","LOYALTY"])),
                     ("SKU_GROUPS", skus), ("LME_LEAD_PRICES", lme), ("SALES_HISTORY", sales),
                     ("ACCOUNT_REVIEWS", events), ("COMPETITOR_RESPONSES", comp)]:
        session.write_pandas(df, name, auto_create_table=True, overwrite=True, quote_identifiers=False)
    return f"sales rows={len(sales)}, lme days={len(lme)}, reviews={len(events)}, comp events={len(comp)}"
$$;

CALL WHATIF_DEMO.SIM.GENERATE_DATA(42);
