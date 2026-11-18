-- Historical analogs: every past price move, what actually happened, and what the fitted model would have predicted.
CREATE OR REPLACE PROCEDURE WHATIF_DEMO.SIM.BUILD_PRICE_EVENTS(MIN_MOVE FLOAT DEFAULT 0.02, PRE_WEEKS INT DEFAULT 8, POST_WEEKS INT DEFAULT 12)
RETURNS STRING
LANGUAGE PYTHON
RUNTIME_VERSION = '3.11'
PACKAGES = ('snowflake-snowpark-python', 'numpy', 'pandas')
HANDLER = 'run'
COMMENT = 'Detects past price moves in SALES_HISTORY and records realized outcomes + model backtest into PRICE_EVENTS.'
AS
$$
import json
import numpy as np
import pandas as pd

def run(session, min_move, pre_w, post_w):
    sales = session.table("SALES_HISTORY").to_pandas()
    sales["WEEK_START"] = pd.to_datetime(sales["WEEK_START"])
    el = session.table("ELASTICITY_PARAMS").to_pandas().set_index(["CUSTOMER_ID", "SKU_GROUP"])
    cust = session.table("CUSTOMERS").to_pandas().set_index("CUSTOMER_ID")
    lead_median = float(sales["LEAD_USD_PER_T"].median())
    out = []
    for (cid, sku), g in sales.groupby(["CUSTOMER_ID", "SKU_GROUP"]):
        g = g.sort_values("WEEK_START").reset_index(drop=True)
        p = el.loc[(cid, sku)]
        season = np.array(json.loads(p.SEASON_IDX))
        adj = g.UNITS / season[g.WEEK_START.dt.month.values - 1]       # seasonally adjusted units
        chg = g.OUR_PRICE.pct_change()
        for i in np.where(chg.abs() >= min_move)[0]:
            if i < pre_w or i + post_w > len(g):
                continue
            pre, post = slice(i - pre_w, i), slice(i, i + post_w)
            # skip if we moved price again inside the post window (outcome would be contaminated)
            if (chg.iloc[i + 1:i + post_w].abs() >= min_move).any():
                continue
            move = float(chg.iloc[i])
            p_pre, p_post = g.OUR_PRICE.iloc[i - 1], g.OUR_PRICE.iloc[i]
            c_pre, c_post = g.COMP_PRICE.iloc[pre].mean(), g.COMP_PRICE.iloc[post].mean()
            comp_end = g.COMP_PRICE.iloc[i + post_w - 1] / g.COMP_PRICE.iloc[i - 1] - 1
            comp_series = g.COMP_PRICE.iloc[i:i + post_w].values
            moved = np.where(np.abs(comp_series / g.COMP_PRICE.iloc[i - 1] - 1) >= 0.01)[0]
            if move > 0:
                resp = "MATCH" if comp_end >= 0.5 * move else ("UNDERCUT" if comp_end <= -0.01 else "HOLD")
            else:
                resp = "MATCH" if comp_end <= 0.5 * move else ("RAISE" if comp_end >= 0.01 else "HOLD")
            actual = float(adj.iloc[post].mean() / adj.iloc[pre].mean() - 1)
            pred_given_comp = float((p_post / p_pre) ** p.OWN_ELASTICITY * (c_post / c_pre) ** max(p.CROSS_ELASTICITY, 0) - 1)
            pred_own_only = float((p_post / p_pre) ** p.OWN_ELASTICITY - 1)
            lead = float(g.LEAD_USD_PER_T.iloc[i])
            out.append(dict(EVENT_ID=f"{cid}-{sku}-{g.WEEK_START.iloc[i].date()}", CUSTOMER_ID=cid,
                            CUSTOMER_NAME=cust.loc[cid, "CUSTOMER_NAME"], CHANNEL=cust.loc[cid, "CHANNEL"], SKU_GROUP=sku,
                            EVENT_WEEK=g.WEEK_START.iloc[i].date(), EVENT_MONTH=int(g.WEEK_START.iloc[i].month),
                            OUR_PRICE_MOVE=move, PRICE_BEFORE=float(p_pre), PRICE_AFTER=float(p_post),
                            COMP_PRICE_CHANGE=float(comp_end), COMP_RESPONSE=resp,
                            WEEKS_TO_COMP_RESPONSE=int(moved[0]) if len(moved) else None,
                            LEAD_USD_PER_T=lead, LEAD_VS_MEDIAN=lead / lead_median - 1,
                            ACTUAL_VOLUME_CHANGE=actual, PREDICTED_VOLUME_CHANGE=pred_given_comp,
                            PREDICTED_OWN_PRICE_ONLY=pred_own_only, PREDICTION_ERROR=actual - pred_given_comp,
                            ACTUAL_REVENUE_CHANGE=float((1 + actual) * (p_post / p_pre) - 1),
                            UNITS_PRE_WEEKLY=float(g.UNITS.iloc[pre].mean()), UNITS_POST_WEEKLY=float(g.UNITS.iloc[post].mean())))
    ev = pd.DataFrame(out)
    ev["EVENT_WEEK"] = pd.to_datetime(ev["EVENT_WEEK"]).dt.date
    session.write_pandas(ev, "PRICE_EVENTS", auto_create_table=True, overwrite=True, quote_identifiers=False)
    inc = ev[ev.OUR_PRICE_MOVE > 0]
    return (f"events={len(ev)} (increases={len(inc)}); competitor response to increases: "
            f"{inc.COMP_RESPONSE.value_counts(normalize=True).round(2).to_dict()}; "
            f"model MAE on volume change={ev.PREDICTION_ERROR.abs().mean():.3f}")
$$;

CALL WHATIF_DEMO.SIM.BUILD_PRICE_EVENTS();
