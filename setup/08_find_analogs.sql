-- Nearest historical precedents for a proposed move (deterministic matching; the LLM only reads the output).
CREATE OR REPLACE PROCEDURE WHATIF_DEMO.SIM.FIND_ANALOGS(
    CUSTOMER_NAME STRING, PRODUCT_LINE STRING, PRICE_CHANGE_PCT FLOAT,
    LEAD_SHOCK_PCT FLOAT DEFAULT 0, TOP_K INT DEFAULT 8)
RETURNS VARIANT
LANGUAGE PYTHON
RUNTIME_VERSION = '3.11'
PACKAGES = ('snowflake-snowpark-python', 'numpy', 'pandas')
HANDLER = 'run'
COMMENT = 'Agent tool: closest past price moves (same customer > same channel > other; size of move, product line, lead level, season) with realized volume, competitor response, and model backtest error. Percent inputs are whole numbers.'
AS
$$
import datetime
import numpy as np
import pandas as pd

SKUS = {"ALL": None, "FLOODED": "FLOODED_STD", "FLOODED_STD": "FLOODED_STD", "STANDARD": "FLOODED_STD", "EFB": "EFB",
        "AGM": "AGM", "HEAVY_DUTY": "HEAVY_DUTY", "HEAVY DUTY": "HEAVY_DUTY", "COMMERCIAL": "HEAVY_DUTY"}

def pct(v):
    return None if v is None or pd.isna(v) else f"{v:+.1%}"

def run(session, customer_name, product_line, price_change, lead_shock, top_k):
    k = int(top_k or 8)
    rows = session.sql("SELECT CUSTOMER_ID, CUSTOMER_NAME, CHANNEL FROM CUSTOMERS WHERE CUSTOMER_NAME ILIKE ? ORDER BY 1",
                       params=[f"%{(customer_name or '').strip()}%"]).collect()
    if not rows:
        return {"error": f"Unknown customer '{customer_name}'"}
    cid, cname, channel = rows[0]
    sku = SKUS.get((product_line or "ALL").strip().upper().replace("-", "_"))
    move = float(price_change or 0) / 100
    spot = session.table("LEAD_PARAMS").collect()[0]["SPOT_USD_PER_T"]
    ev = session.table("PRICE_EVENTS").to_pandas()
    lead_now = spot * (1 + float(lead_shock or 0) / 100) / (ev.LEAD_USD_PER_T.median()) - 1
    ev = ev[np.sign(ev.OUR_PRICE_MOVE) == np.sign(move if move != 0 else 1)].copy()

    tier = np.where(ev.CUSTOMER_ID == cid, 0.0, np.where(ev.CHANNEL == channel, 0.6, 1.5))
    month_gap = np.minimum(abs(ev.EVENT_MONTH - datetime.date.today().month), 12 - abs(ev.EVENT_MONTH - datetime.date.today().month))
    ev["SIMILARITY_DISTANCE"] = (abs(ev.OUR_PRICE_MOVE - move) / 0.03 + tier
                                 + (0 if sku is None else np.where(ev.SKU_GROUP == sku, 0.0, 0.4))
                                 + abs(ev.LEAD_VS_MEDIAN - lead_now) / 0.25 + month_gap / 12)
    top = ev.nsmallest(k, "SIMILARITY_DISTANCE")
    if top.empty:
        return {"error": "No comparable historical price moves found"}

    resp_mix = top.COMP_RESPONSE.value_counts(normalize=True).round(2).to_dict()
    match_basis = ("same customer" if (top.CUSTOMER_ID == cid).all() else
                   f"{int((top.CUSTOMER_ID == cid).sum())} same customer, {int(((top.CUSTOMER_ID != cid) & (top.CHANNEL == channel)).sum())} same channel")
    return {
        "query": {"customer": cname, "channel": channel, "product_line": sku or "ALL", "proposed_price_change": pct(move),
                  "lead_vs_history_median": pct(lead_now)},
        "n_analogs": int(len(top)), "match_basis": match_basis,
        "realized_outcomes": {
            "median_volume_change_12wk": pct(top.ACTUAL_VOLUME_CHANGE.median()),
            "volume_change_range": f"{top.ACTUAL_VOLUME_CHANGE.min():+.1%} to {top.ACTUAL_VOLUME_CHANGE.max():+.1%}",
            "median_revenue_change_12wk": pct(top.ACTUAL_REVENUE_CHANGE.median()),
            "competitor_response_mix": {k_: f"{v:.0%}" for k_, v in resp_mix.items()},
            "median_weeks_to_competitor_response": None if top.WEEKS_TO_COMP_RESPONSE.dropna().empty
                                                    else float(top.WEEKS_TO_COMP_RESPONSE.dropna().median()),
        },
        "model_backtest_on_these_cases": {
            "median_predicted_volume_change": pct(top.PREDICTED_VOLUME_CHANGE.median()),
            "mean_absolute_error": f"{top.PREDICTION_ERROR.abs().mean():.1%} pts",
            "bias": f"model {'over' if top.PREDICTION_ERROR.mean() < 0 else 'under'}-predicts volume by {abs(top.PREDICTION_ERROR.mean()):.1%} pts on average",
        },
        "cases": [{"event_id": r.EVENT_ID, "customer": r.CUSTOMER_NAME, "product_line": r.SKU_GROUP, "week": str(r.EVENT_WEEK),
                   "our_price_move": pct(r.OUR_PRICE_MOVE), "competitor_response": r.COMP_RESPONSE,
                   "competitor_price_change": pct(r.COMP_PRICE_CHANGE), "actual_volume_change": pct(r.ACTUAL_VOLUME_CHANGE),
                   "model_predicted": pct(r.PREDICTED_VOLUME_CHANGE)} for r in top.itertuples()],
        "event_ids": list(top.EVENT_ID),
        "note": "Realized outcomes measured from SALES_HISTORY (12 weeks after vs 8 weeks before, seasonally adjusted). Quote as-is.",
    }
$$;

CALL WHATIF_DEMO.SIM.FIND_ANALOGS('AutoZone', 'ALL', 8, 0, 8);
