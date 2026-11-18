-- Agent tools v2: price sweep (risk-constrained best price) and model card, both name-based with whole-number %.
CREATE OR REPLACE PROCEDURE WHATIF_DEMO.SIM.SWEEP_TOOL(
    CUSTOMER_NAME STRING, PRODUCT_LINE STRING, FROM_PCT FLOAT, TO_PCT FLOAT, STEP_PCT FLOAT DEFAULT 1,
    LEAD_SHOCK_PCT FLOAT DEFAULT 0, COMPETITOR_MOVE_PCT FLOAT DEFAULT NULL, MAX_RISK_PCT FLOAT DEFAULT 25)
RETURNS VARIANT
LANGUAGE PYTHON
RUNTIME_VERSION = '3.11'
PACKAGES = ('snowflake-snowpark-python')
HANDLER = 'run'
COMMENT = 'Agent tool: sweep price changes (whole %) and return the best price whose probability of margin below baseline is <= MAX_RISK_PCT.'
AS
$$
import json

SKUS = {"ALL": "ALL", "FLOODED": "FLOODED_STD", "FLOODED_STD": "FLOODED_STD", "STANDARD": "FLOODED_STD", "EFB": "EFB",
        "AGM": "AGM", "HEAVY_DUTY": "HEAVY_DUTY", "HEAVY DUTY": "HEAVY_DUTY", "COMMERCIAL": "HEAVY_DUTY"}

def m(v):
    return f"${v / 1e6:,.1f}M"

def run(session, customer_name, product_line, from_pct, to_pct, step_pct, lead_shock, comp_move, max_risk):
    rows = session.sql("SELECT CUSTOMER_ID, CUSTOMER_NAME FROM CUSTOMERS WHERE CUSTOMER_NAME ILIKE ? ORDER BY 1",
                       params=[f"%{(customer_name or '').strip()}%"]).collect()
    if not rows:
        return {"error": f"Unknown customer '{customer_name}'"}
    cid = rows[0][0]
    sku = SKUS.get((product_line or "ALL").strip().upper().replace("-", "_"), "ALL")
    mode = "MODEL" if comp_move is None else "FIXED"
    raw = session.sql("CALL RUN_SWEEP(?, ?, ?, ?, ?, ?, ?, ?, 5000, NULL, ?)",
                      params=[cid, sku, float(from_pct) / 100, float(to_pct) / 100, float(step_pct or 1) / 100, mode,
                              0.0 if comp_move is None else float(comp_move) / 100, float(lead_shock or 0) / 100,
                              float(max_risk or 25) / 100]).collect()[0][0]
    r = json.loads(raw) if isinstance(raw, str) else raw
    if r.get("error"):
        return r
    fmt = lambda p: None if not p else {
        "price_change": f"{p['price_change_pct']:+.1%}", "scenario_id": p["scenario_id"],
        "margin_change_vs_baseline_p50": m(p["margin_delta_p50"]),
        "margin_change_range_p10_p90": f"{m(p['margin_delta_p10'])} to {m(p['margin_delta_p90'])}",
        "prob_margin_below_baseline": f"{p['prob_margin_below_baseline']:.0%}",
        "prob_volume_loss_over_10pct": f"{p['prob_volume_loss_gt_10pct']:.0%}"}
    return {"sweep_id": r["sweep_id"], "customer": r["customer"], "product_line": r["product_line"],
            "model_configuration": r["config_summary"], "risk_limit": f"P(margin below baseline) <= {max_risk or 25:.0f}%",
            "best_price_within_risk_limit": fmt(r["best_within_risk_limit"]) or "no tested price meets the risk limit",
            "best_price_ignoring_risk": fmt(r["best_unconstrained"]),
            "all_points": [fmt(p) for p in r["points"]],
            "note": "Computed by the Monte Carlo engine (5,000 runs per point, common random numbers). Quote as-is."}
$$;

CREATE OR REPLACE PROCEDURE WHATIF_DEMO.SIM.DESCRIBE_MODEL(CUSTOMER_NAME STRING, PRODUCT_LINE STRING DEFAULT 'ALL')
RETURNS VARIANT
LANGUAGE PYTHON
RUNTIME_VERSION = '3.11'
PACKAGES = ('snowflake-snowpark-python')
HANDLER = 'run'
COMMENT = 'Agent tool: model card for a customer / product line — fitted parameters with uncertainty, what the simulation samples vs holds fixed, defaults, and backtest on past price moves.'
AS
$$
import json

SKUS = {"ALL": "ALL", "FLOODED": "FLOODED_STD", "FLOODED_STD": "FLOODED_STD", "STANDARD": "FLOODED_STD", "EFB": "EFB",
        "AGM": "AGM", "HEAVY_DUTY": "HEAVY_DUTY", "HEAVY DUTY": "HEAVY_DUTY", "COMMERCIAL": "HEAVY_DUTY"}

def run(session, customer_name, product_line):
    rows = session.sql("SELECT CUSTOMER_ID FROM CUSTOMERS WHERE CUSTOMER_NAME ILIKE ? ORDER BY 1",
                       params=[f"%{(customer_name or '').strip()}%"]).collect()
    if not rows:
        return {"error": f"Unknown customer '{customer_name}'"}
    sku = SKUS.get((product_line or "ALL").strip().upper().replace("-", "_"), "ALL")
    raw = session.call("DESCRIBE_MODEL_BY_ID", rows[0][0], sku)
    return json.loads(raw) if isinstance(raw, str) else raw
$$;

CALL WHATIF_DEMO.SIM.SWEEP_TOOL('AutoZone', 'ALL', 0, 10, 2);
