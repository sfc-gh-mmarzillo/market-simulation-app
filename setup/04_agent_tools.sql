-- Phase 2: agent-facing tools. Thin, deterministic wrappers so the LLM never maps IDs, converts % or does math.
DROP PROCEDURE IF EXISTS WHATIF_DEMO.SIM.SIM_TOOL(STRING, STRING, FLOAT, FLOAT, FLOAT, STRING);
CREATE OR REPLACE PROCEDURE WHATIF_DEMO.SIM.SIM_TOOL(
    CUSTOMER_NAME STRING, PRODUCT_LINE STRING, PRICE_CHANGE_PCT FLOAT,
    LEAD_SHOCK_PCT FLOAT DEFAULT 0, COMPETITOR_MOVE_PCT FLOAT DEFAULT NULL, SCENARIO_NAME STRING DEFAULT NULL,
    HORIZON_MONTHS INT DEFAULT 12, DISABLE_UNCERTAINTY STRING DEFAULT NULL)
RETURNS VARIANT
LANGUAGE PYTHON
RUNTIME_VERSION = '3.11'
PACKAGES = ('snowflake-snowpark-python')
HANDLER = 'run'
COMMENT = 'Agent tool: run a 10,000-path Monte Carlo pricing scenario. Percent inputs are whole numbers (8 = +8%). COMPETITOR_MOVE_PCT NULL = sample competitor response from the model. DISABLE_UNCERTAINTY = comma list of elasticity,competitor,churn,lead,demand.'
AS
$$
import json

PRODUCT_LINES = {"ALL": "ALL", "FLOODED": "FLOODED_STD", "FLOODED_STD": "FLOODED_STD", "STANDARD": "FLOODED_STD",
                 "EFB": "EFB", "AGM": "AGM", "HEAVY_DUTY": "HEAVY_DUTY", "HEAVY DUTY": "HEAVY_DUTY", "COMMERCIAL": "HEAVY_DUTY"}

def m(v):
    return f"${v / 1e6:,.1f}M"

SOURCES = ["elasticity", "competitor", "churn", "lead", "demand"]

def run(session, customer_name, product_line, price_change, lead_shock, comp_move, scenario_name, horizon, disable):
    rows = session.sql("SELECT CUSTOMER_ID, CUSTOMER_NAME FROM CUSTOMERS WHERE CUSTOMER_NAME ILIKE ? ORDER BY CUSTOMER_ID",
                       params=[f"%{(customer_name or '').strip()}%"]).collect()
    if not rows:
        names = [r[0] for r in session.sql("SELECT CUSTOMER_NAME FROM CUSTOMERS ORDER BY 1").collect()]
        return {"error": f"Unknown customer '{customer_name}'. Valid customers: {names}"}
    cid, cname = rows[0][0], rows[0][1]
    sku = PRODUCT_LINES.get((product_line or "ALL").strip().upper().replace("-", "_"), "ALL")
    dp = float(price_change or 0) / 100
    lead = float(lead_shock or 0) / 100
    mode = "MODEL" if comp_move is None else "FIXED"
    cm = 0.0 if comp_move is None else float(comp_move) / 100
    name = scenario_name or f"[agent] {cname} {dp:+.0%} | comp {'model' if mode == 'MODEL' else f'{cm:+.0%}'} | lead {lead:+.0%}"

    off = [x.strip().lower() for x in (disable or "").split(",") if x.strip().lower() in SOURCES]
    cfg = {"uncertainty": {x: False for x in off}}
    H = int(horizon or 12)
    raw = session.sql("CALL RUN_SCENARIO(?, ?, ?, ?, ?, ?, ?, ?, 10000, 7, PARSE_JSON(?))",
                      params=[name, cid, sku, dp, mode, cm, lead, H, json.dumps(cfg)]).collect()[0][0]
    r = json.loads(raw) if isinstance(raw, str) else raw
    if r.get("error"):
        return r
    drv = session.sql("""SELECT DRIVER, VARIANCE_SHARE, MARGIN_DELTA_WHEN_LOW, MARGIN_DELTA_WHEN_HIGH, LOW_LABEL, HIGH_LABEL
                         FROM SCENARIO_DRIVERS WHERE SCENARIO_ID = ? ORDER BY VARIANCE_SHARE DESC""",
                      params=[r["SCENARIO_ID"]]).collect()
    vol_chg = r["VOLUME_P50"] / r["BASELINE_VOLUME_P50"] - 1
    return {
        "scenario_id": r["SCENARIO_ID"],
        "inputs": {"customer": cname, "product_line": sku, "our_price_change": f"{dp:+.1%}",
                   "competitor": "sampled from response model" if mode == "MODEL" else f"fixed {cm:+.1%}",
                   "lme_lead_shock": f"{lead:+.1%}", "horizon_months": H, "monte_carlo_runs": r["N_RUNS"]},
        "model_configuration": r.get("CONFIG_SUMMARY"),
        "gross_margin_over_horizon": {"p10": m(r["MARGIN_P10"]), "p50": m(r["MARGIN_P50"]), "p90": m(r["MARGIN_P90"]),
                              "baseline_p50": m(r["BASELINE_MARGIN_P50"])},
        "margin_change_vs_baseline": {"p10": m(r["MARGIN_DELTA_P10"]), "p50": m(r["MARGIN_DELTA_P50"]),
                                      "p90": m(r["MARGIN_DELTA_P90"])},
        "revenue_over_horizon": {"p10": m(r["REVENUE_P10"]), "p50": m(r["REVENUE_P50"]), "p90": m(r["REVENUE_P90"])},
        "volume_units_p50": f"{r['VOLUME_P50']:,.0f} ({vol_chg:+.1%} vs baseline)",
        "risk": {"prob_margin_below_baseline": f"{r['PROB_MARGIN_BELOW_BASELINE']:.0%}",
                 "prob_volume_loss_over_10pct": f"{r['PROB_VOLUME_LOSS_GT_10PCT']:.0%}",
                 "prob_account_shifts_share": f"{r['PROB_SHARE_SHIFT']:.0%}"},
        "drivers_of_spread": [{"driver": d[0], "share_of_variance": f"{d[1]:.0%}",
                               f"margin_change_when_{d[4].replace(' ', '_')}": m(d[2]),
                               f"margin_change_when_{d[5].replace(' ', '_')}": m(d[3])} for d in drv],
        "simulation_seconds": round(r["RUNTIME_SEC"], 2),
        "note": "All figures computed by the Monte Carlo engine. Quote them as-is; do not recompute.",
    }
$$;

CREATE OR REPLACE PROCEDURE WHATIF_DEMO.SIM.EXPLAIN_SCENARIO(SCENARIO_ID STRING)
RETURNS VARIANT
LANGUAGE PYTHON
RUNTIME_VERSION = '3.11'
PACKAGES = ('snowflake-snowpark-python')
HANDLER = 'run'
COMMENT = 'Agent tool: root-cause a scenario. Returns driver attribution plus what-if sensitivities (competitor hold / match / undercut, lead +/-20%) rerun by the engine.'
AS
$$
import json

def m(v):
    return f"${v / 1e6:,.1f}M"

def run(session, scenario_id):
    rows = session.sql("SELECT INPUTS, MARGIN_P10, MARGIN_P50, MARGIN_DELTA_P50, CUSTOMER_NAME, MARGIN_P90 FROM SCENARIO_RESULTS WHERE SCENARIO_ID = ?",
                       params=[scenario_id]).collect()
    if not rows:
        return {"error": f"Scenario {scenario_id} not found"}
    inp = json.loads(rows[0][0])
    base = dict(p10=rows[0][1], p50=rows[0][2], d50=rows[0][3], p90=rows[0][5])
    drv = session.sql("""SELECT DRIVER, VARIANCE_SHARE, MARGIN_DELTA_WHEN_LOW, MARGIN_DELTA_WHEN_HIGH, LOW_LABEL, HIGH_LABEL
                         FROM SCENARIO_DRIVERS WHERE SCENARIO_ID = ? ORDER BY VARIANCE_SHARE DESC""", params=[scenario_id]).collect()
    dp = inp["price_change_pct"]
    variants = [
        ("Competitor holds price", "FIXED", 0.0, inp["lead_shock_pct"]),
        ("Competitor fully matches our move", "FIXED", dp, inp["lead_shock_pct"]),
        ("Competitor undercuts 5%", "FIXED", -0.05, inp["lead_shock_pct"]),
        ("Lead +20% on top", inp["comp_mode"], inp["comp_move_pct"], inp["lead_shock_pct"] + 0.20),
    ]
    sens = []
    for label, mode, cm, lead in variants:
        cfg = dict(inp.get("config") or {}); cfg["n_runs"] = 5000
        r = session.sql("CALL RUN_SCENARIO(?, ?, ?, ?, ?, ?, ?, ?, 5000, ?, PARSE_JSON(?))",
                        params=[f"[sensitivity] {rows[0][4]}: {label}", inp["customer_id"], inp["sku_group"],
                                dp, mode, cm, lead, int(inp.get("horizon", 12)), int(inp.get("seed", 7)),
                                json.dumps(cfg)]).collect()[0][0]
        r = json.loads(r) if isinstance(r, str) else r
        sens.append({"what_if": label, "scenario_id": r["SCENARIO_ID"], "margin_p10": m(r["MARGIN_P10"]), "margin_p50": m(r["MARGIN_P50"]), "margin_p90": m(r["MARGIN_P90"]),
                     "change_in_p50_vs_original": m(r["MARGIN_P50"] - base["p50"]),
                     "change_in_p10_vs_original": m(r["MARGIN_P10"] - base["p10"]),
                     "prob_margin_below_baseline": f"{r['PROB_MARGIN_BELOW_BASELINE']:.0%}"})
    return {
        "scenario_id": scenario_id, "model_configuration": inp.get("config_summary"), "original": {"margin_p10": m(base["p10"]), "margin_p50": m(base["p50"]), "margin_p90": m(base["p90"]),
                                                  "margin_change_vs_baseline_p50": m(base["d50"])},
        "driver_attribution": [{"driver": d[0], "share_of_variance": f"{d[1]:.0%}",
                                f"margin_change_when_{d[4].replace(' ', '_')}": m(d[2]),
                                f"margin_change_when_{d[5].replace(' ', '_')}": m(d[3])} for d in drv],
        "sensitivities": sens,
        "note": "All figures computed by the Monte Carlo engine. Quote them as-is; do not recompute.",
    }
$$;

CALL WHATIF_DEMO.SIM.SIM_TOOL(CUSTOMER_NAME => 'autozone', PRODUCT_LINE => 'all', PRICE_CHANGE_PCT => 8, LEAD_SHOCK_PCT => 0);
