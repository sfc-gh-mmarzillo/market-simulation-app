-- Monte Carlo what-if engine. Deterministic, vectorized NumPy: the LLM never computes these numbers.
CREATE TABLE IF NOT EXISTS WHATIF_DEMO.SIM.SCENARIO_RESULTS (
  SCENARIO_ID STRING, RUN_TS TIMESTAMP_LTZ, SCENARIO_NAME STRING, CUSTOMER_ID STRING, CUSTOMER_NAME STRING,
  SKU_GROUP STRING, PRICE_CHANGE_PCT FLOAT, COMP_MODE STRING, COMP_MOVE_PCT FLOAT, LEAD_SHOCK_PCT FLOAT,
  HORIZON_MONTHS INT, N_RUNS INT,
  REVENUE_P10 FLOAT, REVENUE_P50 FLOAT, REVENUE_P90 FLOAT,
  MARGIN_P10 FLOAT, MARGIN_P50 FLOAT, MARGIN_P90 FLOAT,
  VOLUME_P10 FLOAT, VOLUME_P50 FLOAT, VOLUME_P90 FLOAT,
  BASELINE_REVENUE_P50 FLOAT, BASELINE_MARGIN_P50 FLOAT, BASELINE_VOLUME_P50 FLOAT,
  MARGIN_DELTA_P10 FLOAT, MARGIN_DELTA_P50 FLOAT, MARGIN_DELTA_P90 FLOAT,
  PROB_MARGIN_BELOW_BASELINE FLOAT, PROB_SHARE_SHIFT FLOAT, PROB_VOLUME_LOSS_GT_10PCT FLOAT,
  RUNTIME_SEC FLOAT, INPUTS VARIANT);

CREATE TABLE IF NOT EXISTS WHATIF_DEMO.SIM.SCENARIO_FAN (
  SCENARIO_ID STRING, MONTH_NUM INT, MARGIN_P10 FLOAT, MARGIN_P25 FLOAT, MARGIN_P50 FLOAT,
  MARGIN_P75 FLOAT, MARGIN_P90 FLOAT, BASELINE_MARGIN_P50 FLOAT);

CREATE TABLE IF NOT EXISTS WHATIF_DEMO.SIM.SCENARIO_DRIVERS (
  SCENARIO_ID STRING, DRIVER STRING, VARIANCE_SHARE FLOAT,
  MARGIN_DELTA_WHEN_LOW FLOAT, MARGIN_DELTA_WHEN_HIGH FLOAT, LOW_LABEL STRING, HIGH_LABEL STRING);

CREATE OR REPLACE PROCEDURE WHATIF_DEMO.SIM.RUN_SCENARIO(
    SCENARIO_NAME STRING, CUSTOMER_ID STRING, SKU_GROUP STRING,
    PRICE_CHANGE_PCT FLOAT, COMP_MODE STRING, COMP_MOVE_PCT FLOAT, LEAD_SHOCK_PCT FLOAT,
    HORIZON_MONTHS INT, N_RUNS INT, SEED INT)
RETURNS VARIANT
LANGUAGE PYTHON
RUNTIME_VERSION = '3.11'
PACKAGES = ('snowflake-snowpark-python', 'numpy', 'pandas')
HANDLER = 'run'
COMMENT = 'Monte Carlo pricing / competitor / lead-cost simulation. COMP_MODE = MODEL (sample competitor response) or FIXED (use COMP_MOVE_PCT). Percent inputs are fractions, e.g. 0.08 = +8%.'
AS
$$
import json, time, uuid
import numpy as np
import pandas as pd

WEEKS_PER_MONTH = 52 / 12

def comp_response(cp, our_move, n, rng):
    """Sample competitor price move + lag (months) from the response model, nearest observed bucket."""
    if abs(our_move) < 0.01:
        p = np.array([0.10, 0.05, 0.85]); match_ratio, um, usd = 0.85, -0.02, 0.01
    else:
        row = cp.iloc[(cp.OUR_PRICE_MOVE - our_move).abs().argmin()]
        p = np.array([row.P_MATCH, row.P_UNDERCUT, row.P_HOLD]); p = p / p.sum()
        match_ratio, um, usd = row.MATCH_RATIO, row.UNDERCUT_MOVE_MEAN, max(row.UNDERCUT_MOVE_SD, 0.005)
    kind = rng.choice(3, size=n, p=p)
    move = np.where(kind == 0, our_move * match_ratio * rng.uniform(0.85, 1.15, n),
           np.where(kind == 1, rng.normal(um, usd, n), 0.0))
    lag_m = np.clip(np.ceil(rng.uniform(2, 12, n) / WEEKS_PER_MONTH), 1, None).astype(int)
    return move, lag_m

def simulate(e, x, el, ch, sku, dp, comp_move, comp_lag, lead_paths, demand_z, sw_u, sw_month, share_lost, H):
    """Return monthly (n, H) revenue, margin, volume for one SKU group."""
    months = np.arange(1, H + 1)
    season = np.array(json.loads(el.SEASON_IDX))
    cal = (np.arange(H) + 11) % 12          # horizon starts in December
    base_units = el.BASE_WEEKLY_UNITS * WEEKS_PER_MONTH * season[cal]
    cur_ratio = el.CURRENT_PRICE / el.CONTRACT_BASE_PRICE
    comp_ratio_now = el.CURRENT_COMP_PRICE / el.CONTRACT_BASE_PRICE
    price = el.CURRENT_PRICE * (1 + dp)
    comp_active = months[None, :] >= comp_lag[:, None]
    comp_ratio = comp_ratio_now * (1 + np.where(comp_active, comp_move[:, None], 0.0))
    own_ratio = cur_ratio * (1 + dp)
    # Share-shift (churn) probability from logistic model on price gap at review
    gap = own_ratio / (comp_ratio_now * (1 + comp_move)) - 1
    p_switch = 1 / (1 + np.exp(-(ch.CHURN_INTERCEPT + ch.CHURN_SLOPE * gap)))
    switched = sw_u < p_switch
    churn_mult = np.where(switched[:, None] & (months[None, :] >= sw_month[:, None]), 1 - share_lost[:, None], 1.0)
    vol = (base_units[None, :] * own_ratio ** e[:, None] * comp_ratio ** x[:, None]
           * churn_mult * np.exp(el.DEMAND_NOISE_SD * 0.5 * demand_z))
    unit_cost = sku.LEAD_KG_PER_UNIT * lead_paths / 1000 + sku.OTHER_UNIT_COST
    rev = vol * price
    return rev, rev - vol * unit_cost, vol, switched, share_lost * switched

def run(session, scenario_name, customer_id, sku_group, dp, comp_mode, comp_move_in, lead_shock,
        horizon, n, seed):
    t0 = time.time()
    H, n = int(horizon or 12), int(n or 10000)
    dp, lead_shock, comp_move_in = float(dp or 0), float(lead_shock or 0), float(comp_move_in or 0)
    comp_mode = (comp_mode or "MODEL").upper()
    rng = np.random.default_rng(seed)

    el_all = session.sql("SELECT * FROM ELASTICITY_PARAMS WHERE CUSTOMER_ID = ?", params=[customer_id]).to_pandas()
    if sku_group and sku_group.upper() != "ALL":
        el_all = el_all[el_all.SKU_GROUP == sku_group.upper()]
    if el_all.empty:
        return {"error": f"No model for customer {customer_id} / sku {sku_group}"}
    ch = session.sql("SELECT * FROM CHURN_PARAMS WHERE CUSTOMER_ID = ?", params=[customer_id]).to_pandas().iloc[0]
    cp = session.table("COMP_RESPONSE_PARAMS").to_pandas()
    lp = session.table("LEAD_PARAMS").to_pandas().iloc[0]
    skus = session.table("SKU_GROUPS").to_pandas().set_index("SKU_GROUP")
    cname = session.sql("SELECT CUSTOMER_NAME FROM CUSTOMERS WHERE CUSTOMER_ID = ?", params=[customer_id]).collect()[0][0]

    # --- Shared random drivers (common random numbers => scenario vs baseline is apples-to-apples) ---
    if comp_mode == "FIXED":
        comp_move = comp_move_in + rng.normal(0, 0.005, n)
        comp_lag = np.clip(np.ceil(rng.uniform(2, 12, n) / WEEKS_PER_MONTH), 1, None).astype(int)
    else:
        comp_move, comp_lag = comp_response(cp, dp, n, rng)
    dt = 1 / 12
    shocks = rng.standard_normal((n, H))
    log_path = np.cumsum((lp.ANNUAL_DRIFT - 0.5 * lp.ANNUAL_VOL ** 2) * dt + lp.ANNUAL_VOL * np.sqrt(dt) * shocks, axis=1)
    lead_paths = lp.SPOT_USD_PER_T * (1 + lead_shock) * np.exp(log_path)
    lead_paths_base = lp.SPOT_USD_PER_T * np.exp(log_path)
    demand_z = rng.standard_normal((n, H))
    sw_u = rng.uniform(size=n)
    sw_month = rng.integers(2, H + 1, n)
    share_lost = np.clip(rng.normal(ch.SHARE_LOST_MEAN, ch.SHARE_LOST_SD, n), 0.05, 0.6)
    z_e, z_x = rng.standard_normal(n), rng.standard_normal(n)

    R = np.zeros((n, H)); M = np.zeros((n, H)); V = np.zeros((n, H))
    Rb = np.zeros((n, H)); Mb = np.zeros((n, H)); Vb = np.zeros((n, H))
    e_avg = np.zeros(n); w = 0
    for _, el in el_all.iterrows():
        sku = skus.loc[el.SKU_GROUP]
        e = np.minimum(el.OWN_ELASTICITY + el.OWN_ELASTICITY_SE * z_e, -0.05)
        x = np.maximum(el.CROSS_ELASTICITY + el.CROSS_ELASTICITY_SE * z_x, 0.0)
        r, m, v, switched, lost = simulate(e, x, el, ch, sku, dp, comp_move, comp_lag, lead_paths,
                                           demand_z, sw_u, sw_month, share_lost, H)
        rb, mb, vb, _, _ = simulate(e, x, el, ch, sku, 0.0, np.zeros(n), comp_lag, lead_paths_base,
                                    demand_z, sw_u, sw_month, share_lost, H)
        R += r; M += m; V += v; Rb += rb; Mb += mb; Vb += vb
        e_avg += e * el.BASE_WEEKLY_UNITS; w += el.BASE_WEEKLY_UNITS
    e_avg /= w

    rev, mar, vol = R.sum(1), M.sum(1), V.sum(1)
    rev_b, mar_b, vol_b = Rb.sum(1), Mb.sum(1), Vb.sum(1)
    d_m = mar - mar_b
    q = lambda a: [float(np.percentile(a, k)) for k in (10, 50, 90)]
    sid = str(uuid.uuid4())

    # --- Driver attribution: variance share (squared rank correlation, normalized) + low/high decile tornado ---
    drivers = {
        "Price elasticity":       (e_avg, "more elastic", "less elastic"),
        "Competitor response":    (comp_move, "competitor cuts", "competitor raises"),
        "Share shift (churn)":    (-lost, "account shifts share", "no shift"),
        "LME lead cost":          (-lead_paths.mean(1), "lead spikes", "lead falls"),
        "Demand / weather noise": (demand_z.mean(1), "weak demand", "strong demand"),
    }
    rank = lambda a: pd.Series(a).rank().values
    rd = rank(d_m)
    rows, tot = [], 0.0
    for name, (s, lo_lbl, hi_lbl) in drivers.items():
        rho = 0.0 if np.std(s) < 1e-12 else float(np.corrcoef(rank(s), rd)[0, 1]) ** 2
        lo, hi = np.percentile(s, 10), np.percentile(s, 90)
        rows.append([sid, name, rho, float(d_m[s <= lo].mean()), float(d_m[s >= hi].mean()), lo_lbl, hi_lbl])
        tot += rho
    for r_ in rows:
        r_[2] = r_[2] / tot if tot > 0 else 0.0

    fan = [[sid, h + 1] + [float(np.percentile(M[:, h], k)) for k in (10, 25, 50, 75, 90)] + [float(np.median(Mb[:, h]))]
           for h in range(H)]
    inputs = dict(scenario_name=scenario_name, customer_id=customer_id, sku_group=sku_group, price_change_pct=dp,
                  comp_mode=comp_mode, comp_move_pct=comp_move_in, lead_shock_pct=lead_shock, horizon=H, n_runs=n, seed=seed)
    runtime = time.time() - t0
    res = dict(SCENARIO_ID=sid, SCENARIO_NAME=scenario_name, CUSTOMER_ID=customer_id, CUSTOMER_NAME=cname,
               SKU_GROUP=(sku_group or "ALL").upper(), PRICE_CHANGE_PCT=dp, COMP_MODE=comp_mode, COMP_MOVE_PCT=comp_move_in,
               LEAD_SHOCK_PCT=lead_shock, HORIZON_MONTHS=H, N_RUNS=n,
               REVENUE_P10=q(rev)[0], REVENUE_P50=q(rev)[1], REVENUE_P90=q(rev)[2],
               MARGIN_P10=q(mar)[0], MARGIN_P50=q(mar)[1], MARGIN_P90=q(mar)[2],
               VOLUME_P10=q(vol)[0], VOLUME_P50=q(vol)[1], VOLUME_P90=q(vol)[2],
               BASELINE_REVENUE_P50=float(np.median(rev_b)), BASELINE_MARGIN_P50=float(np.median(mar_b)),
               BASELINE_VOLUME_P50=float(np.median(vol_b)),
               MARGIN_DELTA_P10=q(d_m)[0], MARGIN_DELTA_P50=q(d_m)[1], MARGIN_DELTA_P90=q(d_m)[2],
               PROB_MARGIN_BELOW_BASELINE=float((d_m < 0).mean()), PROB_SHARE_SHIFT=float((lost > 0).mean()),
               PROB_VOLUME_LOSS_GT_10PCT=float((vol < 0.9 * vol_b).mean()), RUNTIME_SEC=runtime)

    cols = ",".join(res.keys())
    ph = ",".join(["?"] * len(res))
    session.sql(f"INSERT INTO SCENARIO_RESULTS (RUN_TS,{cols},INPUTS) SELECT CURRENT_TIMESTAMP(),{ph},PARSE_JSON(?)",
                params=list(res.values()) + [json.dumps(inputs)]).collect()
    session.write_pandas(pd.DataFrame(fan, columns=["SCENARIO_ID","MONTH_NUM","MARGIN_P10","MARGIN_P25","MARGIN_P50",
                                                     "MARGIN_P75","MARGIN_P90","BASELINE_MARGIN_P50"]),
                         "SCENARIO_FAN", quote_identifiers=False)
    session.write_pandas(pd.DataFrame(rows, columns=["SCENARIO_ID","DRIVER","VARIANCE_SHARE","MARGIN_DELTA_WHEN_LOW",
                                                      "MARGIN_DELTA_WHEN_HIGH","LOW_LABEL","HIGH_LABEL"]),
                         "SCENARIO_DRIVERS", quote_identifiers=False)
    res["SIM_SEC"] = runtime
    res["TOTAL_SEC"] = time.time() - t0
    return res
$$;

CALL WHATIF_DEMO.SIM.RUN_SCENARIO('Raise AutoZone 8%, model competitor', 'C01', 'ALL', 0.08, 'MODEL', 0, 0, 12, 10000, 7);
