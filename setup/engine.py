"""Voltara what-if Monte Carlo engine (shared by RUN_SCENARIO, RUN_SWEEP, DESCRIBE_MODEL).

Deterministic NumPy: given inputs + config + seed the result is reproducible. The LLM never computes these numbers.
Percent inputs are fractions (0.08 = +8%).
"""
import copy
import datetime
import json
import time
import uuid

import numpy as np
import pandas as pd

WEEKS_PER_MONTH = 52 / 12
SOURCES = ["elasticity", "competitor", "churn", "lead", "demand"]
DEFAULT_CONFIG = {
    "n_runs": 10000,
    "horizon_months": 12,
    "seed": 7,
    "uncertainty": {s: True for s in SOURCES},
    "comp_lag_weeks": [2, 12],
    "share_lost_mean": None,          # None = fitted per customer
}
SOURCE_LABELS = {
    "elasticity": "Price elasticity uncertainty (sample own & cross elasticity from fitted mean ± SE)",
    "competitor": "Competitor response (sample match / undercut / hold and lag from response model)",
    "churn": "Share-shift events (logistic churn model on price gap; share lost ~ fitted distribution)",
    "lead": "LME lead price paths (GBM with fitted drift & volatility)",
    "demand": "Demand / weather noise (fitted residual s.d.)",
}


# ----------------------------------------------------------------------------------------------- config
def resolve_config(cfg, horizon=None, n_runs=None, seed=None):
    if isinstance(cfg, str):
        cfg = json.loads(cfg) if cfg.strip() else {}
    if not isinstance(cfg, dict):  # SQL NULL VARIANT arrives as sqlNullWrapper
        cfg = {}
    out = copy.deepcopy(DEFAULT_CONFIG)
    for k, v in (cfg or {}).items():
        if k == "uncertainty" and isinstance(v, dict):
            out["uncertainty"].update({s: bool(b) for s, b in v.items() if s in SOURCES})
        elif k in out and v is not None:
            out[k] = v
    if horizon:
        out["horizon_months"] = int(horizon)
    if n_runs:
        out["n_runs"] = int(n_runs)
    if seed is not None:
        out["seed"] = int(seed)
    out["horizon_months"] = int(min(max(out["horizon_months"], 3), 36))
    out["n_runs"] = int(min(max(out["n_runs"], 500), 100000))
    return out


def config_summary(cfg, inp=None):
    off = [s for s in SOURCES if not cfg["uncertainty"][s]]
    parts = [f"{cfg['n_runs']:,} runs", f"{cfg['horizon_months']} mo", f"seed {cfg['seed']}"]
    if inp:
        parts.append("competitor " + ("sampled" if inp.get("comp_mode") == "MODEL" else f"fixed {inp.get('comp_move_pct', 0):+.0%}"))
    parts.append("all uncertainty on" if not off else "OFF: " + ", ".join(off))
    return " · ".join(parts)


# ----------------------------------------------------------------------------------------------- context
def load_context(session, customer_id, sku_group):
    el = session.sql("SELECT * FROM ELASTICITY_PARAMS WHERE CUSTOMER_ID = ?", params=[customer_id]).to_pandas()
    if sku_group and str(sku_group).upper() != "ALL":
        el = el[el.SKU_GROUP == str(sku_group).upper()]
    if el.empty:
        raise ValueError(f"No model for customer {customer_id} / product line {sku_group}")
    ch = session.sql("SELECT * FROM CHURN_PARAMS WHERE CUSTOMER_ID = ?", params=[customer_id]).to_pandas().iloc[0]
    cust = session.sql("SELECT CUSTOMER_NAME, CHANNEL FROM CUSTOMERS WHERE CUSTOMER_ID = ?", params=[customer_id]).collect()[0]
    fit_ts = session.sql("""SELECT MAX(LAST_ALTERED) FROM INFORMATION_SCHEMA.TABLES
                            WHERE TABLE_SCHEMA = CURRENT_SCHEMA() AND TABLE_NAME = 'ELASTICITY_PARAMS'""").collect()[0][0]
    return dict(el=el, ch=ch, cp=session.table("COMP_RESPONSE_PARAMS").to_pandas(),
                lp=session.table("LEAD_PARAMS").to_pandas().iloc[0],
                skus=session.table("SKU_GROUPS").to_pandas().set_index("SKU_GROUP"),
                customer_name=cust[0], channel=cust[1], fit_ts=str(fit_ts))


def model_snapshot(ctx):
    el = ctx["el"]
    return {
        "model_fit_ts": ctx["fit_ts"],
        "elasticity": [{"sku": r.SKU_GROUP, "own": round(r.OWN_ELASTICITY, 3), "own_se": round(r.OWN_ELASTICITY_SE, 3),
                        "cross": round(r.CROSS_ELASTICITY, 3), "cross_se": round(r.CROSS_ELASTICITY_SE, 3),
                        "r2": round(r.R2, 3), "demand_noise_sd": round(r.DEMAND_NOISE_SD, 3)} for r in el.itertuples()],
        "churn": {"intercept": round(float(ctx["ch"].CHURN_INTERCEPT), 3), "slope": round(float(ctx["ch"].CHURN_SLOPE), 3),
                  "share_lost_mean": round(float(ctx["ch"].SHARE_LOST_MEAN), 3)},
        "lead": {"spot": round(float(ctx["lp"].SPOT_USD_PER_T), 1), "drift": round(float(ctx["lp"].ANNUAL_DRIFT), 4),
                 "vol": round(float(ctx["lp"].ANNUAL_VOL), 4)},
    }


# ----------------------------------------------------------------------------------------------- core math
def comp_response(cp, our_move, n, rng, stochastic):
    if abs(our_move) < 0.01:
        p = np.array([0.10, 0.05, 0.85]); match_ratio, um, usd = 0.85, -0.02, 0.01
    else:
        row = cp.iloc[(cp.OUR_PRICE_MOVE - our_move).abs().argmin()]
        p = np.array([row.P_MATCH, row.P_UNDERCUT, row.P_HOLD]); p = p / p.sum()
        match_ratio, um, usd = row.MATCH_RATIO, row.UNDERCUT_MOVE_MEAN, max(row.UNDERCUT_MOVE_SD, 0.005)
    if not stochastic:  # expected competitor move, no randomness
        return np.full(n, p[0] * our_move * match_ratio + p[1] * um)
    kind = rng.choice(3, size=n, p=p)
    return np.where(kind == 0, our_move * match_ratio * rng.uniform(0.85, 1.15, n),
                    np.where(kind == 1, rng.normal(um, usd, n), 0.0))


def _simulate_sku(e, x, el, ch, sku, dp, comp_move, comp_lag, lead_paths, demand_z, sw_u, sw_month, share_lost, cal, H):
    months = np.arange(1, H + 1)
    season = np.array(json.loads(el.SEASON_IDX))
    base_units = el.BASE_WEEKLY_UNITS * WEEKS_PER_MONTH * season[cal]
    cur_ratio = el.CURRENT_PRICE / el.CONTRACT_BASE_PRICE
    comp_ratio_now = el.CURRENT_COMP_PRICE / el.CONTRACT_BASE_PRICE
    price = el.CURRENT_PRICE * (1 + dp)
    comp_active = months[None, :] >= comp_lag[:, None]
    comp_ratio = comp_ratio_now * (1 + np.where(comp_active, comp_move[:, None], 0.0))
    own_ratio = cur_ratio * (1 + dp)
    gap = own_ratio / (comp_ratio_now * (1 + comp_move)) - 1
    p_switch = 1 / (1 + np.exp(-(ch.CHURN_INTERCEPT + ch.CHURN_SLOPE * gap)))
    switched = sw_u < p_switch
    churn_mult = np.where(switched[:, None] & (months[None, :] >= sw_month[:, None]), 1 - share_lost[:, None], 1.0)
    vol = (base_units[None, :] * own_ratio ** e[:, None] * comp_ratio ** x[:, None]
           * churn_mult * np.exp(el.DEMAND_NOISE_SD * 0.5 * demand_z))
    unit_cost = sku.LEAD_KG_PER_UNIT * lead_paths / 1000 + sku.OTHER_UNIT_COST
    rev = vol * price
    return rev, rev - vol * unit_cost, vol, share_lost * switched


def simulate(ctx, dp, comp_mode, comp_move_in, lead_shock, cfg):
    """Return a result dict + fan rows + driver rows (no persistence)."""
    t0 = time.time()
    H, n, unc = cfg["horizon_months"], cfg["n_runs"], cfg["uncertainty"]
    rng = np.random.default_rng(cfg["seed"])
    el_all, ch, cp, lp, skus = ctx["el"], ctx["ch"], ctx["cp"], ctx["lp"], ctx["skus"]
    lag_lo, lag_hi = cfg["comp_lag_weeks"]

    # Draw every random stream in a fixed order (common random numbers across scenarios & sweep points)
    u_comp = np.random.default_rng(cfg["seed"] + 1)
    if comp_mode == "FIXED":
        comp_move = comp_move_in + (u_comp.normal(0, 0.005, n) if unc["competitor"] else np.zeros(n))
    else:
        comp_move = comp_response(cp, dp, n, u_comp, unc["competitor"])
    lag_w = rng.uniform(lag_lo, lag_hi, n) if unc["competitor"] else np.full(n, (lag_lo + lag_hi) / 2)
    comp_lag = np.clip(np.ceil(lag_w / WEEKS_PER_MONTH), 1, None).astype(int)
    dt = 1 / 12
    shocks = rng.standard_normal((n, H))
    if unc["lead"]:
        log_path = np.cumsum((lp.ANNUAL_DRIFT - 0.5 * lp.ANNUAL_VOL ** 2) * dt + lp.ANNUAL_VOL * np.sqrt(dt) * shocks, axis=1)
    else:
        log_path = np.zeros((n, H))
    lead_paths = lp.SPOT_USD_PER_T * (1 + lead_shock) * np.exp(log_path)
    lead_paths_base = lp.SPOT_USD_PER_T * np.exp(log_path)
    demand_z = rng.standard_normal((n, H)) if unc["demand"] else np.zeros((n, H))
    sw_u = rng.uniform(size=n) if unc["churn"] else np.ones(n)  # ones => never switches
    sw_month = rng.integers(2, H + 1, n)
    lost_mean = cfg["share_lost_mean"] if cfg["share_lost_mean"] is not None else ch.SHARE_LOST_MEAN
    share_lost = np.clip(rng.normal(lost_mean, ch.SHARE_LOST_SD, n), 0.05, 0.6)
    z_e = rng.standard_normal(n) if unc["elasticity"] else np.zeros(n)
    z_x = rng.standard_normal(n) if unc["elasticity"] else np.zeros(n)

    start = datetime.date.today().month  # horizon starts next month
    cal = (np.arange(H) + start) % 12
    R = np.zeros((n, H)); M = np.zeros((n, H)); V = np.zeros((n, H))
    Rb = np.zeros((n, H)); Mb = np.zeros((n, H)); Vb = np.zeros((n, H))
    e_avg = np.zeros(n); w = 0.0; lost = np.zeros(n)
    for el in el_all.itertuples():
        sku = skus.loc[el.SKU_GROUP]
        e = np.minimum(el.OWN_ELASTICITY + el.OWN_ELASTICITY_SE * z_e, -0.05)
        x = np.maximum(el.CROSS_ELASTICITY + el.CROSS_ELASTICITY_SE * z_x, 0.0)
        r, m, v, lost = _simulate_sku(e, x, el, ch, sku, dp, comp_move, comp_lag, lead_paths, demand_z,
                                      sw_u, sw_month, share_lost, cal, H)
        rb, mb, vb, _ = _simulate_sku(e, x, el, ch, sku, 0.0, np.zeros(n), comp_lag, lead_paths_base, demand_z,
                                      sw_u, sw_month, share_lost, cal, H)
        R += r; M += m; V += v; Rb += rb; Mb += mb; Vb += vb
        e_avg += e * el.BASE_WEEKLY_UNITS; w += el.BASE_WEEKLY_UNITS
    e_avg /= w

    rev, mar, vol = R.sum(1), M.sum(1), V.sum(1)
    rev_b, mar_b, vol_b = Rb.sum(1), Mb.sum(1), Vb.sum(1)
    d_m = mar - mar_b
    q = lambda a: [float(np.percentile(a, k)) for k in (10, 50, 90)]
    sid = str(uuid.uuid4())

    drivers = {
        "Price elasticity": ("elasticity", e_avg, "more elastic", "less elastic"),
        "Competitor response": ("competitor", comp_move, "competitor cuts", "competitor raises"),
        "Share shift (churn)": ("churn", -lost, "account shifts share", "no shift"),
        "LME lead cost": ("lead", -lead_paths.mean(1), "lead spikes", "lead falls"),
        "Demand / weather noise": ("demand", demand_z.mean(1), "weak demand", "strong demand"),
    }
    rank = lambda a: pd.Series(a).rank().values
    rd = rank(d_m)
    drv, tot = [], 0.0
    for name, (src, s, lo_lbl, hi_lbl) in drivers.items():
        if np.std(s) < 1e-12 or np.std(d_m) < 1e-9:
            rho, lo_v, hi_v = 0.0, float(d_m.mean()), float(d_m.mean())
        else:
            rho = float(np.corrcoef(rank(s), rd)[0, 1]) ** 2
            lo, hi = np.percentile(s, 10), np.percentile(s, 90)
            lo_v, hi_v = float(d_m[s <= lo].mean()), float(d_m[s >= hi].mean())
        drv.append([sid, name, rho, lo_v, hi_v, lo_lbl, hi_lbl]); tot += rho
    for r_ in drv:
        r_[2] = r_[2] / tot if tot > 0 else 0.0
    fan = [[sid, h + 1] + [float(np.percentile(M[:, h], k)) for k in (10, 25, 50, 75, 90)] + [float(np.median(Mb[:, h]))]
           for h in range(H)]
    res = dict(SCENARIO_ID=sid, CUSTOMER_NAME=ctx["customer_name"],
               REVENUE_P10=q(rev)[0], REVENUE_P50=q(rev)[1], REVENUE_P90=q(rev)[2],
               MARGIN_P10=q(mar)[0], MARGIN_P50=q(mar)[1], MARGIN_P90=q(mar)[2],
               VOLUME_P10=q(vol)[0], VOLUME_P50=q(vol)[1], VOLUME_P90=q(vol)[2],
               BASELINE_REVENUE_P50=float(np.median(rev_b)), BASELINE_MARGIN_P50=float(np.median(mar_b)),
               BASELINE_VOLUME_P50=float(np.median(vol_b)),
               MARGIN_DELTA_P10=q(d_m)[0], MARGIN_DELTA_P50=q(d_m)[1], MARGIN_DELTA_P90=q(d_m)[2],
               PROB_MARGIN_BELOW_BASELINE=float((d_m < 0).mean()), PROB_SHARE_SHIFT=float((lost > 0).mean()),
               PROB_VOLUME_LOSS_GT_10PCT=float((vol < 0.9 * vol_b).mean()), RUNTIME_SEC=time.time() - t0)
    return res, fan, drv


# ----------------------------------------------------------------------------------------------- persistence
RESULT_COLS = ["SCENARIO_ID", "SCENARIO_NAME", "CUSTOMER_ID", "CUSTOMER_NAME", "SKU_GROUP", "PRICE_CHANGE_PCT", "COMP_MODE",
               "COMP_MOVE_PCT", "LEAD_SHOCK_PCT", "HORIZON_MONTHS", "N_RUNS", "REVENUE_P10", "REVENUE_P50", "REVENUE_P90",
               "MARGIN_P10", "MARGIN_P50", "MARGIN_P90", "VOLUME_P10", "VOLUME_P50", "VOLUME_P90",
               "BASELINE_REVENUE_P50", "BASELINE_MARGIN_P50", "BASELINE_VOLUME_P50",
               "MARGIN_DELTA_P10", "MARGIN_DELTA_P50", "MARGIN_DELTA_P90",
               "PROB_MARGIN_BELOW_BASELINE", "PROB_SHARE_SHIFT", "PROB_VOLUME_LOSS_GT_10PCT", "RUNTIME_SEC", "INPUTS_JSON"]


def persist(session, results, fans, drivers):
    """Batch-write scenario rows (one INSERT per table regardless of how many scenarios)."""
    df = pd.DataFrame([{c: r.get(c) for c in RESULT_COLS} for r in results])
    session.write_pandas(df, "SCENARIO_RESULTS_STAGE", auto_create_table=True, overwrite=True,
                         quote_identifiers=False, table_type="temporary")
    cols = ",".join(RESULT_COLS[:-1])
    session.sql(f"""INSERT INTO SCENARIO_RESULTS (RUN_TS, {cols}, INPUTS)
                    SELECT CURRENT_TIMESTAMP(), {cols}, PARSE_JSON(INPUTS_JSON) FROM SCENARIO_RESULTS_STAGE""").collect()
    session.write_pandas(pd.DataFrame([r for f in fans for r in f], columns=[
        "SCENARIO_ID", "MONTH_NUM", "MARGIN_P10", "MARGIN_P25", "MARGIN_P50", "MARGIN_P75", "MARGIN_P90", "BASELINE_MARGIN_P50"]),
        "SCENARIO_FAN", quote_identifiers=False)
    session.write_pandas(pd.DataFrame([r for d in drivers for r in d], columns=[
        "SCENARIO_ID", "DRIVER", "VARIANCE_SHARE", "MARGIN_DELTA_WHEN_LOW", "MARGIN_DELTA_WHEN_HIGH", "LOW_LABEL", "HIGH_LABEL"]),
        "SCENARIO_DRIVERS", quote_identifiers=False)


def _one(ctx, name, customer_id, sku, dp, mode, cm, lead, cfg, extra=None):
    res, fan, drv = simulate(ctx, dp, mode, cm, lead, cfg)
    inputs = dict(scenario_name=name, customer_id=customer_id, sku_group=(sku or "ALL").upper(), price_change_pct=dp,
                  comp_mode=mode, comp_move_pct=cm, lead_shock_pct=lead, horizon=cfg["horizon_months"],
                  n_runs=cfg["n_runs"], seed=cfg["seed"], config=cfg, config_summary=None,
                  model_snapshot=model_snapshot(ctx), **(extra or {}))
    inputs["config_summary"] = config_summary(cfg, inputs)
    res.update(SCENARIO_NAME=name, CUSTOMER_ID=customer_id, SKU_GROUP=(sku or "ALL").upper(), PRICE_CHANGE_PCT=dp,
               COMP_MODE=mode, COMP_MOVE_PCT=cm, LEAD_SHOCK_PCT=lead, HORIZON_MONTHS=cfg["horizon_months"],
               N_RUNS=cfg["n_runs"], INPUTS_JSON=json.dumps(inputs))
    return res, fan, drv, inputs


# ----------------------------------------------------------------------------------------------- handlers
def run_scenario(session, scenario_name, customer_id, sku_group, dp, comp_mode, comp_move_in, lead_shock,
                 horizon, n_runs, seed, config=None):
    t0 = time.time()
    cfg = resolve_config(config, horizon, n_runs, seed)
    mode = (comp_mode or "MODEL").upper()
    try:
        ctx = load_context(session, customer_id, sku_group)
    except ValueError as e:
        return {"error": str(e)}
    res, fan, drv, inputs = _one(ctx, scenario_name, customer_id, sku_group, float(dp or 0), mode,
                                 float(comp_move_in or 0), float(lead_shock or 0), cfg)
    persist(session, [res], [fan], [drv])
    out = {k: v for k, v in res.items() if k != "INPUTS_JSON"}
    out.update(CONFIG_SUMMARY=inputs["config_summary"], SIM_SEC=res["RUNTIME_SEC"], TOTAL_SEC=time.time() - t0)
    return out


def run_sweep(session, customer_id, sku_group, from_pct, to_pct, step_pct, comp_mode, comp_move_in, lead_shock,
              n_runs, config=None, max_prob_below_baseline=0.25):
    t0 = time.time()
    cfg = resolve_config(config, None, n_runs or 5000, None)
    mode = (comp_mode or "MODEL").upper()
    try:
        ctx = load_context(session, customer_id, sku_group)
    except ValueError as e:
        return {"error": str(e)}
    step = max(abs(float(step_pct or 0.01)), 0.0025)
    grid = np.round(np.arange(float(from_pct), float(to_pct) + step / 2, step), 4)[:25]
    sweep_id = str(uuid.uuid4())
    results, fans, drvs, points = [], [], [], []
    for dp in grid:
        name = f"[sweep] {ctx['customer_name']} {dp:+.1%} | comp {'model' if mode == 'MODEL' else f'{comp_move_in:+.0%}'} | lead {lead_shock or 0:+.0%}"
        res, fan, drv, _ = _one(ctx, name, customer_id, sku_group, float(dp), mode, float(comp_move_in or 0),
                                float(lead_shock or 0), cfg, extra={"sweep_id": sweep_id})
        results.append(res); fans.append(fan); drvs.append(drv)
        points.append(dict(scenario_id=res["SCENARIO_ID"], price_change_pct=float(dp),
                           margin_p10=res["MARGIN_P10"], margin_p50=res["MARGIN_P50"], margin_p90=res["MARGIN_P90"],
                           margin_delta_p50=res["MARGIN_DELTA_P50"], margin_delta_p10=res["MARGIN_DELTA_P10"],
                           margin_delta_p90=res["MARGIN_DELTA_P90"],
                           prob_margin_below_baseline=res["PROB_MARGIN_BELOW_BASELINE"],
                           prob_volume_loss_gt_10pct=res["PROB_VOLUME_LOSS_GT_10PCT"]))
    persist(session, results, fans, drvs)
    ok = [p for p in points if p["prob_margin_below_baseline"] <= max_prob_below_baseline and p["price_change_pct"] != 0]
    best = max(ok, key=lambda p: p["margin_delta_p50"]) if ok else None
    best_any = max(points, key=lambda p: p["margin_delta_p50"])
    return {"sweep_id": sweep_id, "customer": ctx["customer_name"], "product_line": (sku_group or "ALL").upper(),
            "config_summary": config_summary(cfg, {"comp_mode": mode, "comp_move_pct": comp_move_in or 0}),
            "risk_limit_prob_margin_below_baseline": max_prob_below_baseline,
            "best_within_risk_limit": best, "best_unconstrained": best_any, "points": points,
            "total_sec": round(time.time() - t0, 2)}


def describe_model(session, customer_id, sku_group):
    ctx = load_context(session, customer_id, sku_group)
    snap = model_snapshot(ctx)
    cp = ctx["cp"].sort_values("OUR_PRICE_MOVE")
    ev = session.sql("SELECT * FROM PRICE_EVENTS WHERE CUSTOMER_ID = ?", params=[customer_id]).to_pandas()
    window = session.sql("SELECT MIN(WEEK_START), MAX(WEEK_START), COUNT(*) FROM SALES_HISTORY WHERE CUSTOMER_ID = ?",
                         params=[customer_id]).collect()[0]
    hist = None
    if not ev.empty:
        inc = ev[ev.OUR_PRICE_MOVE > 0]
        hist = {"n_past_price_moves": int(len(ev)),
                "historical_competitor_response_to_increases": inc.COMP_RESPONSE.value_counts(normalize=True).round(2).to_dict(),
                "backtest_mean_abs_error_volume": round(float(ev.PREDICTION_ERROR.abs().mean()), 4),
                "backtest_bias_volume": round(float(ev.PREDICTION_ERROR.mean()), 4)}
    return {
        "customer": ctx["customer_name"], "channel": ctx["channel"], "product_line": (sku_group or "ALL").upper(),
        "data_window": {"from": str(window[0]), "to": str(window[1]), "weekly_rows": int(window[2])},
        "model_fit_ts": ctx["fit_ts"], **snap,
        "competitor_response_model": [{"our_move": round(r.OUR_PRICE_MOVE, 3), "p_match": round(r.P_MATCH, 2),
                                       "p_undercut": round(r.P_UNDERCUT, 2), "p_hold": round(r.P_HOLD, 2),
                                       "undercut_move_mean": round(r.UNDERCUT_MOVE_MEAN, 3), "lag_weeks_mean": round(r.LAG_WEEKS_MEAN, 1)}
                                      for r in cp.itertuples()],
        "history_check": hist,
        "simulation_defaults": DEFAULT_CONFIG,
        "uncertainty_sources": SOURCE_LABELS,
        "fixed_assumptions": [
            "Our new price applies from month 1 for the full horizon",
            "Competitor reacts once, after a lag drawn uniformly from comp_lag_weeks",
            "At most one share-shift event per scenario, in a random month, persisting to horizon end",
            "Non-lead unit cost fixed per product line; lead cost = kg per unit × LME path",
            "Baseline = same random draws with no price change and no competitor move",
        ],
    }
