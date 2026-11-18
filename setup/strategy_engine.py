"""Voltara strategy engine v3: market-level "starve -> pivot -> adapt" Monte Carlo.

A scenario is a pricing STRATEGY over time in one target market (deep cut, hold, then an adaptation rule).
Competitors respond by moving capacity between the 6 application markets (pivot), driven by fitted margin-squeeze
hazard and destination-choice models; minor competitor price cuts are a small extra reaction.
Deterministic NumPy given inputs + seed: the LLM never computes these numbers. Fractions in, e.g. 0.12 = 12%.
"""
import copy
import datetime
import json
import time
import uuid

import numpy as np
import pandas as pd

MKTS = ["AFTERMARKET", "OEM", "COMMERCIAL", "MARINE", "ENERGY", "OTHER"]
COMPS = ["VOLTX", "APEX", "REST"]
MARKET_ALIASES = {
    "AFTERMARKET": "AFTERMARKET", "AUTO AFTERMARKET": "AFTERMARKET", "AUTO_AFTERMARKET": "AFTERMARKET", "RETAIL": "AFTERMARKET",
    "OEM": "OEM", "AUTO OEM": "OEM", "AUTO_OEM": "OEM",
    "COMMERCIAL": "COMMERCIAL", "HEAVY DUTY": "COMMERCIAL", "HEAVY-DUTY": "COMMERCIAL", "FLEET": "COMMERCIAL",
    "COMMERCIAL / HEAVY-DUTY": "COMMERCIAL", "TRUCK": "COMMERCIAL",
    "MARINE": "MARINE", "POWERSPORTS": "MARINE", "MARINE & POWERSPORTS": "MARINE",
    "ENERGY": "ENERGY", "ENERGY STORAGE": "ENERGY", "INDUSTRIAL": "ENERGY", "ENERGY STORAGE / INDUSTRIAL": "ENERGY",
    "OTHER": "OTHER",
}
COMP_ALIASES = {"VOLTX": "VOLTX", "VOLT-X": "VOLTX", "VOLT-X IMPORTS": "VOLTX", "VOLT X": "VOLTX",
                "APEX": "APEX", "APEX POWER": "APEX", "REST": "REST", "REST OF MARKET": "REST", "OTHERS": "REST"}
SOURCES = {
    "elasticity": "Market demand elasticity (fitted ± SE per market)",
    "share_response": "Our share response to price and to competitor capacity (fitted ± SE per market)",
    "competitor_cost": "Competitor unit cost per market (so a price cut maps to their margin)",
    "pivot_timing": "Whether and when a squeezed competitor starts pivoting (monthly hazard on margin squeeze, ± coefficient SE)",
    "pivot_destination": "Where pivoted capacity goes (fitted destination choice) and how much moves",
    "competitor_price_cuts": "Minor competitor price cuts (fitted frequency and size)",
    "market_growth": "Market growth rates (fitted ± SE)",
    "lead": "LME lead price path (GBM)",
    "demand_noise": "Unexplained month-to-month demand variation",
}
ADAPT_RULES = {
    "restore_after": "Restore price after the cut period (ramp back over restore_months)",
    "restore_on_exit": "Hold the cut until the target competitor has pulled exit_threshold of its capacity out (or max_cut_months), then restore",
    "hold": "Keep the cut for the whole horizon",
    "follow": "Restore on exit, AND defend any market the target competitor pivots into with a follow_cut_pct cut for follow_months",
}
DEFAULT_SPEC = {"target_market": "AFTERMARKET", "target_competitor": "VOLTX", "cut_pct": 0.10, "cut_months": 6,
                "ramp_in_months": 1, "adapt": "restore_after", "restore_months": 3, "exit_threshold": 0.30,
                "max_cut_months": 12, "follow_cut_pct": 0.05, "follow_months": 6, "lead_shock": 0.0, "label": None}
DEFAULT_CONFIG = {"n_runs": 5000, "horizon_months": 24, "seed": 7, "uncertainty": {k: True for k in SOURCES},
                  "disable_pivots": False, "force_destination": None}


def _json(v):
    if isinstance(v, str):
        return json.loads(v) if v.strip() else {}
    return v if isinstance(v, dict) else {}


def resolve(spec, config):
    s = copy.deepcopy(DEFAULT_SPEC); s.update({k: v for k, v in _json(spec).items() if v is not None})
    c = copy.deepcopy(DEFAULT_CONFIG)
    for k, v in _json(config).items():
        if k == "uncertainty" and isinstance(v, dict):
            c["uncertainty"].update({u: bool(b) for u, b in v.items() if u in SOURCES})
        elif v is not None:
            c[k] = v
    s["target_market"] = MARKET_ALIASES.get(str(s["target_market"]).upper(), str(s["target_market"]).upper())
    s["target_competitor"] = COMP_ALIASES.get(str(s["target_competitor"]).upper(), str(s["target_competitor"]).upper())
    if s["target_market"] not in MKTS:
        raise ValueError(f"Unknown market {s['target_market']}; use one of {MKTS}")
    if s["target_competitor"] not in COMPS:
        raise ValueError(f"Unknown competitor {s['target_competitor']}; use one of {COMPS}")
    if s["adapt"] not in ADAPT_RULES:
        raise ValueError(f"Unknown adapt rule {s['adapt']}; use one of {list(ADAPT_RULES)}")
    s["cut_pct"] = float(min(max(abs(float(s["cut_pct"])), 0.0), 0.40))
    for k in ["cut_months", "ramp_in_months", "restore_months", "max_cut_months", "follow_months"]:
        s[k] = int(max(0, int(s[k])))
    c["n_runs"] = int(min(max(int(c["n_runs"]), 200), 50000))
    c["horizon_months"] = int(min(max(int(c["horizon_months"]), 6), 48))
    c["seed"] = int(c["seed"])
    return s, c


def describe_spec(s):
    txt = (f"Cut {s['target_market']} price {s['cut_pct']:.0%} for {s['cut_months']} mo to squeeze {s['target_competitor']}; "
           f"then {s['adapt'].replace('_', ' ')}")
    if s["adapt"] in ("restore_on_exit", "follow"):
        txt += f" (exit = {s['exit_threshold']:.0%} of its capacity, max {s['max_cut_months']} mo)"
    if s["adapt"] == "follow":
        txt += f"; defend pivot markets {s['follow_cut_pct']:.0%} for {s['follow_months']} mo"
    if s.get("lead_shock"):
        txt += f"; lead {s['lead_shock']:+.0%}"
    return txt


def config_summary(c):
    off = [k for k, v in c["uncertainty"].items() if not v]
    parts = [f"{c['n_runs']:,} runs", f"{c['horizon_months']} mo", f"seed {c['seed']}",
             "all 9 uncertainty sources on" if not off else "OFF: " + ", ".join(off)]
    if c.get("disable_pivots"):
        parts.append("pivots disabled")
    if c.get("force_destination"):
        parts.append(f"destination forced → {c['force_destination']}")
    return " · ".join(parts)


# ============================================================================================ context
def load_context(session):
    mk = session.table("MARKETS").to_pandas().set_index("MARKET").loc[MKTS]
    mp = session.table("MARKET_PARAMS").to_pandas().set_index("MARKET").loc[MKTS]
    sp = session.table("SHARE_PARAMS").to_pandas().set_index("MARKET").loc[MKTS]
    cp = session.table("COMPETITOR_PARAMS").to_pandas().set_index("COMPETITOR").loc[COMPS]
    lp = session.table("LEAD_PARAMS").to_pandas().iloc[0]
    last = session.sql("""SELECT MARKET, PLAYER, AVG(SHARE) SHARE, AVG(CAPACITY) CAPACITY
                          FROM MARKET_HISTORY WHERE MONTH >= DATEADD(month, -2, (SELECT MAX(MONTH) FROM MARKET_HISTORY))
                          GROUP BY 1, 2""").to_pandas()
    share = last.pivot(index="PLAYER", columns="MARKET", values="SHARE").loc[["US"] + COMPS, MKTS].values
    share = share / share.sum(0, keepdims=True)
    cap = last.pivot(index="PLAYER", columns="MARKET", values="CAPACITY").loc[COMPS, MKTS].values
    fit_ts = session.sql("""SELECT MAX(LAST_ALTERED) FROM INFORMATION_SCHEMA.TABLES
                            WHERE TABLE_SCHEMA = CURRENT_SCHEMA() AND TABLE_NAME = 'MARKET_PARAMS'""").collect()[0][0]
    cost = np.array([[json.loads(cp.loc[j, "COST_FRAC"])[m] for m in MKTS] for j in COMPS])
    cost_sd = np.array([[json.loads(cp.loc[j, "COST_FRAC_SD"])[m] for m in MKTS] for j in COMPS])
    return dict(mk=mk, mp=mp, sp=sp, cp=cp, lp=lp, share0=share, cap0=cap, cost=cost, cost_sd=cost_sd,
                season=np.array([json.loads(x) for x in mp.SEASON_IDX]), fit_ts=str(fit_ts))


def model_snapshot(ctx):
    mp, sp, cp = ctx["mp"], ctx["sp"], ctx["cp"]
    return {"model_fit_ts": ctx["fit_ts"],
            "markets": {m: {"elasticity": round(mp.loc[m, "ELASTICITY"], 3), "elasticity_se": round(mp.loc[m, "ELASTICITY_SE"], 3),
                            "growth": round(mp.loc[m, "GROWTH"], 4), "share_price_sens": round(sp.loc[m, "PRICE_SENS"], 3),
                            "share_capacity_sens": round(sp.loc[m, "CAPACITY_SENS"], 3)} for m in MKTS},
            "competitors": {j: {"price_follow": round(cp.loc[j, "PRICE_FOLLOW"], 3),
                                "hazard": [round(cp.loc[j, "HAZARD_INTERCEPT"], 3), round(cp.loc[j, "HAZARD_SLOPE"], 2)],
                                "hazard_slope_se": round(cp.loc[j, "HAZARD_SLOPE_SE"], 2),
                                "pivot_fraction": round(cp.loc[j, "PIVOT_FRACTION_MEAN"], 3),
                                "destination_theta": round(cp.loc[j, "DEST_THETA"], 2), "n_pivots": int(cp.loc[j, "N_PIVOTS"])}
                            for j in COMPS}}


# ============================================================================================ simulation
def _draws(ctx, c):
    """All random inputs for n paths (common random numbers shared by scenario, baseline and sweep points)."""
    n, H, u = c["n_runs"], c["horizon_months"], c["uncertainty"]
    rng = np.random.default_rng(c["seed"])
    J, M = len(COMPS), len(MKTS)
    mp, sp, cp, lp = ctx["mp"], ctx["sp"], ctx["cp"], ctx["lp"]
    z = lambda *shape: rng.standard_normal(shape)
    d = {}
    d["e"] = np.minimum(mp.ELASTICITY.values + mp.ELASTICITY_SE.values * z(n, M) * u["elasticity"], -0.05)
    d["beta"] = np.maximum(sp.PRICE_SENS.values + sp.PRICE_SENS_SE.values * z(n, M) * u["share_response"], 0.2)
    d["gam"] = np.maximum(sp.CAPACITY_SENS.values + sp.CAPACITY_SENS_SE.values * z(n, M) * u["share_response"], 0.0)
    d["cost"] = np.clip(ctx["cost"][None] + ctx["cost_sd"][None] * z(n, J, M) * u["competitor_cost"], 0.3, 0.97)
    a = cp.HAZARD_INTERCEPT.values; b = cp.HAZARD_SLOPE.values
    d["haz_a"] = a[None, :] + cp.HAZARD_INTERCEPT_SE.values * z(n, J) * 0.25 * u["pivot_timing"]
    d["haz_b"] = np.maximum(b[None, :] + cp.HAZARD_SLOPE_SE.values * z(n, J) * u["pivot_timing"], 0.0)
    d["haz_u"] = rng.uniform(size=(n, H, J, M))
    d["dest_u"] = rng.uniform(size=(n, J, M))
    d["frac"] = np.clip(cp.PIVOT_FRACTION_MEAN.values[None, :, None]
                        + cp.PIVOT_FRACTION_SD.values[None, :, None] * z(n, J, M) * u["pivot_destination"], 0.05, 0.8)
    pcut = cp.MINOR_CUT_PROB.values[None, None, :, None]
    d["cut_on"] = (rng.uniform(size=(n, H, J, M)) < pcut) & u["competitor_price_cuts"]
    d["cut_size"] = np.minimum(cp.MINOR_CUT_MEAN.values[None, None, :, None]
                               + cp.MINOR_CUT_SD.values[None, None, :, None] * z(n, H, J, M), -0.005)
    d["g"] = mp.GROWTH.values + mp.GROWTH_SE.values * z(n, M) * u["market_growth"]
    dt = 1 / 12
    if u["lead"]:
        lp_path = np.cumsum((lp.ANNUAL_DRIFT - 0.5 * lp.ANNUAL_VOL ** 2) * dt + lp.ANNUAL_VOL * np.sqrt(dt) * z(n, H), axis=1)
    else:
        lp_path = np.zeros((n, H))
    d["lead"] = lp.SPOT_USD_PER_T * np.exp(lp_path)
    d["noise"] = z(n, H, M) * mp.NOISE_SD.values * u["demand_noise"]
    return d


def _price_plan(s, H):
    """Our price multiplier in the target market for the non-adaptive part of the strategy (per month)."""
    plan = np.ones(H)
    cut, ramp_in, months = s["cut_pct"], max(s["ramp_in_months"], 1), s["cut_months"]
    for t in range(H):
        if t < ramp_in:
            plan[t] = 1 - cut * (t + 1) / ramp_in
        elif t < months:
            plan[t] = 1 - cut
    return plan


def simulate(ctx, s, c, d, strategy_on=True):
    """Run all paths month by month. Returns per-path arrays (margins by market/month, capacities, flows, events)."""
    n, H = c["n_runs"], c["horizon_months"]
    J, M = len(COMPS), len(MKTS)
    mk, mp, cp = ctx["mk"], ctx["mp"], ctx["cp"]
    tm, tj = MKTS.index(s["target_market"]), COMPS.index(s["target_competitor"])
    P0 = mk.BASE_PRICE.values
    nl, kg = mk.NONLEAD_COST_FRAC.values * P0, mk.LEAD_KG_PER_UNIT.values
    s0 = ctx["share0"]                      # (1+J, M)
    cap0 = ctx["cap0"]                      # (J, M)
    rho = cp.PRICE_FOLLOW.values[None, :, None]
    theta = cp.DEST_THETA.values
    ramp = np.maximum(np.round(cp.RAMP_MONTHS_MEAN.values).astype(int), 1)
    ret_rate = cp.RETURN_RATE.values
    base_units = mp.CURRENT_MONTHLY_UNITS.values
    cal = (np.arange(H) + datetime.date.today().month) % 12
    season = ctx["season"][:, cal].T        # (H, M)
    u = c["uncertainty"]
    force = c.get("force_destination")
    force_idx = MKTS.index(MARKET_ALIASES.get(str(force).upper(), str(force).upper())) if force else None

    cap = np.broadcast_to(cap0, (n, J, M)).copy()
    started = np.zeros((n, J, M), bool)
    start_t = np.full((n, J, M), H + 1)
    total = np.zeros((n, J, M))            # units/yr to move
    moved = np.zeros((n, J, M))
    returned = np.zeros((n, J, M))
    W = np.zeros((n, J, M, M))             # destination weights (one-hot when stochastic)
    surv = np.ones((n, J, M))              # for deterministic timing
    our = np.ones((n, M))
    restore_start = np.full(n, -1)
    exited = np.zeros(n, bool)
    follow_until = np.full((n, M), -1)
    plan = _price_plan(s, H) if strategy_on else np.ones(H)

    margin = np.zeros((n, H, M)); units_us = np.zeros((n, H, M)); units_comp = np.zeros((n, H, J, M))
    cap_hist = np.zeros((n, H, J, M)); price_hist = np.zeros((n, H, M))
    base_margin = 1 - d["cost"]            # competitor margin at no cut (n, J, M)

    for t in range(H):
        # ---- our prices: strategy + adaptation
        our[:] = 1.0
        if strategy_on:
            rule = s["adapt"]
            exit_now = cap[:, tj, tm] <= (1 - s["exit_threshold"]) * cap0[tj, tm]
            exited |= exit_now
            if rule == "hold":
                our[:, tm] = plan[t] if t < s["ramp_in_months"] else 1 - s["cut_pct"]
            elif rule == "restore_after":
                if t < s["cut_months"]:
                    our[:, tm] = plan[t]
                else:
                    k = t - s["cut_months"] + 1
                    our[:, tm] = 1 - s["cut_pct"] * max(0.0, 1 - k / max(s["restore_months"], 1))
            else:  # restore_on_exit / follow: hold cut until exit or max_cut_months
                cutting = (restore_start < 0) & ~((exited & (t >= s["cut_months"])) | (t >= s["max_cut_months"]))
                newly = (restore_start < 0) & ~cutting
                restore_start[newly] = t
                k = np.where(restore_start >= 0, t - restore_start + 1, 0)
                level = np.where(cutting, 1 - s["cut_pct"],
                                 1 - s["cut_pct"] * np.maximum(0.0, 1 - k / max(s["restore_months"], 1)))
                if t < s["ramp_in_months"]:
                    level = np.full(n, plan[t])
                our[:, tm] = level
                if rule == "follow":
                    arrived = (cap[:, tj, :] - cap0[tj]) > 0.10 * cap0[tj]
                    arrived[:, tm] = False
                    start_follow = arrived & (follow_until < 0)
                    follow_until[start_follow] = t + s["follow_months"]
                    defend = (follow_until >= t)
                    our = np.where(defend, np.minimum(our, 1 - s["follow_cut_pct"]), our)
        # ---- competitor prices and margins
        follow = 1 + rho * (our[:, None, :] - 1)                     # (n, J, M)
        minor = np.where(d["cut_on"][:, t], 1 + d["cut_size"][:, t], 1.0)
        p_comp = P0[None, None, :] * follow * minor
        margin_c = (follow - d["cost"]) / follow                       # squeeze ignores their own minor cuts
        squeeze = np.clip(base_margin - margin_c, 0, None)
        # ---- pivot decisions
        if not (c.get("disable_pivots")):
            h = 1 / (1 + np.exp(-(d["haz_a"][:, :, None] + d["haz_b"][:, :, None] * squeeze)))
            h = np.where(squeeze > 0.005, h, 0.0)
            if u["pivot_timing"]:
                trig = (~started) & (d["haz_u"][:, t] < h)
            else:
                surv = surv * (1 - h)
                trig = (~started) & (surv < 0.5)
            if trig.any():
                # destination attractiveness: competitor margin elsewhere + growth - our share there
                share_us_now = s0[0][None, :]
                attr = margin_c + d["g"][:, None, :] - share_us_now[:, None, :]          # (n, J, M) for destination k
                for fm in range(M):
                    tr = trig[:, :, fm]
                    if not tr.any():
                        continue
                    a_ = attr.copy(); a_[:, :, fm] = -np.inf
                    if force_idx is not None and force_idx != fm:
                        probs = np.zeros((n, J, M)); probs[:, :, force_idx] = 1.0
                    else:
                        ex = np.exp(theta[None, :, None] * (a_ - np.nanmax(np.where(np.isfinite(a_), a_, -1e9), axis=2, keepdims=True)))
                        ex[:, :, fm] = 0
                        probs = ex / ex.sum(2, keepdims=True)
                    if u["pivot_destination"] and force_idx is None:
                        cum = probs.cumsum(2)
                        pick = (d["dest_u"][:, :, fm][:, :, None] > cum).sum(2).clip(0, M - 1)
                        onehot = np.eye(M)[pick]
                        w = onehot
                    else:
                        w = probs
                    W[:, :, fm, :] = np.where(tr[:, :, None], w, W[:, :, fm, :])
                    total[:, :, fm] = np.where(tr, d["frac"][:, :, fm] * cap[:, :, fm], total[:, :, fm])
                started |= trig
                start_t = np.where(trig, t, start_t)
        # ---- move capacity (ramp out), and slow return once their margin recovers
        ramping = started & (moved < total - 1e-9)
        step = np.where(ramping, total / ramp[None, :, None], 0.0)
        step = np.minimum(step, total - moved)
        moved += step
        recovered = started & ~ramping & (squeeze < 0.01)
        back = np.where(recovered, np.minimum(ret_rate[None, :, None] * moved, moved - returned), 0.0)
        returned += back
        net = step - back                                              # (n, J, from)
        cap -= net
        cap += np.einsum("njf,njft->njt", net, W)
        cap = np.maximum(cap, 0.0)
        # ---- shares and demand
        cap_ratio = np.where(cap0[None] > 0, cap / np.maximum(cap0[None], 1e-9), 1.0)
        A_us = s0[0][None, :] * our ** (-d["beta"])
        A_c = s0[1:][None] * np.maximum(cap_ratio, 1e-6) ** d["gam"][:, None, :] * (p_comp / P0) ** (-d["beta"][:, None, :])
        tot = A_us + A_c.sum(1)
        sh_us, sh_c = A_us / tot, A_c / tot[:, None, :]
        pavg = sh_us * our + (sh_c * p_comp / P0).sum(1)
        demand = base_units * (1 + d["g"]) ** (t / 12) * season[t] * pavg ** d["e"] * np.exp(d["noise"][:, t])
        lead = d["lead"][:, t:t + 1] * (1 + (s.get("lead_shock") or 0.0))
        unit_cost = nl[None, :] + kg[None, :] * lead / 1000
        units_us[:, t] = sh_us * demand
        units_comp[:, t] = sh_c * demand[:, None, :]
        margin[:, t] = units_us[:, t] * (P0 * our - unit_cost)
        cap_hist[:, t] = cap
        price_hist[:, t] = our
    flows = moved - returned                                           # net units/yr moved out of each from-market
    flow_mat = np.einsum("njf,njft->njft", flows, W)
    return dict(margin=margin, units_us=units_us, units_comp=units_comp, cap=cap_hist, price=price_hist,
                flow_mat=flow_mat, start_t=start_t, frac=total / np.maximum(np.broadcast_to(cap0, (n, J, M)), 1e-9),
                exited=exited)


# ============================================================================================ summaries
def _pct(a, qs=(10, 50, 90)):
    return [float(np.percentile(a, q)) for q in qs]


def summarize(ctx, s, c, d, sim, base):
    n, H = c["n_runs"], c["horizon_months"]
    tm, tj = MKTS.index(s["target_market"]), COMPS.index(s["target_competitor"])
    dm_month = sim["margin"].sum(2) - base["margin"].sum(2)           # (n, H)
    cum = dm_month.cumsum(1)
    tot = cum[:, -1]
    trough = cum.min(1)
    dipped = trough < 0
    paid = dipped & (tot >= 0)
    pay_m = np.full(n, np.nan)
    for i in np.where(paid)[0]:
        tmin = cum[i].argmin()
        after = np.where(cum[i, tmin:] >= 0)[0]
        if len(after):
            pay_m[i] = tmin + after[0] + 1
    cap_t = sim["cap"][:, :, tj, tm]
    exit_p = float((cap_t[:, -1] <= (1 - s["exit_threshold"]) * ctx["cap0"][tj, tm]).mean())
    piv_month = sim["start_t"][:, tj, tm] + 1          # 1-based month the pivot starts (H+2 = never)
    by_market = []
    for k, m in enumerate(MKTS):
        dmk = sim["margin"][:, :, k].sum(1) - base["margin"][:, :, k].sum(1)
        dvk = sim["units_us"][:, :, k].sum(1) - base["units_us"][:, :, k].sum(1)
        dck = sim["units_comp"][:, :, tj, k].sum(1) - base["units_comp"][:, :, tj, k].sum(1)
        sh_end = sim["units_us"][:, -1, k] / (sim["units_us"][:, -1, k] + sim["units_comp"][:, -1, :, k].sum(1))
        sh_base = base["units_us"][:, -1, k] / (base["units_us"][:, -1, k] + base["units_comp"][:, -1, :, k].sum(1))
        by_market.append(dict(market=m, margin_delta=_pct(dmk), our_volume_delta=_pct(dvk),
                              target_comp_volume_delta=float(np.median(dck)),
                              our_share_end=float(np.median(sh_end)), our_share_end_baseline=float(np.median(sh_base))))
    flows = []
    fm = sim["flow_mat"]                                               # (n, J, from, to)
    for j, comp in enumerate(COMPS):
        for f, mf in enumerate(MKTS):
            for to, mt in enumerate(MKTS):
                if f == to:
                    continue
                v = fm[:, j, f, to]
                p = float((v > 1).mean())
                if p >= 0.02:
                    flows.append(dict(competitor=comp, from_market=mf, to_market=mt, expected_units_per_year=float(v.mean()),
                                      p50_if_happens=float(np.median(v[v > 1])) if (v > 1).any() else 0.0, probability=p))
    flows.sort(key=lambda x: -x["expected_units_per_year"])
    cap_med = np.median(sim["cap"][:, :, tj, :], axis=0)              # (H, M)
    cap_base_med = np.median(base["cap"][:, :, tj, :], axis=0)
    drivers_src = {
        "Market elasticity (target)": d["e"][:, tm], "Share price sensitivity (target)": d["beta"][:, tm],
        "Target competitor cost": d["cost"][:, tj, tm], "Pivot timing (month target pivots)": piv_month.astype(float),
        "Pivot size": sim["frac"][:, tj, tm], "Market growth (target)": d["g"][:, tm],
        "LME lead price": -d["lead"].mean(1), "Demand noise": d["noise"][:, :, tm].mean(1)}
    rk = pd.Series(tot).rank().values
    drivers = []
    for name, x in drivers_src.items():
        if np.std(x) < 1e-12 or np.std(tot) < 1e-9:
            drivers.append(dict(driver=name, variance_share=0.0)); continue
        drivers.append(dict(driver=name, variance_share=float(np.corrcoef(pd.Series(x).rank().values, rk)[0, 1] ** 2)))
    ssum = sum(x["variance_share"] for x in drivers) or 1.0
    for x in drivers:
        x["variance_share"] /= ssum
    drivers.sort(key=lambda x: -x["variance_share"])
    total_margin = sim["margin"].sum((1, 2)); base_total = base["margin"].sum((1, 2))
    return dict(
        summary=dict(margin_total=_pct(total_margin), baseline_margin_total_p50=float(np.median(base_total)),
                     margin_delta=_pct(tot), trough_p50=float(np.median(trough)), prob_loss=float((tot < 0).mean()),
                     prob_payback=float(paid.mean() / max(dipped.mean(), 1e-9)) if dipped.any() else 1.0,
                     payback_month_p50=float(np.nanmedian(pay_m)) if np.isfinite(pay_m).any() else None,
                     prob_target_exit=exit_p,
                     prob_target_pivots=float((piv_month <= H).mean()),
                     target_pivot_month_p50=float(np.median(piv_month[piv_month <= H])) if (piv_month <= H).any() else None,
                     our_share_target_end=by_market[tm]["our_share_end"],
                     our_share_target_end_baseline=by_market[tm]["our_share_end_baseline"]),
        cum_curve=[dict(month=t + 1, **{f"p{q}": float(np.percentile(cum[:, t], q)) for q in (10, 25, 50, 75, 90)}) for t in range(H)],
        target_capacity=[dict(month=t + 1, market=m, p50=float(cap_med[t, k]), baseline_p50=float(cap_base_med[t, k]),
                              **({"p10": float(np.percentile(sim["cap"][:, t, tj, k], 10)),
                                  "p90": float(np.percentile(sim["cap"][:, t, tj, k], 90))} if k == tm else {}))
                         for t in range(H) for k, m in enumerate(MKTS)],
        price_path=[dict(month=t + 1, market=m, p50=float(np.median(sim["price"][:, t, k]))) for t in range(H)
                    for k, m in enumerate(MKTS) if np.ptp(sim["price"][:, :, k]) > 1e-9 or k == tm],
        by_market=by_market, flows=flows[:30], drivers=drivers)


# ============================================================================================ persistence
RESULT_COLS = ["SCENARIO_ID", "LABEL", "QUESTION", "TARGET_MARKET", "TARGET_COMPETITOR", "CUT_PCT", "CUT_MONTHS", "ADAPT_RULE",
               "HORIZON_MONTHS", "N_RUNS", "MARGIN_DELTA_P10", "MARGIN_DELTA_P50", "MARGIN_DELTA_P90", "PROB_LOSS",
               "PROB_PAYBACK", "PAYBACK_MONTH_P50", "PROB_TARGET_EXIT", "SWEEP_ID", "RUNTIME_SEC",
               "SPEC_JSON", "CONFIG_JSON", "OUTPUT_JSON", "MODEL_JSON"]


def ensure_table(session):
    session.sql("""CREATE TABLE IF NOT EXISTS STRATEGY_RESULTS (
        SCENARIO_ID STRING, RUN_TS TIMESTAMP_LTZ, LABEL STRING, QUESTION STRING, TARGET_MARKET STRING, TARGET_COMPETITOR STRING,
        CUT_PCT FLOAT, CUT_MONTHS INT, ADAPT_RULE STRING, HORIZON_MONTHS INT, N_RUNS INT,
        MARGIN_DELTA_P10 FLOAT, MARGIN_DELTA_P50 FLOAT, MARGIN_DELTA_P90 FLOAT, PROB_LOSS FLOAT, PROB_PAYBACK FLOAT,
        PAYBACK_MONTH_P50 FLOAT, PROB_TARGET_EXIT FLOAT, SWEEP_ID STRING, RUNTIME_SEC FLOAT,
        SPEC VARIANT, CONFIG VARIANT, OUTPUT VARIANT, MODEL_SNAPSHOT VARIANT)""").collect()


def persist(session, rows):
    ensure_table(session)
    df = pd.DataFrame([{k: r.get(k) for k in RESULT_COLS} for r in rows])
    session.write_pandas(df, "STRATEGY_RESULTS_STAGE", auto_create_table=True, overwrite=True, quote_identifiers=False,
                         table_type="temporary")
    plain = ",".join(RESULT_COLS[:-4])
    session.sql(f"""INSERT INTO STRATEGY_RESULTS (RUN_TS, {plain}, SPEC, CONFIG, OUTPUT, MODEL_SNAPSHOT)
                    SELECT CURRENT_TIMESTAMP(), {plain}, PARSE_JSON(SPEC_JSON), PARSE_JSON(CONFIG_JSON),
                           PARSE_JSON(OUTPUT_JSON), PARSE_JSON(MODEL_JSON) FROM STRATEGY_RESULTS_STAGE""").collect()


def _row(sid, s, c, out, ctx, runtime, question=None, sweep_id=None):
    sm = out["summary"]
    label = s.get("label") or describe_spec(s)
    out = dict(out, spec_text=describe_spec(s), config_text=config_summary(c))
    return dict(SCENARIO_ID=sid, LABEL=label, QUESTION=question, TARGET_MARKET=s["target_market"],
                TARGET_COMPETITOR=s["target_competitor"], CUT_PCT=s["cut_pct"], CUT_MONTHS=s["cut_months"], ADAPT_RULE=s["adapt"],
                HORIZON_MONTHS=c["horizon_months"], N_RUNS=c["n_runs"], MARGIN_DELTA_P10=sm["margin_delta"][0],
                MARGIN_DELTA_P50=sm["margin_delta"][1], MARGIN_DELTA_P90=sm["margin_delta"][2], PROB_LOSS=sm["prob_loss"],
                PROB_PAYBACK=sm["prob_payback"], PAYBACK_MONTH_P50=sm["payback_month_p50"], PROB_TARGET_EXIT=sm["prob_target_exit"],
                SWEEP_ID=sweep_id, RUNTIME_SEC=runtime, SPEC_JSON=json.dumps(s), CONFIG_JSON=json.dumps(c),
                OUTPUT_JSON=json.dumps(out), MODEL_JSON=json.dumps(model_snapshot(ctx)))


# ============================================================================================ handlers (SQL procedures)
def run_strategy(session, spec, config=None, question=None):
    t0 = time.time()
    try:
        s, c = resolve(spec, config)
    except ValueError as e:
        return {"error": str(e)}
    ctx = load_context(session)
    d = _draws(ctx, c)
    sim = simulate(ctx, s, c, d, True)
    base = simulate(ctx, s, c, d, False)
    out = summarize(ctx, s, c, d, sim, base)
    sid = str(uuid.uuid4())
    persist(session, [_row(sid, s, c, out, ctx, time.time() - t0, question)])
    return {"scenario_id": sid, "spec": s, "config_summary": config_summary(c), "spec_text": describe_spec(s),
            "summary": out["summary"], "by_market": out["by_market"], "top_flows": out["flows"][:6],
            "drivers": out["drivers"][:5], "runtime_sec": round(time.time() - t0, 2)}


def run_strategy_sweep(session, spec, cuts, months, config=None, max_loss_risk=0.25, question=None):
    t0 = time.time()
    cuts = [float(x) for x in (json.loads(cuts) if isinstance(cuts, str) else cuts)][:6]
    months = [int(x) for x in (json.loads(months) if isinstance(months, str) else months)][:5]
    cfg = _json(config); cfg.setdefault("n_runs", 2000)
    try:
        s0, c = resolve(spec, cfg)
    except ValueError as e:
        return {"error": str(e)}
    ctx = load_context(session)
    d = _draws(ctx, c)
    base = simulate(ctx, s0, c, d, False)
    sweep_id, rows, points = str(uuid.uuid4()), [], []
    for cut in cuts:
        for mo in months:
            s = dict(s0, cut_pct=abs(cut), cut_months=mo, label=None)
            s["max_cut_months"] = max(s0["max_cut_months"], mo)
            sim = simulate(ctx, s, c, d, True)
            out = summarize(ctx, s, c, d, sim, base)
            sid = str(uuid.uuid4())
            s["label"] = f"[sweep] {describe_spec(s)}"
            rows.append(_row(sid, s, c, out, ctx, 0.0, question, sweep_id))
            sm = out["summary"]
            points.append(dict(scenario_id=sid, cut_pct=abs(cut), cut_months=mo, margin_delta=sm["margin_delta"],
                               prob_loss=sm["prob_loss"], prob_target_exit=sm["prob_target_exit"],
                               payback_month_p50=sm["payback_month_p50"]))
    persist(session, rows)
    ok = [p for p in points if p["prob_loss"] <= max_loss_risk]
    best = max(ok, key=lambda p: p["margin_delta"][1]) if ok else None
    return {"sweep_id": sweep_id, "spec_text": describe_spec(s0) + " (cut depth × duration varied)",
            "config_summary": config_summary(c), "risk_limit": max_loss_risk, "points": points,
            "best_within_risk_limit": best, "best_unconstrained": max(points, key=lambda p: p["margin_delta"][1]),
            "runtime_sec": round(time.time() - t0, 2)}


def explain_strategy(session, scenario_id):
    r = session.sql("SELECT SPEC, CONFIG, OUTPUT FROM STRATEGY_RESULTS WHERE SCENARIO_ID = ?", params=[scenario_id]).collect()
    if not r:
        return {"error": f"Scenario {scenario_id} not found"}
    s, c, out = json.loads(r[0][0]), json.loads(r[0][1]), json.loads(r[0][2])
    c = dict(c, n_runs=min(int(c["n_runs"]), 3000))
    ctx = load_context(session)
    d = _draws(ctx, c)
    base = simulate(ctx, s, c, d, False)
    dests = [f["to_market"] for f in out["flows"] if f["competitor"] == s["target_competitor"]
             and f["from_market"] == s["target_market"]][:2]
    variants = [("Competitors never pivot", dict(c, disable_pivots=True), s)]
    for m in dests:
        variants.append((f"{s['target_competitor']} pivots into {m}", dict(c, force_destination=m), s))
    variants.append(("No minor competitor price cuts", dict(c, uncertainty=dict(c["uncertainty"], competitor_price_cuts=False)), s))
    variants.append(("Lead +20% on top", c, dict(s, lead_shock=(s.get("lead_shock") or 0) + 0.20)))
    res, rows = [], []
    for name, cv, sv in variants:
        dv = _draws(ctx, cv)
        sim = simulate(ctx, sv, cv, dv, True)
        bv = simulate(ctx, sv, cv, dv, False)
        o = summarize(ctx, sv, cv, dv, sim, bv)
        sid = str(uuid.uuid4())
        sv2 = dict(sv, label=f"[stress] {name}: {describe_spec(sv)}")
        rows.append(_row(sid, sv2, cv, o, ctx, 0.0, f"explain {scenario_id}"))
        res.append(dict(what_if=name, scenario_id=sid, margin_delta=o["summary"]["margin_delta"],
                        prob_loss=o["summary"]["prob_loss"], prob_target_exit=o["summary"]["prob_target_exit"]))
    persist(session, rows)
    return {"scenario_id": scenario_id, "original": out["summary"], "drivers": out["drivers"], "stress_tests": res,
            "spec_text": describe_spec(s), "config_summary": config_summary(c)}


def market_outlook(session, market):
    m = MARKET_ALIASES.get(str(market).upper(), str(market).upper())
    if m not in MKTS:
        return {"error": f"Unknown market {market}; use one of {MKTS}"}
    ctx = load_context(session)
    k = MKTS.index(m)
    hist = session.sql("""SELECT PLAYER, SUM(UNITS) UNITS, AVG(SHARE) SHARE, AVG(CAPACITY) CAPACITY,
                                 AVG((PRICE - UNIT_COST) / PRICE) MARGIN_RATE
                          FROM MARKET_HISTORY WHERE MARKET = ? AND MONTH > DATEADD(month, -12, (SELECT MAX(MONTH) FROM MARKET_HISTORY))
                          GROUP BY 1""", params=[m]).to_pandas().set_index("PLAYER")
    moves = session.sql("""SELECT COMPETITOR, FROM_MARKET, TO_MARKET, START_MONTH, FRACTION_OF_CAPACITY FROM COMPETITOR_MOVES
                           WHERE FROM_MARKET = ? OR TO_MARKET = ? ORDER BY START_MONTH DESC""", params=[m, m]).to_pandas()
    mp = ctx["mp"].loc[m]
    return {"market": m, "market_name": ctx["mk"].loc[m, "MARKET_NAME"],
            "units_last_12m": float(hist.UNITS.sum()), "growth_per_year": float(mp.GROWTH), "elasticity": float(mp.ELASTICITY),
            "players": {p: {"share": round(float(hist.loc[p, "SHARE"]), 4), "margin_rate": round(float(hist.loc[p, "MARGIN_RATE"]), 4),
                            "capacity_units_per_year": None if p == "US" else round(float(hist.loc[p, "CAPACITY"]), 0)}
                        for p in hist.index},
            "competitor_moves_in_or_out": moves.astype(str).to_dict("records")}


def describe_market_model(session, market=None, competitor=None):
    ctx = load_context(session)
    snap = model_snapshot(ctx)
    cp = ctx["cp"]
    comps = {}
    for j in COMPS:
        comps[j] = dict(snap["competitors"][j], cost_frac_by_market=json.loads(cp.loc[j, "COST_FRAC"]),
                        destination_counts=json.loads(cp.loc[j, "DEST_COUNTS"]), return_rate=float(cp.loc[j, "RETURN_RATE"]),
                        minor_cut_prob_per_market_month=float(cp.loc[j, "MINOR_CUT_PROB"]))
    return {"models": [
        {"name": "Market demand", "count": 6, "method": "log-log OLS: total units ~ price index + trend + seasonality, per market"},
        {"name": "Market share response", "count": 6, "method": "OLS on logit of our share ~ our/competitor price + competitor capacity"},
        {"name": "Competitor cost / margin + price follow", "count": 3, "method": "observed unit cost per market; regression of their price change on ours"},
        {"name": "Competitor pivot hazard", "count": 3, "method": "logistic regression: P(start moving capacity out this month) ~ margin squeeze"},
        {"name": "Destination choice", "count": 3, "method": "softmax over other markets (margin + growth - our share), temperature by max likelihood"},
        {"name": "Minor competitor price cuts", "count": 1, "method": "frequency and size of observed cuts"},
        {"name": "LME lead price", "count": 1, "method": "geometric Brownian motion on daily log returns"}],
        "markets": snap["markets"], "competitors": comps, "uncertainty_sources": SOURCES, "adapt_rules": ADAPT_RULES,
        "defaults": {"spec": DEFAULT_SPEC, "config": DEFAULT_CONFIG}, "model_fit_ts": ctx["fit_ts"],
        "fixed_assumptions": [
            "Our capacity is not constrained; competitors' capacity moves between markets",
            "A competitor considers pivoting only when its margin in a market is squeezed (>0.5 pt below normal)",
            "At most one pivot per competitor per market per path; capacity ramps out over the fitted ramp months",
            "Pivoted capacity returns slowly (fitted monthly rate) once their margin recovers",
            "Competitors follow our price moves with their fitted price-follow ratio",
            "Baseline = same random draws with no strategy"]}


def find_squeeze_precedents(session, market=None, competitor=None, cut_pct=None):
    q = "SELECT * FROM SQUEEZE_EPISODES WHERE 1=1"
    params = []
    if market:
        q += " AND MARKET = ?"; params.append(MARKET_ALIASES.get(str(market).upper(), str(market).upper()))
    if competitor:
        q += " AND COMPETITOR = ?"; params.append(COMP_ALIASES.get(str(competitor).upper(), str(competitor).upper()))
    ep = session.sql(q, params=params).to_pandas()
    if ep.empty:
        ep = session.table("SQUEEZE_EPISODES").to_pandas()
    if cut_pct is not None:
        ep["DIST"] = (ep.CUT_PCT - abs(float(cut_pct))).abs()
        ep = ep.sort_values(["DIST", "START_MONTH"])
    ep = ep.drop(columns=[c for c in ["DIST"] if c in ep.columns]).head(12)
    return {"n_episodes": int(ep.EPISODE_ID.nunique()), "episodes": ep.astype(str).to_dict("records"),
            "pivot_rate": float(ep.PIVOTED.astype(bool).mean()) if len(ep) else None,
            "median_capacity_change_12m": float(ep.CAPACITY_CHANGE_12M.median()) if len(ep) else None,
            "note": "Measured from MARKET_HISTORY / COMPETITOR_MOVES. Quote as-is."}


# ============================================================================================ agent-facing wrappers
def _m(v):
    return f"${v / 1e6:,.1f}M"


def _fmt_summary(sm):
    return {"margin_change_vs_baseline_p50": _m(sm["margin_delta"][1]),
            "margin_change_range_p10_p90": f"{_m(sm['margin_delta'][0])} to {_m(sm['margin_delta'][2])}",
            "deepest_cumulative_dip_p50": _m(sm["trough_p50"]),
            "prob_strategy_loses_money_over_horizon": f"{sm['prob_loss']:.0%}",
            "prob_payback_within_horizon_if_it_dips": f"{sm['prob_payback']:.0%}",
            "payback_month_p50": sm["payback_month_p50"],
            "prob_target_competitor_exits_target_market": f"{sm['prob_target_exit']:.0%}",
            "prob_target_competitor_pivots": f"{sm['prob_target_pivots']:.0%}",
            "target_pivot_month_p50": sm["target_pivot_month_p50"],
            "our_share_in_target_market_end": f"{sm['our_share_target_end']:.1%} (baseline {sm['our_share_target_end_baseline']:.1%})"}


def tool_run_strategy(session, target_market, target_competitor, cut_pct, cut_months, adapt_rule, exit_threshold_pct,
                      follow_cut_pct, horizon_months, disable_uncertainty, lead_shock_pct, label):
    spec = {"target_market": target_market, "target_competitor": target_competitor or "VOLTX",
            "cut_pct": abs(float(cut_pct or 10)) / 100, "cut_months": int(cut_months or 6), "adapt": adapt_rule or "restore_after",
            "exit_threshold": (exit_threshold_pct or 30) / 100, "follow_cut_pct": (follow_cut_pct or 5) / 100,
            "lead_shock": (lead_shock_pct or 0) / 100, "label": label}
    off = [x.strip().lower() for x in (disable_uncertainty or "").split(",") if x.strip().lower() in SOURCES]
    cfg = {"horizon_months": int(horizon_months or 24), "uncertainty": {x: False for x in off}}
    r = run_strategy(session, spec, cfg, question=label)
    if r.get("error"):
        return r
    return {"scenario_id": r["scenario_id"], "strategy": r["spec_text"], "model_configuration": r["config_summary"],
            "outcome": _fmt_summary(r["summary"]),
            "by_market": [{"market": b["market"], "our_margin_change_p50": _m(b["margin_delta"][1]),
                           "our_volume_change_p50_units": f"{b['our_volume_delta'][1]:,.0f}",
                           "our_share_end": f"{b['our_share_end']:.1%} (baseline {b['our_share_end_baseline']:.1%})"}
                          for b in r["by_market"]],
            "competitor_capacity_flows": [{"competitor": f["competitor"], "from": f["from_market"], "to": f["to_market"],
                                           "expected_units_per_year": f"{f['expected_units_per_year']:,.0f}",
                                           "probability": f"{f['probability']:.0%}"} for f in r["top_flows"]],
            "drivers_of_outcome": [{"driver": x["driver"], "share_of_variance": f"{x['variance_share']:.0%}"} for x in r["drivers"]],
            "note": "All figures computed by the Monte Carlo engine. Quote as-is; do not recompute."}


def tool_strategy_sweep(session, target_market, target_competitor, cut_pcts, cut_months_list, adapt_rule, max_loss_risk_pct):
    cuts = [abs(float(x)) / 100 for x in str(cut_pcts or "5,10,15,20").replace(" ", "").split(",") if x]
    months = [int(float(x)) for x in str(cut_months_list or "3,6,9").replace(" ", "").split(",") if x]
    spec = {"target_market": target_market, "target_competitor": target_competitor or "VOLTX", "adapt": adapt_rule or "restore_on_exit"}
    r = run_strategy_sweep(session, spec, cuts, months, {"n_runs": 2000}, (max_loss_risk_pct or 25) / 100)
    if r.get("error"):
        return r
    f = lambda p: None if not p else {"cut": f"{p['cut_pct']:.0%}", "months": p["cut_months"], "scenario_id": p["scenario_id"],
                                      "margin_change_p50": _m(p["margin_delta"][1]),
                                      "range_p10_p90": f"{_m(p['margin_delta'][0])} to {_m(p['margin_delta'][2])}",
                                      "prob_loses_money": f"{p['prob_loss']:.0%}",
                                      "prob_target_exits": f"{p['prob_target_exit']:.0%}", "payback_month_p50": p["payback_month_p50"]}
    return {"sweep_id": r["sweep_id"], "strategy": r["spec_text"], "model_configuration": r["config_summary"],
            "risk_limit": f"P(strategy loses money) <= {r['risk_limit']:.0%}",
            "best_within_risk_limit": f(r["best_within_risk_limit"]) or "no tested strategy meets the risk limit",
            "best_ignoring_risk": f(r["best_unconstrained"]), "grid": [f(p) for p in r["points"]],
            "note": "2,000 runs per grid point, common random numbers. Quote as-is."}


def tool_explain_strategy(session, scenario_id):
    r = explain_strategy(session, scenario_id)
    if r.get("error"):
        return r
    return {"scenario_id": scenario_id, "strategy": r["spec_text"], "model_configuration": r["config_summary"],
            "original": _fmt_summary(r["original"]),
            "drivers_of_outcome": [{"driver": x["driver"], "share_of_variance": f"{x['variance_share']:.0%}"} for x in r["drivers"]],
            "stress_tests": [{"what_if": x["what_if"], "scenario_id": x["scenario_id"], "margin_change_p50": _m(x["margin_delta"][1]),
                              "range_p10_p90": f"{_m(x['margin_delta'][0])} to {_m(x['margin_delta'][2])}",
                              "prob_loses_money": f"{x['prob_loss']:.0%}", "prob_target_exits": f"{x['prob_target_exit']:.0%}"}
                             for x in r["stress_tests"]],
            "note": "Stress tests rerun by the engine (3,000 runs). Quote as-is."}
