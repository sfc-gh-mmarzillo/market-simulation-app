-- Market-level synthetic history for the "starve -> pivot -> adapt" demo (fictional Voltara Energy).
-- 6 application markets, 4 players (Voltara + Volt-X Imports, Apex Power, Rest of market), 72 months.
-- The TRUE process below is hidden from the models; FIT_MARKET_MODELS must recover it from the history.
CREATE OR REPLACE PROCEDURE WHATIF_DEMO.SIM.GENERATE_MARKET_DATA(SEED INT)
RETURNS STRING
LANGUAGE PYTHON
RUNTIME_VERSION = '3.11'
PACKAGES = ('snowflake-snowpark-python', 'numpy', 'pandas')
HANDLER = 'run'
AS
$$
import numpy as np
import pandas as pd

MKTS = ["AFTERMARKET", "OEM", "COMMERCIAL", "MARINE", "ENERGY", "OTHER"]
NAMES = {"AFTERMARKET": "Auto aftermarket", "OEM": "Auto OEM", "COMMERCIAL": "Commercial / heavy-duty",
         "MARINE": "Marine & powersports", "ENERGY": "Energy storage / industrial", "OTHER": "Other"}
COMPS = ["VOLTX", "APEX", "REST"]
CNAMES = {"VOLTX": "Volt-X Imports", "APEX": "Apex Power", "REST": "Rest of market"}

# market: annual units, base price, growth, true market elasticity, share price sens (beta), capacity effect (gamma),
#         lead kg/unit, non-lead cost (fraction of price), initial shares [US, VOLTX, APEX, REST]
M = {
    "AFTERMARKET": (6_000_000, 130, 0.02, -0.6, 3.0, 1.10, 10.5, 0.42, [0.38, 0.22, 0.12, 0.28]),
    "OEM":         (3_500_000, 110, 0.01, -0.3, 2.0, 0.80, 11.0, 0.45, [0.30, 0.05, 0.25, 0.40]),
    "COMMERCIAL":  (1_200_000, 240, 0.03, -0.5, 2.5, 1.00, 22.0, 0.40, [0.33, 0.10, 0.15, 0.42]),
    "MARINE":      (  800_000, 180, 0.04, -0.8, 3.0, 1.20, 14.0, 0.40, [0.25, 0.12, 0.10, 0.53]),
    "ENERGY":      (  600_000, 900, 0.18, -0.9, 2.0, 1.00,  0.0, 0.62, [0.12, 0.08, 0.30, 0.50]),
    "OTHER":       (1_500_000, 120, 0.02, -0.6, 2.0, 0.80, 10.0, 0.45, [0.15, 0.15, 0.10, 0.60]),
}
# competitor: unit cost as fraction of market base price (per market), price-follow rho, monthly pivot hazard (a, b)
#             on margin squeeze, fraction moved range, ramp months range, destination temperature theta,
#             monthly return rate once the squeeze ends, minor-cut probability
COST = {
    "VOLTX": {"AFTERMARKET": 0.80, "OEM": 0.86, "COMMERCIAL": 0.82, "MARINE": 0.78, "ENERGY": 0.84, "OTHER": 0.76},
    "APEX":  {"AFTERMARKET": 0.72, "OEM": 0.68, "COMMERCIAL": 0.70, "MARINE": 0.73, "ENERGY": 0.62, "OTHER": 0.75},
    "REST":  {"AFTERMARKET": 0.73, "OEM": 0.74, "COMMERCIAL": 0.71, "MARINE": 0.70, "ENERGY": 0.76, "OTHER": 0.70},
}
C = {
    "VOLTX": (None, 0.85, (-4.6, 42.0), (0.40, 0.80), (3, 6), 9.0, 0.008, 0.03),
    "APEX":  (None, 0.55, (-4.8, 48.0), (0.15, 0.40), (4, 8), 7.0, 0.01, 0.02),
    "REST":  (None, 0.65, (-5.0, 38.0), (0.10, 0.30), (5, 8), 5.0, 0.01, 0.02),
}

def sig(x):
    return 1 / (1 + np.exp(-x))

def run(session, seed):
    rng = np.random.default_rng(seed)
    T = 72
    months = pd.date_range("2020-12-01", periods=T, freq="MS")
    lead = 2150 * np.exp(np.cumsum(rng.normal(0, 0.22 / np.sqrt(12), T)))

    # our price path (multiplier on base) per market: deliberate squeeze episodes + small noise
    ours = {m: np.ones(T) for m in MKTS}
    def episode(m, start, length, depth, restore=3):
        for t in range(start, min(T, start + length)):
            ours[m][t] = 1 - depth
        for k in range(restore):
            t = start + length + k
            if t < T:
                ours[m][t] = 1 - depth * (1 - (k + 1) / (restore + 1))
    episodes = [("AFTERMARKET", 6, 8, 0.10), ("MARINE", 12, 7, 0.12), ("OEM", 18, 6, 0.08), ("AFTERMARKET", 26, 9, 0.15),
                ("COMMERCIAL", 33, 6, 0.10), ("OTHER", 38, 6, 0.12), ("ENERGY", 44, 6, 0.10), ("MARINE", 50, 6, 0.08),
                ("AFTERMARKET", 56, 8, 0.12)]
    for ep in episodes:
        episode(*ep)
    for m in MKTS:
        ours[m] *= 1 + rng.normal(0, 0.006, T)

    cap0 = {(j, m): M[m][0] * M[m][8][1 + COMPS.index(j)] for j in COMPS for m in MKTS}
    cap = dict(cap0)
    base_margin = {(j, m): 1 - COST[j][m] for j in COMPS for m in MKTS}
    active = []          # live pivots: dict(j, frm, to, units_total, ramp, t0, moved)
    pivoting = set()     # (j, m) currently pivoting out
    minor = {}           # (j, m) -> (until_t, factor)
    moves, price_events, rows = [], [], []

    for t in range(T):
        season = 1 + 0.12 * np.cos(2 * np.pi * (months[t].month - 1) / 12)
        for m in MKTS:
            D0, P0, g, e, beta, gam, kg, nl, s0 = M[m]
            p_us = P0 * ours[m][t]
            prices = {"US": p_us}
            for j in COMPS:
                f = 1 + C[j][1] * (ours[m][t] - 1)
                if (j, m) in minor and minor[(j, m)][0] >= t:
                    f *= minor[(j, m)][1]
                elif rng.random() < C[j][7]:
                    fac = 1 + min(rng.normal(-0.04, 0.015), -0.01)
                    minor[(j, m)] = (t + int(rng.integers(1, 3)), fac)
                    price_events.append((j, m, months[t].date(), "MINOR_CUT", round(fac - 1, 4)))
                    f *= fac
                prices[j] = P0 * f * (1 + rng.normal(0, 0.004))
            # attraction shares (identity at t=0)
            A = {"US": s0[0] * (p_us / P0) ** (-beta)}
            for k, j in enumerate(COMPS):
                A[j] = s0[k + 1] * (cap[(j, m)] / cap0[(j, m)]) ** gam * (prices[j] / P0) ** (-beta)
            A = {k: v * np.exp(rng.normal(0, 0.03)) for k, v in A.items()}  # unexplained share noise
            tot = sum(A.values())
            shares = {k: v / tot for k, v in A.items()}
            pavg = sum(shares[k] * prices[k] for k in shares) / P0
            demand = D0 / 12 * (1 + g) ** (t / 12) * season * pavg ** e * np.exp(rng.normal(0, 0.03))
            for k in ["US"] + COMPS:
                units = shares[k] * demand
                if k == "US":
                    cost = P0 * nl + kg * lead[t] / 1000
                    capacity = None
                else:
                    cost = P0 * COST[k][m] * (1 + rng.normal(0, 0.02))
                    capacity = cap[(k, m)]
                rows.append((months[t].date(), m, k, round(prices[k], 3), round(units, 1), round(cost, 3),
                             None if capacity is None else round(capacity, 1), round(shares[k], 5), round(float(lead[t]), 2)))
        # competitor pivot decisions (based on this month's margins)
        margins = {}
        for j in COMPS:
            for m in MKTS:
                P0 = M[m][1]
                pj = P0 * (1 + C[j][1] * (ours[m][t] - 1))
                margins[(j, m)] = (pj - P0 * COST[j][m]) / pj
        for j in COMPS:
            a, b = C[j][2]
            for m in MKTS:
                if (j, m) in pivoting:
                    continue
                squeeze = max(0.0, base_margin[(j, m)] - margins[(j, m)])
                if squeeze <= 0.005 or rng.random() >= sig(a + b * squeeze):
                    continue
                cand = [k for k in MKTS if k != m]
                attr = np.array([margins[(j, k)] + M[k][2] - M[k][8][0] for k in cand])
                w = np.exp(C[j][5] * (attr - attr.max())); w /= w.sum()
                to = cand[rng.choice(len(cand), p=w)]
                frac = rng.uniform(*C[j][3])
                ramp = int(rng.integers(C[j][4][0], C[j][4][1] + 1))
                units = frac * cap[(j, m)]
                active.append(dict(j=j, frm=m, to=to, units=units, ramp=ramp, t0=t, moved=0.0, ret=0.0))
                pivoting.add((j, m))
                moves.append(dict(MOVE_ID=f"{j}-{m}-{months[t].date()}", COMPETITOR=j, COMPETITOR_NAME=CNAMES[j],
                                  FROM_MARKET=m, TO_MARKET=to, START_MONTH=months[t].date(), RAMP_MONTHS=ramp,
                                  FRACTION_OF_CAPACITY=round(frac, 4), UNITS_PER_YEAR=round(units, 0),
                                  MARGIN_AT_START=round(margins[(j, m)], 4), SQUEEZE_AT_START=round(squeeze, 4),
                                  CANDIDATE_ATTRACTIVENESS=str({k: round(float(v), 4) for k, v in zip(cand, attr)})))
        # advance capacity flows (ramp out, then slow return once the squeeze is gone)
        for mv in active:
            j, frm, to = mv["j"], mv["frm"], mv["to"]
            if mv["moved"] < mv["units"] - 1e-6:
                step = mv["units"] / mv["ramp"]
                cap[(j, frm)] -= step; cap[(j, to)] += step; mv["moved"] += step
            else:
                squeeze = max(0.0, base_margin[(j, frm)] - margins[(j, frm)])
                if squeeze < 0.01 and mv["ret"] < mv["moved"]:
                    back = min(C[j][6] * mv["moved"], mv["moved"] - mv["ret"])
                    cap[(j, frm)] += back; cap[(j, to)] -= back; mv["ret"] += back
                if mv["ret"] >= 0.5 * mv["moved"]:
                    pivoting.discard((j, frm))

    hist = pd.DataFrame(rows, columns=["MONTH", "MARKET", "PLAYER", "PRICE", "UNITS", "UNIT_COST", "CAPACITY",
                                       "SHARE", "LEAD_USD_PER_T"])
    mv_df = pd.DataFrame(moves)
    returned = {f"{a['j']}-{a['frm']}-{months[a['t0']].date()}": a["ret"] for a in active}
    mv_df["UNITS_RETURNED"] = mv_df.MOVE_ID.map(returned).round(0)

    markets = pd.DataFrame([dict(MARKET=m, MARKET_NAME=NAMES[m], ANNUAL_UNITS=M[m][0], BASE_PRICE=M[m][1],
                                 LEAD_KG_PER_UNIT=M[m][6], NONLEAD_COST_FRAC=M[m][7],
                                 SHARE_US=M[m][8][0], SHARE_VOLTX=M[m][8][1], SHARE_APEX=M[m][8][2], SHARE_REST=M[m][8][3])
                            for m in MKTS])
    comps = pd.DataFrame([dict(COMPETITOR=j, COMPETITOR_NAME=CNAMES[j],
                               PROFILE={"VOLTX": "Low-cost importer, thin margins, follows price cuts, quick to redeploy capacity",
                                        "APEX": "Premium AGM / lithium, healthy margins, drawn to fast-growing energy storage",
                                        "REST": "Fragmented regional players"}[j]) for j in COMPS])
    # squeeze episodes: what each competitor did in the 12 months after each of our cuts
    ep_rows = []
    hist["MONTH"] = pd.to_datetime(hist["MONTH"])
    for (m, st, ln, d) in episodes:
        t0, t1 = months[st], months[min(T - 1, st + 12)]
        pre = hist[(hist.MARKET == m) & (hist.MONTH < t0) & (hist.MONTH >= months[max(0, st - 3)])]
        post = hist[(hist.MARKET == m) & (hist.MONTH >= t0) & (hist.MONTH <= t1)]
        for j in COMPS:
            c_pre = pre[pre.PLAYER == j].CAPACITY.mean(); c_end = hist[(hist.MARKET == m) & (hist.PLAYER == j) & (hist.MONTH == t1)].CAPACITY.mean()
            mvj = mv_df[(mv_df.COMPETITOR == j) & (mv_df.FROM_MARKET == m) & (pd.to_datetime(mv_df.START_MONTH) >= t0)
                        & (pd.to_datetime(mv_df.START_MONTH) <= t1)]
            ep_rows.append(dict(EPISODE_ID=f"{m}-{t0.date()}", MARKET=m, START_MONTH=t0.date(), CUT_PCT=d, CUT_MONTHS=ln,
                                COMPETITOR=j, CAPACITY_CHANGE_12M=round(c_end / c_pre - 1, 4),
                                PIVOTED=bool(len(mvj)), DESTINATIONS=",".join(mvj.TO_MARKET) if len(mvj) else None,
                                MONTHS_TO_PIVOT=int((pd.to_datetime(mvj.START_MONTH).min() - t0).days // 30) if len(mvj) else None))
        us_pre, us_post = pre[pre.PLAYER == "US"], post[post.PLAYER == "US"]
        for r in ep_rows[-3:]:
            r["OUR_SHARE_CHANGE"] = round(us_post.SHARE.mean() - us_pre.SHARE.mean(), 4)
            r["OUR_MARGIN_RATE_CHANGE"] = round(((us_post.PRICE - us_post.UNIT_COST) / us_post.PRICE).mean()
                                                - ((us_pre.PRICE - us_pre.UNIT_COST) / us_pre.PRICE).mean(), 4)
    eps = pd.DataFrame(ep_rows)
    pe = pd.DataFrame(price_events, columns=["COMPETITOR", "MARKET", "MONTH", "EVENT_TYPE", "PRICE_CHANGE"])
    hist["MONTH"] = hist["MONTH"].dt.date
    for name, df in [("MARKETS", markets), ("COMPETITORS", comps), ("MARKET_HISTORY", hist), ("COMPETITOR_MOVES", mv_df),
                     ("SQUEEZE_EPISODES", eps), ("COMPETITOR_PRICE_EVENTS", pe)]:
        session.write_pandas(df, name, auto_create_table=True, overwrite=True, quote_identifiers=False)
    return f"history rows={len(hist)}, pivots={len(mv_df)}, squeeze episodes={eps.EPISODE_ID.nunique()}, minor cuts={len(pe)}"
$$;

CALL WHATIF_DEMO.SIM.GENERATE_MARKET_DATA(11);
