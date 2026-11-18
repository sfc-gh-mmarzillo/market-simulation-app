-- Fit market-level models from MARKET_HISTORY / COMPETITOR_MOVES / COMPETITOR_PRICE_EVENTS.
-- Writes MARKET_PARAMS (6 demand models), SHARE_PARAMS (6 share-response models),
-- COMPETITOR_PARAMS (3 competitors x {cost/margin, price-follow, pivot hazard, pivot size/ramp/return, destination choice, minor cuts}).
CREATE OR REPLACE PROCEDURE WHATIF_DEMO.SIM.FIT_MARKET_MODELS()
RETURNS STRING
LANGUAGE PYTHON
RUNTIME_VERSION = '3.11'
PACKAGES = ('snowflake-snowpark-python', 'numpy', 'pandas', 'scikit-learn')
HANDLER = 'run'
AS
$$
import ast
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

def ols(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    s2 = resid @ resid / max(len(y) - X.shape[1], 1)
    se = np.sqrt(np.diag(s2 * np.linalg.pinv(X.T @ X)))
    return beta, se, float(np.sqrt(s2)), float(1 - resid.var() / y.var())

def run(session):
    h = session.table("MARKET_HISTORY").to_pandas()
    h["MONTH"] = pd.to_datetime(h["MONTH"])
    mk = session.table("MARKETS").to_pandas().set_index("MARKET")
    mv = session.table("COMPETITOR_MOVES").to_pandas()
    pe = session.table("COMPETITOR_PRICE_EVENTS").to_pandas()
    t0 = h.MONTH.min()
    T = h.MONTH.nunique()

    # 1) market demand + 2) share response, per market
    mp, sp = [], []
    for m, g in h.groupby("MARKET"):
        P0 = mk.loc[m, "BASE_PRICE"]
        tot = g.groupby("MONTH").apply(lambda d: pd.Series({
            "units": d.UNITS.sum(), "pidx": (d.UNITS * d.PRICE).sum() / d.UNITS.sum() / P0}))
        tt = ((tot.index - t0).days / 365.25).values
        mon = tot.index.month.values
        X = np.column_stack([np.ones(len(tot)), np.log(tot.pidx), tt, np.cos(2 * np.pi * (mon - 1) / 12),
                             np.sin(2 * np.pi * (mon - 1) / 12)])
        b, se, sd, r2 = ols(X, np.log(tot.units.values))
        season = [float(np.exp(b[3] * np.cos(2 * np.pi * k / 12) + b[4] * np.sin(2 * np.pi * k / 12))) for k in range(12)]
        mp.append(dict(MARKET=m, MARKET_NAME=mk.loc[m, "MARKET_NAME"], ELASTICITY=float(b[1]), ELASTICITY_SE=float(se[1]),
                       GROWTH=float(np.exp(b[2]) - 1), GROWTH_SE=float(se[2]), NOISE_SD=sd, R2=r2,
                       SEASON_IDX=json.dumps([round(s, 4) for s in season]),
                       CURRENT_MONTHLY_UNITS=float(tot.units.iloc[-6:].mean() / np.mean([season[(x - 1) % 12] for x in mon[-6:]]))))
        us = g[g.PLAYER == "US"].set_index("MONTH").sort_index()
        cp = g[g.PLAYER != "US"]
        comp = cp.groupby("MONTH").apply(lambda d: pd.Series({
            "p": (d.UNITS * d.PRICE).sum() / d.UNITS.sum(), "cap": d.CAPACITY.sum()}))
        s = us.SHARE.values
        Xs = np.column_stack([np.ones(len(s)), np.log(us.PRICE.values / comp.p.values), np.log(comp.cap.values / comp.cap.iloc[0])])
        bs, ses, sds, r2s = ols(Xs, np.log(s / (1 - s)))
        sp.append(dict(MARKET=m, PRICE_SENS=float(-bs[1]), PRICE_SENS_SE=float(ses[1]), CAPACITY_SENS=float(-bs[2]),
                       CAPACITY_SENS_SE=float(ses[2]), NOISE_SD=sds, R2=r2s,
                       CURRENT_SHARE=float(us.SHARE.iloc[-3:].mean())))

    # 3) competitor models
    cpar = []
    for j, g in h[h.PLAYER != "US"].groupby("PLAYER"):
        cost = {m: float((d.UNIT_COST / mk.loc[m, "BASE_PRICE"]).mean()) for m, d in g.groupby("MARKET")}
        cost_sd = {m: float((d.UNIT_COST / mk.loc[m, "BASE_PRICE"]).std()) for m, d in g.groupby("MARKET")}
        # price follow: competitor price change vs our price change (excluding minor-cut months)
        us = h[h.PLAYER == "US"][["MONTH", "MARKET", "PRICE"]].rename(columns={"PRICE": "P_US"})
        d = g.merge(us, on=["MONTH", "MARKET"])
        d["P0"] = d.MARKET.map(mk.BASE_PRICE)
        cuts = pe[pe.COMPETITOR == j]
        cut_keys = set(zip(cuts.MARKET, pd.to_datetime(cuts.MONTH)))
        dd = d[[(a, b) not in cut_keys for a, b in zip(d.MARKET, d.MONTH)]]
        x, y = (dd.P_US / dd.P0 - 1).values, (dd.PRICE / dd.P0 - 1).values
        rho = float((x @ y) / (x @ x))
        # pivot hazard on margin squeeze (panel of market-months, excluding 12 months after each pivot start)
        d["MARGIN"] = (d.PRICE - d.MARKET.map(cost) * d.P0) / d.PRICE
        base = d[d.MONTH < t0 + pd.DateOffset(months=6)].groupby("MARKET").MARGIN.median()
        d["SQUEEZE"] = (d.MARKET.map(base) - d.MARGIN).clip(lower=0)
        mj = mv[mv.COMPETITOR == j].copy()
        mj["START_MONTH"] = pd.to_datetime(mj.START_MONTH)
        start = set(zip(mj.FROM_MARKET, mj.START_MONTH))
        d["Y"] = [int((a, b) in start) for a, b in zip(d.MARKET, d.MONTH)]
        excl = np.zeros(len(d), bool)
        for r in mj.itertuples():
            # once a competitor is pivoting out of a market it is no longer "at risk" of starting a pivot there
            excl |= (d.MARKET.values == r.FROM_MARKET) & (d.MONTH.values > np.datetime64(r.START_MONTH))
        minor_cut = np.array([(a, b) in cut_keys for a, b in zip(d.MARKET, d.MONTH)])
        panel = d[~excl & ~minor_cut]   # their own minor cuts are not a squeeze from us
        lr = LogisticRegression(C=1e6, max_iter=1000).fit(panel[["SQUEEZE"]].values, panel.Y.values)
        p = lr.predict_proba(panel[["SQUEEZE"]].values)[:, 1]
        W = p * (1 - p)
        Xh = np.column_stack([np.ones(len(panel)), panel.SQUEEZE.values])
        cov = np.linalg.pinv(Xh.T @ (Xh * W[:, None]))
        # destination choice: softmax temperature theta by maximum likelihood over observed destinations
        thetas = np.linspace(0, 20, 81)
        ll = np.zeros_like(thetas)
        for r in mj.itertuples():
            attr = ast.literal_eval(r.CANDIDATE_ATTRACTIVENESS)
            keys = list(attr); a = np.array([attr[k] for k in keys])
            for i, th in enumerate(thetas):
                w = np.exp(th * (a - a.max())); w /= w.sum()
                ll[i] += np.log(w[keys.index(r.TO_MARKET)] + 1e-12)
        theta = float(thetas[ll.argmax()]) if len(mj) else 5.0
        months_after = [(h.MONTH.max() - (r.START_MONTH + pd.DateOffset(months=int(r.RAMP_MONTHS)))).days / 30.4
                        for r in mj.itertuples()]
        ret = [r.UNITS_RETURNED / r.UNITS_PER_YEAR / max(ma, 1) for r, ma in zip(mj.itertuples(), months_after) if ma > 1]
        cuts_n = len(cuts)
        cpar.append(dict(COMPETITOR=j, COST_FRAC=json.dumps(cost), COST_FRAC_SD=json.dumps(cost_sd), PRICE_FOLLOW=rho,
                         HAZARD_INTERCEPT=float(lr.intercept_[0]), HAZARD_SLOPE=float(lr.coef_[0][0]),
                         HAZARD_INTERCEPT_SE=float(np.sqrt(cov[0, 0])), HAZARD_SLOPE_SE=float(np.sqrt(cov[1, 1])),
                         N_PIVOTS=int(len(mj)), N_PANEL=int(len(panel)),
                         PIVOT_FRACTION_MEAN=float(mj.FRACTION_OF_CAPACITY.mean()) if len(mj) else 0.3,
                         PIVOT_FRACTION_SD=float(mj.FRACTION_OF_CAPACITY.std()) if len(mj) > 1 else 0.1,
                         RAMP_MONTHS_MEAN=float(mj.RAMP_MONTHS.mean()) if len(mj) else 5.0,
                         RETURN_RATE=float(np.mean(ret)) if ret else 0.01, DEST_THETA=theta,
                         DEST_COUNTS=json.dumps(mj.TO_MARKET.value_counts().to_dict()),
                         MINOR_CUT_PROB=float(cuts_n / (T * 6)), MINOR_CUT_MEAN=float(cuts.PRICE_CHANGE.mean()) if cuts_n else -0.04,
                         MINOR_CUT_SD=float(cuts.PRICE_CHANGE.std()) if cuts_n > 1 else 0.015))
    for name, df in [("MARKET_PARAMS", pd.DataFrame(mp)), ("SHARE_PARAMS", pd.DataFrame(sp)),
                     ("COMPETITOR_PARAMS", pd.DataFrame(cpar))]:
        session.write_pandas(df, name, auto_create_table=True, overwrite=True, quote_identifiers=False)
    c = pd.DataFrame(cpar)
    return ("demand models=6, share models=6, competitor models=3 | hazard slopes " +
            ", ".join(f"{r.COMPETITOR}:{r.HAZARD_SLOPE:.0f}±{r.HAZARD_SLOPE_SE:.0f} (n={r.N_PIVOTS})" for r in c.itertuples()) +
            " | follow " + ", ".join(f"{r.COMPETITOR}:{r.PRICE_FOLLOW:.2f}" for r in c.itertuples()) +
            " | theta " + ", ".join(f"{r.COMPETITOR}:{r.DEST_THETA:.1f}" for r in c.itertuples()))
$$;

CALL WHATIF_DEMO.SIM.FIT_MARKET_MODELS();
