-- Fit the models the simulator samples from. All coefficients + uncertainty land in *_PARAMS tables.
CREATE OR REPLACE PROCEDURE WHATIF_DEMO.SIM.FIT_MODELS()
RETURNS STRING
LANGUAGE PYTHON
RUNTIME_VERSION = '3.11'
PACKAGES = ('snowflake-snowpark-python', 'numpy', 'pandas', 'scikit-learn')
HANDLER = 'run'
AS
$$
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

def harmonics(doy):
    a = 2 * np.pi * doy / 365.25
    return np.column_stack([np.cos(a), np.sin(a), np.cos(2 * a), np.sin(2 * a)])

def run(session):
    sales = session.table("SALES_HISTORY").to_pandas()
    sales["WEEK_START"] = pd.to_datetime(sales["WEEK_START"])

    # 1) Price elasticity per customer x SKU group: log-log OLS with seasonality
    mid_month_doy = np.array([pd.Timestamp(2026, m, 15).dayofyear for m in range(1, 13)])
    out = []
    for (cid, sku), g in sales.groupby(["CUSTOMER_ID", "SKU_GROUP"]):
        base = g["CONTRACT_BASE_PRICE"].iloc[0]
        X = np.column_stack([np.ones(len(g)), np.log(g.OUR_PRICE / base), np.log(g.COMP_PRICE / base),
                             harmonics(g.WEEK_START.dt.dayofyear.values)])
        y = np.log(g.UNITS.values)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        sigma2 = resid @ resid / (len(y) - X.shape[1])
        se = np.sqrt(np.diag(sigma2 * np.linalg.inv(X.T @ X)))
        season = np.exp(harmonics(mid_month_doy) @ beta[3:])
        last = g.sort_values("WEEK_START").iloc[-1]
        out.append(dict(CUSTOMER_ID=cid, SKU_GROUP=sku,
                        OWN_ELASTICITY=float(beta[1]), OWN_ELASTICITY_SE=float(se[1]),
                        CROSS_ELASTICITY=float(beta[2]), CROSS_ELASTICITY_SE=float(se[2]),
                        BASE_WEEKLY_UNITS=float(np.exp(beta[0])), DEMAND_NOISE_SD=float(np.sqrt(sigma2)),
                        CONTRACT_BASE_PRICE=float(base), CURRENT_PRICE=float(last.OUR_PRICE),
                        CURRENT_COMP_PRICE=float(last.COMP_PRICE),
                        SEASON_IDX=json.dumps([round(float(s), 4) for s in season]),
                        R2=float(1 - resid.var() / y.var())))
    el = pd.DataFrame(out)

    # 2) Churn / share-shift model per customer: logistic on price gap vs competitor
    rev = session.table("ACCOUNT_REVIEWS").to_pandas()
    ch = []
    for cid, g in rev.groupby("CUSTOMER_ID"):
        m = LogisticRegression(C=10.0).fit(g[["PRICE_GAP_PCT"]].values, g.SHARE_SHIFTED.values)
        lost = g.loc[g.SHARE_SHIFTED == 1, "SHARE_LOST"]
        ch.append(dict(CUSTOMER_ID=cid, CHURN_INTERCEPT=float(m.intercept_[0]), CHURN_SLOPE=float(m.coef_[0][0]),
                       SHARE_LOST_MEAN=float(lost.mean()) if len(lost) else 0.25,
                       SHARE_LOST_SD=float(lost.std()) if len(lost) > 1 else 0.05,
                       BASE_SWITCH_RATE=float(g.SHARE_SHIFTED.mean())))
    churn = pd.DataFrame(ch)

    # 3) Competitor response (rule-based from observed reactions, bucketed by our move)
    cr = session.table("COMPETITOR_RESPONSES").to_pandas()
    rs = []
    for move, g in cr.groupby("OUR_PRICE_MOVE"):
        n = len(g)
        rs.append(dict(OUR_PRICE_MOVE=float(move),
                       P_MATCH=float((g.RESPONSE == "MATCH").mean()), P_UNDERCUT=float((g.RESPONSE == "UNDERCUT").mean()),
                       P_HOLD=float((g.RESPONSE == "HOLD").mean()),
                       MATCH_RATIO=float((g.loc[g.RESPONSE == "MATCH", "COMP_PRICE_MOVE"] / move).mean()),
                       UNDERCUT_MOVE_MEAN=float(g.loc[g.RESPONSE == "UNDERCUT", "COMP_PRICE_MOVE"].mean()),
                       UNDERCUT_MOVE_SD=float(g.loc[g.RESPONSE == "UNDERCUT", "COMP_PRICE_MOVE"].std()),
                       LAG_WEEKS_MEAN=float(g.LAG_WEEKS.mean()), N_OBS=n))
    comp = pd.DataFrame(rs).fillna(0.0)

    # 4) LME lead: GBM params from daily log returns
    lme = session.table("LME_LEAD_PRICES").sort("PRICE_DATE").to_pandas()
    r = np.diff(np.log(lme.LEAD_USD_PER_T.values))
    lead = pd.DataFrame([dict(SPOT_USD_PER_T=float(lme.LEAD_USD_PER_T.iloc[-1]),
                              ANNUAL_DRIFT=float(r.mean() * 252), ANNUAL_VOL=float(r.std() * np.sqrt(252)),
                              AS_OF=str(lme.PRICE_DATE.iloc[-1]))])

    for name, df in [("ELASTICITY_PARAMS", el), ("CHURN_PARAMS", churn),
                     ("COMP_RESPONSE_PARAMS", comp), ("LEAD_PARAMS", lead)]:
        session.write_pandas(df, name, auto_create_table=True, overwrite=True, quote_identifiers=False)
    return f"elasticity models={len(el)} (median R2={el.R2.median():.2f}), churn models={len(churn)}, lead vol={lead.ANNUAL_VOL[0]:.2f}"
$$;

CALL WHATIF_DEMO.SIM.FIT_MODELS();
