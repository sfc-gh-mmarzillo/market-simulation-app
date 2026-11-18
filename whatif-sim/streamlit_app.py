"""Voltara Energy — Market Strategy Simulator (starve → pivot → adapt).

The Cortex Agent builds every scenario: it turns a question into a pricing strategy, runs the Monte Carlo strategy
engine (strategy_engine.py behind RUN_STRATEGY / RUN_STRATEGY_SWEEP), and explains the results. This app only displays
what the engine computed: each scenario's configuration plus volume-flow and outcome plots. No number here comes from an LLM.
"""
import json
import os

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Voltara Market Strategy Simulator", page_icon=":material/battery_charging_full:", layout="wide")

DB = "WHATIF_DEMO.SIM"
AGENT = f"{DB}.WHATIF_AGENT"
MKTS = ["AFTERMARKET", "OEM", "COMMERCIAL", "MARINE", "ENERGY", "OTHER"]
MKT_NAMES = {"AFTERMARKET": "Auto aftermarket", "OEM": "Auto OEM", "COMMERCIAL": "Commercial / HD", "MARINE": "Marine & powersports",
             "ENERGY": "Energy storage", "OTHER": "Other"}
COMP_NAMES = {"VOLTX": "Volt-X Imports", "APEX": "Apex Power", "REST": "Rest of market", "US": "Voltara (us)"}
COMP_COLORS = {"VOLTX": "#d62728", "APEX": "#9467bd", "REST": "#8c8c8c", "US": "#1f77b4"}
ADAPT_TEXT = {"restore_after": "Restore price after the cut period", "restore_on_exit": "Hold the cut until the competitor exits, then restore",
              "hold": "Hold the cut for the whole horizon", "follow": "Restore on exit + defend markets the competitor moves into"}
SAMPLE_QS = [
    "If we cut Auto aftermarket prices 12% for 6 months to squeeze Volt-X, where does Volt-X go and how should we adapt?",
    "What cut depth and duration in aftermarket pays back best within our risk policy?",
    "Compare restoring price after 6 months vs waiting until Volt-X exits vs following them into their new market.",
    "What is randomized in the simulation and how confident are the pivot models?",
]

st.title("Market strategy simulator")
st.caption("Voltara Energy (demo) · cut price to squeeze a competitor → it pivots capacity to other markets → we adapt · "
           "the agent builds and runs every scenario; the Monte Carlo engine computes every number")

conn = st.connection("snowflake", ttl=os.getenv("SNOWFLAKE_CONNECTION_TTL"))
ss = st.session_state
ss.setdefault("chat", [])
ss.setdefault("scenarios", [])   # ordered list of {sid, question}
ss.setdefault("sweeps", [])      # list of {sweep_id, question}


# ============================================================================================ data access
@st.cache_data(ttl=3600, show_spinner=False)
def load_strategies(sids: tuple) -> dict:
    if not sids:
        return {}
    df = conn.session().sql(f"""SELECT SCENARIO_ID, RUN_TS, LABEL, QUESTION, SPEC, CONFIG, OUTPUT, SWEEP_ID
                                FROM {DB}.STRATEGY_RESULTS WHERE ARRAY_CONTAINS(SCENARIO_ID::VARIANT, PARSE_JSON(?))""",
                            params=[json.dumps(list(sids))]).to_pandas()
    out = {}
    for r in df.to_dict("records"):
        for k in ("SPEC", "CONFIG", "OUTPUT"):
            r[k] = json.loads(r[k]) if isinstance(r[k], str) else r[k]
        out[r["SCENARIO_ID"]] = r
    return out


@st.cache_data(ttl=3600, show_spinner=False)
def load_sweep(sweep_id: str) -> pd.DataFrame:
    return conn.session().sql(f"""SELECT SCENARIO_ID, CUT_PCT, CUT_MONTHS, MARGIN_DELTA_P10, MARGIN_DELTA_P50, MARGIN_DELTA_P90,
                                         PROB_LOSS, PROB_TARGET_EXIT, PAYBACK_MONTH_P50, TARGET_MARKET, TARGET_COMPETITOR, ADAPT_RULE
                                  FROM {DB}.STRATEGY_RESULTS WHERE SWEEP_ID = ? ORDER BY CUT_PCT, CUT_MONTHS""",
                              params=[sweep_id]).to_pandas()


@st.cache_data(ttl=600, show_spinner=False)
def recent_log() -> pd.DataFrame:
    return conn.query(f"""SELECT SCENARIO_ID, RUN_TS, LABEL, ROUND(MARGIN_DELTA_P50/1e6, 1) AS DELTA_P50_M, PROB_LOSS
                          FROM {DB}.STRATEGY_RESULTS WHERE SWEEP_ID IS NULL AND LABEL NOT LIKE '[stress]%'
                          AND LABEL NOT LIKE '[test]%' ORDER BY RUN_TS DESC LIMIT 100""")


@st.cache_data(ttl=3600, show_spinner=False)
def model_card() -> dict:
    raw = conn.session().call(f"{DB}.DESCRIBE_MARKET_MODEL")
    return json.loads(raw) if isinstance(raw, str) else raw


@st.cache_data(ttl=3600, show_spinner=False)
def squeeze_episodes() -> pd.DataFrame:
    return conn.query(f"SELECT * FROM {DB}.SQUEEZE_EPISODES ORDER BY START_MONTH, COMPETITOR")


def add_scenarios(sids, question):
    have = {x["sid"] for x in ss.scenarios}
    for sid in sids:
        if sid and sid not in have:
            ss.scenarios.insert(0, dict(sid=sid, question=question))


# ============================================================================================ charts
def money(v):
    return f"${v / 1e6:,.1f}M"


def flow_chart(flows: list, title=""):
    """Sankey-style bipartite flow: competitor capacity moving from market (left) to market (right)."""
    f = pd.DataFrame([x for x in flows if x["probability"] >= 0.05 and x["expected_units_per_year"] > 1000])
    if f.empty:
        return None
    pos = {m: i for i, m in enumerate(MKTS)}
    rows = []
    for i, r in f.reset_index(drop=True).iterrows():
        for side, m in ((0, r.from_market), (1, r.to_market)):
            rows.append(dict(link=i, x=side, y=-pos[m], competitor=COMP_NAMES[r.competitor], comp=r.competitor,
                             units=r.expected_units_per_year, prob=r.probability,
                             label=f"{COMP_NAMES[r.competitor]}: {MKT_NAMES[r.from_market]} → {MKT_NAMES[r.to_market]}"))
    links = pd.DataFrame(rows)
    nodes = pd.DataFrame([dict(x=s, y=-pos[m], name=MKT_NAMES[m]) for m in MKTS for s in (0, 1)])
    width_scale = alt.Scale(domain=[0, max(f.expected_units_per_year.max(), 1)], range=[1.5, 26])
    line = alt.Chart(links).mark_line(opacity=0.55, strokeCap="round", interpolate="monotone").encode(
        x=alt.X("x:Q", axis=None, scale=alt.Scale(domain=[-0.45, 1.45])),
        y=alt.Y("y:Q", axis=None, scale=alt.Scale(domain=[-len(MKTS) + 0.5, 0.5])),
        detail="link:N", strokeWidth=alt.StrokeWidth("units:Q", scale=width_scale, legend=None),
        color=alt.Color("competitor:N", scale=alt.Scale(domain=[COMP_NAMES[c] for c in ["VOLTX", "APEX", "REST"]],
                                                        range=[COMP_COLORS[c] for c in ["VOLTX", "APEX", "REST"]]),
                        legend=alt.Legend(orient="bottom", title=None)),
        tooltip=["label:N", alt.Tooltip("units:Q", format=",.0f", title="expected units / yr"),
                 alt.Tooltip("prob:Q", format=".0%", title="probability it happens")])
    pts = alt.Chart(nodes).mark_circle(size=120, color="#333").encode(x="x:Q", y="y:Q")
    left = alt.Chart(nodes[nodes.x == 0]).mark_text(align="right", dx=-10, fontSize=12).encode(x="x:Q", y="y:Q", text="name:N")
    right = alt.Chart(nodes[nodes.x == 1]).mark_text(align="left", dx=10, fontSize=12).encode(x="x:Q", y="y:Q", text="name:N")
    return (line + pts + left + right).properties(height=260, title=alt.TitleParams(title, fontSize=13, anchor="start"))


def volume_by_market_chart(by_market: list, target_comp: str):
    rows = []
    for b in by_market:
        rows.append(dict(market=MKT_NAMES[b["market"]], who="Voltara (us)", units=b["our_volume_delta"][1]))
        rows.append(dict(market=MKT_NAMES[b["market"]], who=COMP_NAMES[target_comp], units=b["target_comp_volume_delta"]))
    d = pd.DataFrame(rows)
    return alt.Chart(d).mark_bar().encode(
        y=alt.Y("market:N", sort=[MKT_NAMES[m] for m in MKTS], title=None),
        x=alt.X("units:Q", title="Volume change vs baseline over horizon (units, P50)", axis=alt.Axis(format="~s")),
        yOffset="who:N",
        color=alt.Color("who:N", scale=alt.Scale(domain=["Voltara (us)", COMP_NAMES[target_comp]],
                                                range=[COMP_COLORS["US"], COMP_COLORS[target_comp]]),
                        legend=alt.Legend(orient="bottom", title=None)),
        tooltip=["market", "who", alt.Tooltip("units:Q", format=",.0f")]).properties(height=260)


def capacity_chart(cap_rows: list, target_market: str, target_comp: str):
    d = pd.DataFrame(cap_rows)
    d["market_name"] = d.market.map(MKT_NAMES)
    area = alt.Chart(d).mark_area(opacity=0.85).encode(
        x=alt.X("month:Q", title="Month"), y=alt.Y("p50:Q", stack=True, title=f"{COMP_NAMES[target_comp]} capacity (units/yr, P50)",
                                                    axis=alt.Axis(format="~s")),
        color=alt.Color("market_name:N", sort=[MKT_NAMES[m] for m in MKTS], legend=alt.Legend(orient="bottom", title=None, columns=3)),
        order=alt.Order("market_name:N"), tooltip=["month", "market_name", alt.Tooltip("p50:Q", format=",.0f")])
    return area.properties(height=240)


def target_capacity_band(cap_rows: list, target_market: str, target_comp: str):
    t = pd.DataFrame([r for r in cap_rows if r["market"] == target_market])
    base = alt.Chart(t).encode(x=alt.X("month:Q", title="Month"))
    return (base.mark_area(opacity=0.25, color=COMP_COLORS[target_comp]).encode(
                y=alt.Y("p10:Q", title=f"{COMP_NAMES[target_comp]} capacity in {MKT_NAMES[target_market]}", axis=alt.Axis(format="~s")),
                y2="p90:Q")
            + base.mark_line(color=COMP_COLORS[target_comp], strokeWidth=2.5).encode(y="p50:Q")
            + base.mark_line(color="#888", strokeDash=[5, 4]).encode(y="baseline_p50:Q")).properties(height=240)


def cum_margin_chart(curves: dict):
    """curves: label -> list of {month, p10..p90}. Cumulative margin change vs baseline ($M)."""
    rows = []
    for label, cur in curves.items():
        for r in cur:
            rows.append(dict(scenario=label, month=r["month"], **{k: r[k] / 1e6 for k in ("p10", "p25", "p50", "p75", "p90")}))
    d = pd.DataFrame(rows)
    color = alt.Color("scenario:N", legend=alt.Legend(orient="bottom", title=None, columns=1, labelLimit=600))
    base = alt.Chart(d).encode(x=alt.X("month:Q", title="Month"), color=color)
    chart = (base.mark_area(opacity=0.12).encode(y=alt.Y("p10:Q", title="Cumulative margin vs baseline ($M)"), y2="p90:Q")
             + base.mark_area(opacity=0.22).encode(y="p25:Q", y2="p75:Q")
             + base.mark_line(strokeWidth=2.5).encode(y="p50:Q", tooltip=["scenario", "month", alt.Tooltip("p50:Q", format=",.1f")])
             + alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="#333", strokeDash=[3, 3]).encode(y="y:Q"))
    return chart.properties(height=280)


def market_margin_chart(by_market: list):
    d = pd.DataFrame([dict(market=MKT_NAMES[b["market"]], p10=b["margin_delta"][0] / 1e6, p50=b["margin_delta"][1] / 1e6,
                           p90=b["margin_delta"][2] / 1e6) for b in by_market])
    base = alt.Chart(d).encode(y=alt.Y("market:N", sort=[MKT_NAMES[m] for m in MKTS], title=None))
    return (base.mark_bar(color="#1f77b4", opacity=0.75).encode(
                x=alt.X("p50:Q", title="Our margin change vs baseline by market ($M, P50 with P10–P90)"),
                color=alt.condition("datum.p50 < 0", alt.value("#d62728"), alt.value("#2ca02c")),
                tooltip=["market", alt.Tooltip("p10:Q", format=",.1f"), alt.Tooltip("p50:Q", format=",.1f"),
                         alt.Tooltip("p90:Q", format=",.1f")])
            + base.mark_rule(color="#333").encode(x="p10:Q", x2="p90:Q")).properties(height=240)


def price_path_chart(rows: list):
    d = pd.DataFrame(rows)
    d["market_name"] = d.market.map(MKT_NAMES)
    return alt.Chart(d).mark_line(strokeWidth=2, interpolate="step-after").encode(
        x=alt.X("month:Q", title="Month"), y=alt.Y("p50:Q", title="Our price vs today (P50)", axis=alt.Axis(format="%"),
                                                    scale=alt.Scale(zero=False)),
        color=alt.Color("market_name:N", legend=alt.Legend(orient="bottom", title=None))).properties(height=160)


def sweep_heatmap(df: pd.DataFrame, risk=0.25):
    d = df.copy()
    d["cut"] = (d.CUT_PCT * 100).round(0).astype(int).astype(str) + "%"
    d["p50"] = d.MARGIN_DELTA_P50 / 1e6
    d["txt"] = d.p50.map(lambda v: f"{v:+.1f}M") + "\n" + d.PROB_LOSS.map(lambda v: f"P(loss) {v:.0%}")
    ok = d[d.PROB_LOSS <= risk]
    best = ok.loc[ok.p50.idxmax()] if not ok.empty else None
    base = alt.Chart(d).encode(x=alt.X("cut:N", title="Cut depth", sort=sorted(d.cut.unique(), key=lambda s: int(s[:-1]))),
                               y=alt.Y("CUT_MONTHS:O", title="Cut duration (months)"))
    heat = base.mark_rect().encode(color=alt.Color("p50:Q", scale=alt.Scale(scheme="redyellowgreen", domainMid=0),
                                                   title="P50 $M"),
                                   tooltip=["cut", "CUT_MONTHS", alt.Tooltip("p50:Q", format="+.1f"),
                                            alt.Tooltip("PROB_LOSS:Q", format=".0%"), alt.Tooltip("PROB_TARGET_EXIT:Q", format=".0%"),
                                            "PAYBACK_MONTH_P50"])
    text = base.mark_text(fontSize=11, lineBreak="\n").encode(text="txt:N")
    layers = heat + text
    if best is not None:
        layers += alt.Chart(pd.DataFrame([best])).mark_rect(fill=None, stroke="#111", strokeWidth=3).encode(
            x=alt.X("cut:N", sort=sorted(d.cut.unique(), key=lambda s: int(s[:-1]))), y="CUT_MONTHS:O")
    return layers.properties(height=60 * d.CUT_MONTHS.nunique() + 40), best


# ============================================================================================ cards
def config_table(r: dict):
    s, c = r["SPEC"], r["CONFIG"]
    rows = [("Target market", MKT_NAMES.get(s["target_market"], s["target_market"])),
            ("Target competitor", COMP_NAMES.get(s["target_competitor"], s["target_competitor"])),
            ("Our price cut", f"{s['cut_pct']:.0%} for {s['cut_months']} months (ramp-in {s['ramp_in_months']} mo)"),
            ("How we adapt", ADAPT_TEXT.get(s["adapt"], s["adapt"]))]
    if s["adapt"] in ("restore_on_exit", "follow"):
        rows.append(("Exit trigger", f"{s['exit_threshold']:.0%} of its capacity has left · max cut {s['max_cut_months']} mo"))
    if s["adapt"] == "follow":
        rows.append(("Defend pivot markets", f"{s['follow_cut_pct']:.0%} cut for {s['follow_months']} mo"))
    rows.append(("Restore ramp", f"{s['restore_months']} months"))
    if s.get("lead_shock"):
        rows.append(("LME lead shock", f"{s['lead_shock']:+.0%}"))
    off = [k for k, v in c["uncertainty"].items() if not v]
    rows += [("Monte Carlo", f"{c['n_runs']:,} runs · {c['horizon_months']} months · seed {c['seed']}"),
             ("Randomized", "all 9 sources" if not off else f"all except: {', '.join(off)}")]
    if c.get("disable_pivots"):
        rows.append(("Stress setting", "competitors never pivot"))
    if c.get("force_destination"):
        rows.append(("Stress setting", f"pivot destination forced → {MKT_NAMES.get(c['force_destination'], c['force_destination'])}"))
    return pd.DataFrame(rows, columns=["setting", "value"])


def kpis(sm: dict, H: int):
    k = st.columns(5)
    k[0].metric(f"Margin vs baseline ({H} mo, P50)", money(sm["margin_delta"][1]),
                f"P10 {money(sm['margin_delta'][0])} · P90 {money(sm['margin_delta'][2])}", delta_color="off")
    k[1].metric("P(strategy loses money)", f"{sm['prob_loss']:.0%}")
    k[2].metric("Payback month (P50)", "—" if sm["payback_month_p50"] is None else f"{sm['payback_month_p50']:.0f}",
                f"P(payback | dip) {sm['prob_payback']:.0%}", delta_color="off")
    k[3].metric("P(target competitor exits)", f"{sm['prob_target_exit']:.0%}",
                f"pivots in {sm['prob_target_pivots']:.0%} of paths", delta_color="off")
    k[4].metric("Our share in target market (end)", f"{sm['our_share_target_end']:.1%}",
                f"baseline {sm['our_share_target_end_baseline']:.1%}", delta_color="off")


def render_card(r: dict, key: str, show_question=True):
    s, o = r["SPEC"], r["OUTPUT"]
    if show_question and r.get("QUESTION"):
        st.caption(f"Built by the agent for: “{r['QUESTION']}”")
    kpis(o["summary"], r["CONFIG"]["horizon_months"])
    c1, c2 = st.columns([1, 1.6])
    with c1:
        st.markdown("**Configuration**")
        st.dataframe(config_table(r), hide_index=True, width="stretch", key=f"cfg_{key}")
        st.caption(f"`{r['SCENARIO_ID'][:8]}` · {o.get('config_text', '')}")
    with c2:
        fc = flow_chart(o["flows"], "Where competitor capacity flows (line width = expected units/yr)")
        if fc is not None:
            st.altair_chart(fc, width="stretch")
        else:
            st.info("No material competitor capacity flows in this scenario.", icon=":material/info:")
    g1, g2 = st.columns(2)
    with g1:
        st.markdown("**Cumulative margin vs baseline** — the cost of the squeeze, then the payback")
        st.altair_chart(cum_margin_chart({r["LABEL"][:80]: o["cum_curve"]}), width="stretch")
        st.altair_chart(price_path_chart(o["price_path"]), width="stretch")
    with g2:
        st.markdown("**Volume shift by market** — what we gain where the competitor leaves, and lose where it lands")
        st.altair_chart(volume_by_market_chart(o["by_market"], s["target_competitor"]), width="stretch")
    g3, g4 = st.columns(2)
    with g3:
        st.markdown(f"**{COMP_NAMES[s['target_competitor']]} capacity by market over time** (P50, stacked)")
        st.altair_chart(capacity_chart(o["target_capacity"], s["target_market"], s["target_competitor"]), width="stretch")
    with g4:
        st.markdown(f"**{COMP_NAMES[s['target_competitor']]} capacity in {MKT_NAMES[s['target_market']]}** — P10–P90 band, dashed = no strategy")
        st.altair_chart(target_capacity_band(o["target_capacity"], s["target_market"], s["target_competitor"]), width="stretch")
    g5, g6 = st.columns(2)
    with g5:
        st.altair_chart(market_margin_chart(o["by_market"]), width="stretch")
    with g6:
        dr = pd.DataFrame(o["drivers"])
        st.altair_chart(alt.Chart(dr).mark_bar(color="#ff7f0e").encode(
            y=alt.Y("driver:N", sort="-x", title=None), x=alt.X("variance_share:Q", title="Share of outcome variance",
                                                               axis=alt.Axis(format="%")),
            tooltip=["driver", alt.Tooltip("variance_share:Q", format=".0%")]).properties(height=240), width="stretch")


# ============================================================================================ agent plumbing
def ask_agent(history: list) -> dict:
    body = json.dumps({"messages": history})
    raw = conn.session().sql("SELECT SNOWFLAKE.CORTEX.DATA_AGENT_RUN(?, ?)", params=[AGENT, body]).collect()[0][0]
    return json.loads(raw)


def tool_payload(c: dict) -> dict:
    for item in c["tool_result"].get("content", []):
        r = (item.get("json") or {}).get("result")
        if isinstance(r, dict):
            return r
        if isinstance(r, str):
            try:
                return json.loads(r)
            except ValueError:
                pass
    return {}


def parse_agent(resp: dict) -> dict:
    out = dict(steps=[], runs=[], sweeps=[], stress=[], outlooks=[], precedents=[], model=False)
    status = {}
    for c in resp.get("content", []):
        if c.get("type") != "tool_result":
            continue
        tr = c["tool_result"]; status[tr.get("tool_use_id")] = tr.get("status")
        if tr.get("status") != "success":
            continue
        name, p = tr.get("name"), tool_payload(c)
        if name == "run_strategy" and p.get("scenario_id"):
            out["runs"].append(p["scenario_id"])
        elif name == "run_strategy_sweep" and p.get("sweep_id"):
            out["sweeps"].append(p["sweep_id"])
        elif name == "explain_strategy" and p.get("scenario_id"):
            out["stress"].append(dict(original=p["scenario_id"], tests=p.get("stress_tests", [])))
        elif name == "market_outlook" and p.get("market"):
            out["outlooks"].append(p)
        elif name == "find_squeeze_precedents" and p.get("episodes") is not None:
            out["precedents"].append(p)
        elif name == "describe_model":
            out["model"] = True
    for c in resp.get("content", []):
        if c.get("type") == "tool_use":
            tu = c["tool_use"]
            out["steps"].append(dict(tool=tu.get("name"), input=tu.get("input"), status=status.get(tu.get("tool_use_id"), "")))
    texts = [c["text"] for c in resp.get("content", []) if c.get("type") == "text"]
    out["text"] = texts[-1] if texts else "_The agent returned no text._"
    return out


def render_agent_visuals(msg: dict, idx: int):
    data = load_strategies(tuple(msg.get("runs", [])))
    runs = [data[s] for s in msg.get("runs", []) if s in data]
    if len(runs) > 1:
        st.markdown("**Options the agent simulated** — cumulative margin vs baseline")
        st.altair_chart(cum_margin_chart({f"{i + 1}. {r['LABEL'][:70]}": r["OUTPUT"]["cum_curve"] for i, r in enumerate(runs)}),
                        width="stretch")
    if runs:
        tabs = st.tabs([f"{i + 1}. {r['LABEL'][:40]}" for i, r in enumerate(runs)]) if len(runs) > 1 else [st.container()]
        for i, (t, r) in enumerate(zip(tabs, runs)):
            with t, st.container(border=True):
                render_card(r, f"a{idx}_{i}", show_question=False)
    for j, sw in enumerate(msg.get("sweeps", [])):
        df = load_sweep(sw)
        if not df.empty:
            st.markdown(f"**Strategy sweep** — {MKT_NAMES.get(df.TARGET_MARKET.iloc[0])} vs "
                        f"{COMP_NAMES.get(df.TARGET_COMPETITOR.iloc[0])}, {ADAPT_TEXT.get(df.ADAPT_RULE.iloc[0], '')}")
            ch, best = sweep_heatmap(df)
            st.altair_chart(ch, width="stretch")
            st.caption("Cell = P50 margin vs baseline and P(loss); outlined = best within the 25% loss-risk policy.")
    for j, sx in enumerate(msg.get("stress", [])):
        if sx["tests"]:
            d = pd.DataFrame([dict(label=t["what_if"], p10=_num(t["range_p10_p90"].split(" to ")[0]),
                                   p50=_num(t["margin_change_p50"]), p90=_num(t["range_p10_p90"].split(" to ")[1]))
                              for t in sx["tests"]])
            st.markdown("**Stress tests the agent ran** — margin vs baseline ($M), P10–P90 with P50")
            base = alt.Chart(d).encode(y=alt.Y("label:N", title=None, sort=None))
            st.altair_chart((base.mark_bar(height=12, opacity=0.35, color="#ff7f0e").encode(x=alt.X("p10:Q", title="$M"), x2="p90:Q")
                             + base.mark_point(filled=True, size=80, color="#ff7f0e").encode(x="p50:Q")
                             + alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(strokeDash=[3, 3]).encode(x="x:Q"))
                            .properties(height=32 * len(d) + 30), width="stretch")
    for p in msg.get("precedents", []):
        with st.expander(f"Past squeeze episodes the agent used ({p['n_episodes']})", icon=":material/history_edu:"):
            st.dataframe(pd.DataFrame(p["episodes"]), hide_index=True, width="stretch")
    for o in msg.get("outlooks", []):
        with st.expander(f"Market outlook — {o.get('market_name', o['market'])}", icon=":material/insights:"):
            st.dataframe(pd.DataFrame(o["players"]).T.rename(index=COMP_NAMES), width="stretch")
    if msg.get("model"):
        st.caption(":material/model_training: The agent used the model card — see the **Model & assumptions** tab.")


def _num(txt: str) -> float:
    return float(txt.replace("$", "").replace("M", "").replace(",", "").strip())


def render_steps(steps: list):
    if steps:
        with st.expander(f"Agent steps — {len(steps)} tool call(s); every number comes from these", icon=":material/build:"):
            for s in steps:
                st.markdown(f"**`{s['tool']}`** · {s['status']}")
                st.json(s["input"], expanded=False)


# ============================================================================================ tabs
tab_agent, tab_scen, tab_model = st.tabs([":material/smart_toy: Ask the agent", ":material/dashboard: Scenarios",
                                          ":material/model_training: Model & assumptions"])

with tab_agent:
    st.caption(f"Cortex Agent `{AGENT}` · tools: run_strategy, run_strategy_sweep, explain_strategy, find_squeeze_precedents, "
               "market_outlook, describe_model, market_analyst, policy_search. Every scenario it runs appears on the **Scenarios** tab.")
    _, c2 = st.columns([4, 1])
    if c2.button("New conversation", icon=":material/refresh:", width="stretch"):
        ss.chat = []
    picked = None
    if not ss.chat:
        st.markdown("**Try one:**")
        cols = st.columns(2)
        for i, q in enumerate(SAMPLE_QS):
            if cols[i % 2].button(q, key=f"sq{i}", width="stretch"):
                picked = q
    for idx, msg in enumerate(ss.chat):
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                render_steps(msg.get("steps", []))
            st.markdown(msg["text"])
            if msg["role"] == "assistant":
                render_agent_visuals(msg, idx)
    prompt = st.chat_input("Ask about a market strategy, e.g. 'Can we squeeze Volt-X out of Marine, and where would it go?'")
    question = prompt or picked or ss.pop("pending_q", None)
    if question:
        ss.chat.append(dict(role="user", text=question))
        with st.chat_message("user"):
            st.markdown(question)
        history = [{"role": m["role"], "content": [{"type": "text", "text": m["text"]}]} for m in ss.chat]
        with st.chat_message("assistant"):
            with st.spinner("Agent is building strategies, running Monte Carlo simulations and checking history & policy "
                            "(typically 45–120s)..."):
                try:
                    parsed = parse_agent(ask_agent(history))
                except Exception as e:  # surface agent/permission errors in the UI
                    parsed = dict(steps=[], runs=[], sweeps=[], stress=[], outlooks=[], precedents=[], model=False,
                                  text=f":red[Agent call failed:] `{e}`")
        add_scenarios(parsed["runs"], question)
        for sw in parsed["sweeps"]:
            if sw not in [x["sweep_id"] for x in ss.sweeps]:
                ss.sweeps.insert(0, dict(sweep_id=sw, question=question))
        ss.chat.append(dict(role="assistant", question=question, **parsed))
        st.rerun()

with tab_scen:
    data = load_strategies(tuple(x["sid"] for x in ss.scenarios))
    items = [(x, data[x["sid"]]) for x in ss.scenarios if x["sid"] in data]
    top1, top2 = st.columns([3, 2])
    top1.markdown(f"##### {len(items)} scenario(s) built by the agent this session")
    with top2.popover("Load earlier scenarios", icon=":material/history:", width="stretch"):
        lg = recent_log()
        lg = lg[~lg.SCENARIO_ID.isin([x["sid"] for x in ss.scenarios])]
        opts = {f"{r.RUN_TS:%m-%d %H:%M} · {r.LABEL[:80]} · {r.DELTA_P50_M:+.1f}M": r.SCENARIO_ID for r in lg.itertuples()}
        pick = st.multiselect("Earlier agent scenarios", list(opts), key="log_pick")
        if st.button("Load", disabled=not pick):
            for p in pick:
                add_scenarios([opts[p]], "loaded from scenario log")
            st.rerun()
    if not items:
        st.info("No scenarios yet. Ask the agent a strategy question — every simulation it runs lands here with its "
                "configuration, volume-flow and outcome plots.", icon=":material/smart_toy:")
    else:
        labels = {f"{i + 1}. {r['LABEL'][:90]}": x["sid"] for i, (x, r) in enumerate(items)}
        sel = st.multiselect("Compare up to 3 scenarios", list(labels), default=list(labels)[:min(2, len(labels))],
                             max_selections=3, key="cmp")
        if sel:
            chosen = [data[labels[l]] for l in sel]
            st.altair_chart(cum_margin_chart({l: data[labels[l]]["OUTPUT"]["cum_curve"] for l in sel}), width="stretch")
            cmp = pd.DataFrame([{"scenario": l, "margin vs base P50": money(r["OUTPUT"]["summary"]["margin_delta"][1]),
                                 "P10 → P90": f"{money(r['OUTPUT']['summary']['margin_delta'][0])} → "
                                              f"{money(r['OUTPUT']['summary']['margin_delta'][2])}",
                                 "P(loss)": f"{r['OUTPUT']['summary']['prob_loss']:.0%}",
                                 "payback (P50)": r["OUTPUT"]["summary"]["payback_month_p50"],
                                 "P(competitor exits)": f"{r['OUTPUT']['summary']['prob_target_exit']:.0%}",
                                 "config": r["OUTPUT"].get("config_text", "")} for l, r in zip(sel, chosen)])
            st.dataframe(cmp, hide_index=True, width="stretch")
            fcols = st.columns(len(chosen))
            for col, l, r in zip(fcols, sel, chosen):
                fc = flow_chart(r["OUTPUT"]["flows"], l[:60])
                if fc is not None:
                    col.altair_chart(fc, width="stretch")
            if st.button("Ask the agent to compare these", icon=":material/smart_toy:"):
                ids = "; ".join(f"scenario {r['SCENARIO_ID']} ({r['LABEL']})" for r in chosen)
                ss.pending_q = (f"Compare these strategies: {ids}. Explain what drives the difference, where competitor capacity "
                                "flows in each, check the policy guardrails, and recommend one as a proposal.")
                st.toast("Question sent — open the **Ask the agent** tab.", icon=":material/smart_toy:")
                st.rerun()
        st.divider()
        for i, (x, r) in enumerate(items):
            sm = r["OUTPUT"]["summary"]
            with st.expander(f"{i + 1}. {r['LABEL']} · {money(sm['margin_delta'][1])} P50 · P(loss) {sm['prob_loss']:.0%}",
                             expanded=(i == 0), icon=":material/stacked_line_chart:"):
                render_card(r, f"s{i}")
    for j, sw in enumerate(ss.sweeps):
        df = load_sweep(sw["sweep_id"])
        if df.empty:
            continue
        with st.expander(f"Sweep · {MKT_NAMES.get(df.TARGET_MARKET.iloc[0])} vs {COMP_NAMES.get(df.TARGET_COMPETITOR.iloc[0])} · "
                         f"{len(df)} strategies", icon=":material/grid_on:", expanded=(j == 0 and not items)):
            st.caption(f"Built by the agent for: “{sw['question']}”")
            ch, best = sweep_heatmap(df)
            st.altair_chart(ch, width="stretch")
            if best is not None:
                st.success(f"Best within the 25% loss-risk policy: **{best.cut} for {best.CUT_MONTHS} months** → "
                           f"{best.p50:+.1f}M P50 (P10 {best.MARGIN_DELTA_P10 / 1e6:+.1f}M, P90 {best.MARGIN_DELTA_P90 / 1e6:+.1f}M), "
                           f"P(loss) {best.PROB_LOSS:.0%}, P(competitor exits) {best.PROB_TARGET_EXIT:.0%}.", icon=":material/flag:")
            else:
                st.warning("No tested strategy meets the 25% loss-risk policy.")

with tab_model:
    card = model_card()
    st.markdown("##### Models behind the simulation (fitted once from market history, sampled with their uncertainty)")
    md = pd.DataFrame(card["models"])
    st.dataframe(md, hide_index=True, width="stretch")
    st.caption(f"{int(md['count'].sum())} fitted models · fitted {card['model_fit_ts'][:16]} · not retrained per simulation; every "
               "scenario stores a snapshot of the parameters it used.")
    m1, m2 = st.columns(2)
    with m1:
        st.markdown("**Markets**")
        mk = pd.DataFrame(card["markets"]).T
        mk.index = mk.index.map(MKT_NAMES)
        st.dataframe(mk, width="stretch")
    with m2:
        st.markdown("**Competitors**")
        cp = pd.DataFrame({j: {"price follow": v["price_follow"], "pivot hazard (a, b)": str(v["hazard"]),
                               "b ± SE": v["hazard_slope_se"], "fraction moved": v["pivot_fraction"],
                               "destination preference θ": v["destination_theta"], "past pivots": v["n_pivots"],
                               "return rate / mo": round(v["return_rate"], 4),
                               "past destinations": str(v["destination_counts"])} for j, v in card["competitors"].items()}).T
        cp.index = cp.index.map(COMP_NAMES)
        st.dataframe(cp, width="stretch")
    st.markdown("**Pivot probability per month vs margin squeeze** (fitted logistic hazard; band = ±1 SE on the slope)")
    sq = np.linspace(0, 0.2, 81)
    rows = []
    for j, v in card["competitors"].items():
        a, b, se = v["hazard"][0], v["hazard"][1], v["hazard_slope_se"]
        for q in sq:
            f = lambda bb: 1 / (1 + np.exp(-(a + bb * q)))
            rows.append(dict(competitor=COMP_NAMES[j], squeeze=q, p=f(b), lo=f(max(b - se, 0)), hi=f(b + se)))
    hz = pd.DataFrame(rows)
    color = alt.Color("competitor:N", scale=alt.Scale(domain=[COMP_NAMES[c] for c in ["VOLTX", "APEX", "REST"]],
                                                      range=[COMP_COLORS[c] for c in ["VOLTX", "APEX", "REST"]]),
                      legend=alt.Legend(orient="bottom", title=None))
    base = alt.Chart(hz).encode(x=alt.X("squeeze:Q", title="Margin squeeze (points below their normal margin)",
                                        axis=alt.Axis(format="%")), color=color)
    st.altair_chart((base.mark_area(opacity=0.12).encode(y=alt.Y("lo:Q", title="P(start pivoting this month)",
                                                                 axis=alt.Axis(format="%")), y2="hi:Q")
                     + base.mark_line(strokeWidth=2.5).encode(y="p:Q")).properties(height=260), width="stretch")
    r1, r2 = st.columns(2)
    with r1:
        st.markdown("**Randomized in every Monte Carlo path (9 sources)**")
        for k, v in card["uncertainty_sources"].items():
            st.markdown(f"- `{k}` — {v}")
    with r2:
        st.markdown("**Held fixed**")
        for a in card["fixed_assumptions"]:
            st.markdown(f"- {a}")
        st.markdown("**Adaptation rules the agent can choose**")
        for k, v in card["adapt_rules"].items():
            st.markdown(f"- `{k}` — {v}")
    with st.expander("History check — our past price-cut episodes and what competitors actually did", icon=":material/history_edu:"):
        st.dataframe(squeeze_episodes(), hide_index=True, width="stretch")
