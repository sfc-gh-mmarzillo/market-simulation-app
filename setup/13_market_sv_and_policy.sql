-- Market strategy semantic view (Cortex Analyst) + market-strategy policy docs (Cortex Search)
CREATE OR REPLACE SEMANTIC VIEW WHATIF_DEMO.SIM.MARKET_SV
  TABLES (
    markets AS WHATIF_DEMO.SIM.MARKETS PRIMARY KEY (MARKET)
      COMMENT = 'The 6 application markets (Auto aftermarket, Auto OEM, Commercial/heavy-duty, Marine & powersports, Energy storage/industrial, Other)',
    history AS WHATIF_DEMO.SIM.MARKET_HISTORY
      COMMENT = 'Monthly history per market and player (US = Voltara, VOLTX = Volt-X Imports, APEX = Apex Power, REST = rest of market)',
    moves AS WHATIF_DEMO.SIM.COMPETITOR_MOVES PRIMARY KEY (MOVE_ID)
      COMMENT = 'Every past competitor capacity pivot from one market to another',
    episodes AS WHATIF_DEMO.SIM.SQUEEZE_EPISODES
      COMMENT = 'Our past price-cut episodes and what each competitor did in the following 12 months',
    strategies AS WHATIF_DEMO.SIM.STRATEGY_RESULTS PRIMARY KEY (SCENARIO_ID)
      COMMENT = 'Every simulated starve-pivot-adapt strategy (Monte Carlo results)'
  )
  RELATIONSHIPS (
    history_to_markets AS history (MARKET) REFERENCES markets,
    moves_to_markets AS moves (FROM_MARKET) REFERENCES markets,
    episodes_to_markets AS episodes (MARKET) REFERENCES markets,
    strategies_to_markets AS strategies (TARGET_MARKET) REFERENCES markets
  )
  FACTS (
    history.units_f AS UNITS, history.price_f AS PRICE, history.revenue_f AS UNITS * PRICE,
    history.margin_f AS UNITS * (PRICE - UNIT_COST), history.share_f AS SHARE, history.capacity_f AS CAPACITY,
    moves.units_moved_f AS UNITS_PER_YEAR, moves.fraction_f AS FRACTION_OF_CAPACITY, moves.squeeze_f AS SQUEEZE_AT_START,
    episodes.cap_change_f AS CAPACITY_CHANGE_12M, episodes.cut_f AS CUT_PCT,
    strategies.delta_p50_f AS MARGIN_DELTA_P50, strategies.delta_p10_f AS MARGIN_DELTA_P10, strategies.delta_p90_f AS MARGIN_DELTA_P90,
    strategies.prob_loss_f AS PROB_LOSS, strategies.prob_exit_f AS PROB_TARGET_EXIT
  )
  DIMENSIONS (
    markets.market AS MARKET, markets.market_name AS MARKET_NAME WITH SYNONYMS = ('segment', 'end market', 'application'),
    history.month AS MONTH, history.player AS PLAYER WITH SYNONYMS = ('competitor', 'company')
      COMMENT = 'US = Voltara (us); VOLTX = Volt-X Imports; APEX = Apex Power; REST = rest of market',
    moves.competitor AS COMPETITOR, moves.to_market AS TO_MARKET COMMENT = 'Market the capacity moved into',
    moves.start_month AS START_MONTH,
    episodes.episode_id AS EPISODE_ID, episodes.episode_competitor AS COMPETITOR, episodes.pivoted AS PIVOTED,
    episodes.destinations AS DESTINATIONS,
    strategies.scenario_id AS SCENARIO_ID, strategies.label AS LABEL, strategies.run_ts AS RUN_TS,
    strategies.target_competitor AS TARGET_COMPETITOR, strategies.adapt_rule AS ADAPT_RULE,
    strategies.cut_months AS CUT_MONTHS, strategies.cut_pct_dim AS CUT_PCT
  )
  METRICS (
    history.total_units AS SUM(history.units_f), history.total_revenue AS SUM(history.revenue_f),
    history.total_margin AS SUM(history.margin_f), history.avg_share AS AVG(history.share_f),
    history.avg_capacity AS AVG(history.capacity_f), history.avg_price AS AVG(history.price_f),
    moves.pivot_count AS COUNT(moves.units_moved_f), moves.units_moved AS SUM(moves.units_moved_f),
    moves.avg_fraction_moved AS AVG(moves.fraction_f),
    episodes.avg_capacity_change AS AVG(episodes.cap_change_f),
    strategies.strategy_count AS COUNT(strategies.delta_p50_f), strategies.avg_margin_delta_p50 AS AVG(strategies.delta_p50_f),
    strategies.avg_prob_loss AS AVG(strategies.prob_loss_f), strategies.avg_prob_exit AS AVG(strategies.prob_exit_f)
  )
  COMMENT = 'Voltara market history, competitor pivots, squeeze episodes and simulated strategies';

DELETE FROM WHATIF_DEMO.SIM.POLICY_DOCS WHERE DOC_ID IN ('POL-PREDATORY', 'POL-MARKET', 'POL-ENERGY');
INSERT INTO WHATIF_DEMO.SIM.POLICY_DOCS VALUES
('POL-PREDATORY', 'Competition Law Guardrails for Pricing Strategy', 'policy', NULL,
 'Pricing below average variable cost, or a sustained deep cut whose stated purpose is to eliminate a competitor, can create predatory-pricing risk under competition law. Any market-wide price reduction deeper than 15% or longer than 6 months, or any plan documented as intended to drive a named competitor out of a market, requires prior Legal / Competition Counsel review. Internal documents must describe objectives as winning share and customer value, not harming competitors. Prices must stay above our average variable cost at all times.'),
('POL-MARKET', 'Market Strategy & Capacity Allocation Policy', 'policy', NULL,
 'Market-level price strategies are approved by the Commercial Strategy Council. Proposals must show the Monte Carlo cumulative margin path, payback month, probability of loss over the horizon (must be 25% or lower without a mitigation plan), and expected competitor capacity flows into our other markets. When a competitor is expected to redeploy capacity into Energy storage or Marine, the proposal must include a defence plan for those markets.'),
('POL-ENERGY', 'Energy Storage Growth Playbook', 'policy', NULL,
 'Energy storage / industrial is our fastest-growing market (about 18% per year) where our share is low. Competitor capacity arriving in this market is the main risk to the growth plan. Defensive pricing in Energy storage is limited to 5% for up to 6 months; prefer service, warranty and integration offers over list-price cuts.');
ALTER CORTEX SEARCH SERVICE WHATIF_DEMO.SIM.POLICY_SEARCH REFRESH;
