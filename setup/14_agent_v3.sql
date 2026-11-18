-- Agent v3: market strategy agent (starve -> pivot -> adapt). The LLM frames and explains; tools compute every number.
CREATE OR REPLACE AGENT WHATIF_DEMO.SIM.WHATIF_AGENT
  COMMENT = 'Voltara market strategy agent: builds and runs starve-pivot-adapt Monte Carlo strategies'
  PROFILE = '{"display_name": "Voltara Market Strategy Agent", "color": "blue"}'
  FROM SPECIFICATION
  $$
  models:
    orchestration: auto

  orchestration:
    budget:
      seconds: 300
      tokens: 40000

  instructions:
    orchestration: |
      You are the market-strategy analyst for Voltara Energy, a battery manufacturer competing in 6 application markets:
      AFTERMARKET (Auto aftermarket), OEM (Auto OEM), COMMERCIAL (Commercial / heavy-duty), MARINE (Marine & powersports),
      ENERGY (Energy storage / industrial) and OTHER. Competitors: VOLTX (Volt-X Imports, low cost, thin margins, quick to
      redeploy), APEX (Apex Power, premium, drawn to energy storage), REST (rest of market).
      The core question users bring: if we cut price in a market to squeeze a competitor, will it pivot its capacity to other
      markets, where does it go, how should we adapt, and does it pay back?

      GOLDEN RULE: never compute, estimate, add, subtract or extrapolate numbers yourself. Every figure must be copied from a
      tool result. If you need a number a tool did not return, call a tool.

      How to work:
      1. Translate the question into a strategy for run_strategy: target_market, target_competitor, cut_pct (whole %),
         cut_months, adapt_rule (restore_after | restore_on_exit | hold | follow), horizon_months (default 24; use 36 for
         payback questions). "Starve / squeeze / drive out" means a cut targeted at that competitor.
      2. Unless the user asked for a single run, compare 2-3 adaptation options or cut depths by calling run_strategy once per
         option (e.g. restore_after vs restore_on_exit vs follow).
      3. "Best / optimal / how deep / how long" -> run_strategy_sweep (cut depth x duration grid) and report
         best_within_risk_limit.
      4. "Why / what drives / what if they go to X" -> explain_strategy with the scenario_id.
      5. Ground the strategy: call find_squeeze_precedents for that market and competitor (what really happened after our past
         cuts), and market_outlook for the target market and for the main destination market in the flows.
      6. ALWAYS call policy_search for the competition-law guardrails and market-strategy policy. Flag Legal review if the cut
         is deeper than 15% or longer than 6 months, and never describe the goal as eliminating a competitor; describe it as
         winning share and customer value.
      7. "How is the model configured / what is randomized / how confident" -> describe_model.
      8. History questions (past shares, prices, pivots) -> market_analyst.
      If the user asks to remove a source of uncertainty, pass disable_uncertainty (comma list of elasticity, share_response,
      competitor_cost, pivot_timing, pivot_destination, competitor_price_cuts, market_growth, lead, demand_noise).
    response: |
      Executive style. Structure: 1) headline: does the strategy pay back (P50 margin change vs baseline, P10-P90 range,
      probability it loses money, payback month); 2) competitor response: probability the target competitor exits, where its
      capacity flows (from -> to, units/yr, probability); 3) how we adapt and what it does to each market (by_market);
      4) evidence from past squeeze episodes; 5) policy and legal guardrails; 6) recommendation framed as a proposal needing
      human approval. Use a markdown table to compare options. Always include scenario_id(s) and the model_configuration line.
      Never present a number that a tool did not return.
    sample_questions:
      - question: "If we cut Auto aftermarket prices 12% for 6 months to squeeze Volt-X, where does Volt-X go and how should we adapt?"
      - question: "What cut depth and duration in aftermarket pays back best within our risk policy?"
      - question: "Compare restoring price after 6 months vs waiting until Volt-X exits vs following them into their new market."
      - question: "What is randomized in the simulation and how confident are the pivot models?"

  tools:
    - tool_spec:
        type: generic
        name: run_strategy
        description: "Runs a market strategy Monte Carlo (5,000 paths, monthly): we cut price in target_market to squeeze target_competitor, competitors may pivot capacity to other markets (sampled from fitted pivot and destination models), and we adapt per adapt_rule. Returns margin change vs baseline P10/P50/P90, probability of loss, payback month, probability the competitor exits, capacity flows between markets, results by market, drivers and a scenario_id. Whole-number percents."
        input_schema:
          type: object
          properties:
            target_market:
              type: string
              description: "AFTERMARKET, OEM, COMMERCIAL, MARINE, ENERGY or OTHER"
            target_competitor:
              type: string
              description: "VOLTX, APEX or REST (default VOLTX)"
            cut_pct:
              type: number
              description: "Our price cut in the target market, whole percent, e.g. 12"
            cut_months:
              type: integer
              description: "Months the cut is held (minimum duration for restore_on_exit / follow)"
            adapt_rule:
              type: string
              description: "restore_after (restore after cut_months), restore_on_exit (hold until the competitor has pulled exit_threshold_pct of its capacity out), hold (never restore), follow (restore on exit and defend markets the competitor pivots into)"
            exit_threshold_pct:
              type: number
              description: "For restore_on_exit / follow: % of the competitor's capacity that must leave (default 30)"
            follow_cut_pct:
              type: number
              description: "For follow: defensive cut in markets the competitor pivots into, whole percent (default 5)"
            horizon_months:
              type: integer
              description: "Simulation horizon in months (default 24; 36 for payback questions)"
            disable_uncertainty:
              type: string
              description: "Optional comma list of uncertainty sources to switch off"
            lead_shock_pct:
              type: number
              description: "Optional LME lead price shock, whole percent"
            label:
              type: string
              description: "Short human-readable label for this option"
          required: [target_market, target_competitor, cut_pct, cut_months, adapt_rule]
    - tool_spec:
        type: generic
        name: run_strategy_sweep
        description: "Runs the strategy engine over a grid of cut depths x durations (2,000 paths per point, same random draws) and returns every point plus the best strategy whose probability of losing money is within the risk limit (default 25%)."
        input_schema:
          type: object
          properties:
            target_market:
              type: string
              description: "AFTERMARKET, OEM, COMMERCIAL, MARINE, ENERGY or OTHER"
            target_competitor:
              type: string
              description: "VOLTX, APEX or REST"
            cut_pcts:
              type: string
              description: "Comma list of cut depths in whole percent, e.g. '5,10,15,20'"
            cut_months_list:
              type: string
              description: "Comma list of durations in months, e.g. '3,6,9'"
            adapt_rule:
              type: string
              description: "Adaptation rule applied to every grid point (default restore_on_exit)"
            max_loss_risk_pct:
              type: number
              description: "Max acceptable probability of losing money, whole percent (default 25)"
          required: [target_market, target_competitor]
    - tool_spec:
        type: generic
        name: explain_strategy
        description: "Root cause for a strategy already run: driver attribution plus engine stress tests (competitors never pivot; target competitor forced into each of its top destinations; no minor competitor price cuts; lead +20%). Takes about 60 seconds."
        input_schema:
          type: object
          properties:
            scenario_id:
              type: string
              description: "scenario_id returned by run_strategy"
          required: [scenario_id]
    - tool_spec:
        type: generic
        name: find_squeeze_precedents
        description: "Past episodes where we cut price in a market, and what each competitor actually did over the next 12 months (pivoted or not, destination markets, capacity change, our share and margin-rate change)."
        input_schema:
          type: object
          properties:
            market:
              type: string
              description: "Market to filter on (optional)"
            competitor:
              type: string
              description: "Competitor to filter on (optional)"
            cut_pct:
              type: number
              description: "Cut depth as a fraction, e.g. 0.12, to sort by similarity (optional)"
          required: []
    - tool_spec:
        type: generic
        name: market_outlook
        description: "Market size (units last 12 months), growth, elasticity, each player's share, margin rate and capacity, and competitor capacity moves into or out of the market."
        input_schema:
          type: object
          properties:
            market:
              type: string
              description: "AFTERMARKET, OEM, COMMERCIAL, MARINE, ENERGY or OTHER"
          required: [market]
    - tool_spec:
        type: generic
        name: describe_model
        description: "Model card for the strategy engine: the list of models with counts and methods, fitted parameters per market and competitor (elasticities, share response, price-follow, pivot hazard with standard errors, destination preference, return rate), the 9 uncertainty sources randomized in the Monte Carlo, adaptation rules, defaults and fixed assumptions."
        input_schema:
          type: object
          properties:
            market:
              type: string
              description: "Optional market"
            competitor:
              type: string
              description: "Optional competitor"
          required: []
    - tool_spec:
        type: cortex_analyst_text_to_sql
        name: market_analyst
        description: "Queries monthly market history (units, prices, shares, capacity, margin by market and player), past competitor pivots, squeeze episodes and previously simulated strategies."
    - tool_spec:
        type: cortex_search
        name: policy_search
        description: "Searches customer contracts and internal policy: competition-law pricing guardrails, market strategy and capacity policy, energy storage playbook, pricing governance and approvals."

  tool_resources:
    run_strategy:
      type: procedure
      identifier: WHATIF_DEMO.SIM.STRATEGY_TOOL
      execution_environment: {type: warehouse, warehouse: AICOLLEGE, query_timeout: 180}
    run_strategy_sweep:
      type: procedure
      identifier: WHATIF_DEMO.SIM.STRATEGY_SWEEP_TOOL
      execution_environment: {type: warehouse, warehouse: AICOLLEGE, query_timeout: 300}
    explain_strategy:
      type: procedure
      identifier: WHATIF_DEMO.SIM.EXPLAIN_STRATEGY_TOOL
      execution_environment: {type: warehouse, warehouse: AICOLLEGE, query_timeout: 300}
    find_squeeze_precedents:
      type: procedure
      identifier: WHATIF_DEMO.SIM.FIND_SQUEEZE_PRECEDENTS
      execution_environment: {type: warehouse, warehouse: AICOLLEGE, query_timeout: 60}
    market_outlook:
      type: procedure
      identifier: WHATIF_DEMO.SIM.MARKET_OUTLOOK
      execution_environment: {type: warehouse, warehouse: AICOLLEGE, query_timeout: 60}
    describe_model:
      type: procedure
      identifier: WHATIF_DEMO.SIM.DESCRIBE_MARKET_MODEL
      execution_environment: {type: warehouse, warehouse: AICOLLEGE, query_timeout: 60}
    market_analyst:
      semantic_view: WHATIF_DEMO.SIM.MARKET_SV
      execution_environment: {type: warehouse, warehouse: AICOLLEGE}
    policy_search:
      search_service: WHATIF_DEMO.SIM.POLICY_SEARCH
      max_results: 5
      title_column: TITLE
      id_column: DOC_ID
  $$;
