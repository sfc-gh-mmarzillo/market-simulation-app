-- Phase 2: Cortex Agent. LLM frames + explains; tools compute every number.
CREATE OR REPLACE AGENT WHATIF_DEMO.SIM.WHATIF_AGENT
  COMMENT = 'Voltara what-if pricing agent: Scenario Builder + Explainer over the Monte Carlo engine'
  PROFILE = '{"display_name": "Voltara What-If Agent", "color": "blue"}'
  FROM SPECIFICATION
  $$
  models:
    orchestration: auto

  orchestration:
    budget:
      seconds: 180
      tokens: 32000

  instructions:
    orchestration: |
      You are the pricing what-if analyst for Voltara Energy, a battery manufacturer.
      GOLDEN RULE: never compute, estimate, add, subtract or extrapolate numbers yourself. Every figure in your answer
      must be copied from a tool result. If a number you need was not returned by a tool, run another tool call.
      Routing:
      - Hypothetical / forward-looking questions ("what if we raise...", "what happens if competitor cuts...", "lead goes up...")
        -> run_scenario. Translate the question into inputs: customer name, product line (ALL, FLOODED, EFB, AGM, HEAVY_DUTY),
        our price change in whole percent, competitor move in whole percent (omit it to let the model sample competitor
        reactions; set it only when the user states a competitor move), and lead shock in whole percent.
        "Short on lead" or "lead is tight" with no number means lead_shock_pct = 20.
        When comparing options (e.g. 4% vs 6% vs 8%), call run_scenario once per option.
      - "What is the best / optimal / highest safe price" questions -> run_sweep over a sensible range (e.g. -2 to 12 in
        steps of 2) and report best_price_within_risk_limit, the trade-off, and the scenario_id of the recommended point.
      - "How is the model configured / what does it assume / how confident is it" -> describe_model, and explain the
        fitted elasticity with its standard error, what is sampled vs fixed, and the backtest error on past price moves.
      - If the user asks to remove a source of uncertainty (e.g. "ignore competitors", "hold lead flat"), pass
        disable_uncertainty to run_scenario (comma list of elasticity, competitor, churn, lead, demand). For a different
        horizon pass horizon_months.
      - "Why" / root-cause / "what drives the downside" questions -> explain_scenario with the scenario_id from the earlier
        run_scenario result.
      - Questions about past scenarios already run, or historical sales, prices, volumes, lead prices -> scenario_analyst.
      - Precedent / evidence: for ANY proposed price move, ALSO call find_similar_cases with the same customer, product line,
        price change and lead shock, and compare the simulation with what actually happened in the closest past cases
        (realized volume change, competitor response mix, model backtest error). If the precedents disagree with the
        simulation, say so plainly. "Have we done this before" / "similar examples" questions -> find_similar_cases.
      - Contract terms, notice periods, surcharge clauses, approval thresholds, competitive playbook -> policy_search.
        For any recommended price move, ALWAYS check policy_search for that customer's contract and the pricing governance
        policy, and state the approval level and notice period.
    response: |
      Be concise and executive-friendly. Structure: 1) headline answer with the P50 margin change vs baseline and the
      P10-P90 range, 2) key risks (probabilities as returned), 3) top drivers, 3b) historical precedents (n cases, realized
      median volume change, competitor response mix, model backtest bias), 4) contract/policy constraints with document
      title, 5) recommendation framed as a proposal requiring human approval. Always include the scenario_id(s).
      Use a small markdown table when comparing scenarios. Never present a number not returned by a tool.
      Always state the model_configuration line returned by the tool (runs, horizon, what uncertainty is on).
    sample_questions:
      - question: "What if we raise AutoZone prices 8% and let the model decide how competitors react?"
      - question: "What if Walmart pushes back on a 6% increase while we're short on lead?"
      - question: "Compare 4%, 6% and 8% increases for O'Reilly AGM batteries."
      - question: "What's the highest price increase for Walmart that keeps the risk of losing margin under 25%?"
      - question: "How is the model configured for AutoZone, and how confident is it?"
      - question: "Why is the downside so bad on that scenario?"
      - question: "Have we raised prices like this at Advance Auto before, and what happened?"
      - question: "What approvals and notice do we need for a 7% increase at Costco?"

  tools:
    - tool_spec:
        type: generic
        name: run_scenario
        description: "Runs a 10,000-path Monte Carlo simulation (12-month horizon) of a price change for one customer. Returns P10/P50/P90 gross margin, revenue, volume, margin change vs no-change baseline, risk probabilities, driver attribution and a scenario_id. Percent inputs are whole numbers: 8 means +8%, -5 means -5%."
        input_schema:
          type: object
          properties:
            customer_name:
              type: string
              description: "Customer account, e.g. AutoZone, Walmart, O'Reilly, Advance Auto, NAPA, Costco, Interstate Dealer Network, OEM Assembly"
            product_line:
              type: string
              description: "ALL (default), FLOODED, EFB, AGM, or HEAVY_DUTY"
            price_change_pct:
              type: number
              description: "Our price change in whole percent, e.g. 8 for +8%"
            competitor_move_pct:
              type: number
              description: "Competitor price move in whole percent, e.g. -5. Omit to sample competitor response from the historical response model."
            lead_shock_pct:
              type: number
              description: "LME lead price shock in whole percent, e.g. 20. Use 0 if not mentioned."
            scenario_name:
              type: string
              description: "Short label for the scenario log"
            horizon_months:
              type: integer
              description: "Simulation horizon in months (default 12; 6-24 typical)"
            disable_uncertainty:
              type: string
              description: "Optional comma list of uncertainty sources to switch off: elasticity, competitor, churn, lead, demand"
          required: [customer_name, product_line, price_change_pct, lead_shock_pct]
    - tool_spec:
        type: generic
        name: explain_scenario
        description: "Root-cause analysis of a scenario already run: driver variance attribution plus engine-computed sensitivities (competitor holds / matches / undercuts 5%, lead +20%). Takes about 20 seconds."
        input_schema:
          type: object
          properties:
            scenario_id:
              type: string
              description: "scenario_id returned by run_scenario"
          required: [scenario_id]
    - tool_spec:
        type: generic
        name: find_similar_cases
        description: "Finds the closest real historical price moves (same customer first, then same channel; matched on size of move, product line, lead price level and season) and returns what actually happened over the next 12 weeks: realized volume and revenue change, competitor response mix and timing, and how accurate the model was on those cases. Percent inputs are whole numbers."
        input_schema:
          type: object
          properties:
            customer_name:
              type: string
              description: "Customer account name"
            product_line:
              type: string
              description: "ALL, FLOODED, EFB, AGM, or HEAVY_DUTY"
            price_change_pct:
              type: number
              description: "Proposed price change in whole percent, e.g. 8"
            lead_shock_pct:
              type: number
              description: "LME lead shock in whole percent, 0 if none"
          required: [customer_name, product_line, price_change_pct]
    - tool_spec:
        type: generic
        name: run_sweep
        description: "Runs the Monte Carlo engine across a range of price changes in one call (5,000 runs per point, same random draws) and returns every point plus the best price whose probability of margin below baseline stays within the risk limit (default 25%, the pricing-policy threshold). Whole-number percents."
        input_schema:
          type: object
          properties:
            customer_name:
              type: string
              description: "Customer account name"
            product_line:
              type: string
              description: "ALL, FLOODED, EFB, AGM, or HEAVY_DUTY"
            from_pct:
              type: number
              description: "Lowest price change to test, whole percent, e.g. -2"
            to_pct:
              type: number
              description: "Highest price change to test, whole percent, e.g. 12"
            step_pct:
              type: number
              description: "Step in whole percent, e.g. 2"
            lead_shock_pct:
              type: number
              description: "LME lead shock in whole percent, 0 if none"
            competitor_move_pct:
              type: number
              description: "Fixed competitor move in whole percent; omit to sample competitor response"
            max_risk_pct:
              type: number
              description: "Max acceptable probability (whole %) that margin ends below baseline; default 25"
          required: [customer_name, product_line, from_pct, to_pct]
    - tool_spec:
        type: generic
        name: describe_model
        description: "Returns the model card for a customer / product line: fitted own and cross price elasticities with standard errors and R², churn model, competitor-response probabilities, LME lead drift and volatility, demand noise, data window, which uncertainties the simulation samples vs what it holds fixed, default settings, and the backtest error on past price moves."
        input_schema:
          type: object
          properties:
            customer_name:
              type: string
              description: "Customer account name"
            product_line:
              type: string
              description: "ALL, FLOODED, EFB, AGM, or HEAVY_DUTY"
          required: [customer_name]
    - tool_spec:
        type: cortex_analyst_text_to_sql
        name: scenario_analyst
        description: "Queries past scenario results (margins, probabilities, inputs) and historical weekly sales (units, revenue, our price, competitor price, LME lead price) by customer, channel and product line."
    - tool_spec:
        type: cortex_search
        name: policy_search
        description: "Searches customer contract terms (notice periods, increase caps, surcharge clauses, most-favored-customer) and internal pricing governance, surcharge, competitive-response and approval policies."

  tool_resources:
    run_scenario:
      type: procedure
      identifier: WHATIF_DEMO.SIM.SIM_TOOL
      execution_environment:
        type: warehouse
        warehouse: AICOLLEGE
        query_timeout: 120
    explain_scenario:
      type: procedure
      identifier: WHATIF_DEMO.SIM.EXPLAIN_SCENARIO
      execution_environment:
        type: warehouse
        warehouse: AICOLLEGE
        query_timeout: 180
    run_sweep:
      type: procedure
      identifier: WHATIF_DEMO.SIM.SWEEP_TOOL
      execution_environment:
        type: warehouse
        warehouse: AICOLLEGE
        query_timeout: 180
    describe_model:
      type: procedure
      identifier: WHATIF_DEMO.SIM.DESCRIBE_MODEL
      execution_environment:
        type: warehouse
        warehouse: AICOLLEGE
        query_timeout: 60
    find_similar_cases:
      type: procedure
      identifier: WHATIF_DEMO.SIM.FIND_ANALOGS
      execution_environment:
        type: warehouse
        warehouse: AICOLLEGE
        query_timeout: 60
    scenario_analyst:
      semantic_view: WHATIF_DEMO.SIM.SCENARIO_SV
      execution_environment:
        type: warehouse
        warehouse: AICOLLEGE
    policy_search:
      search_service: WHATIF_DEMO.SIM.POLICY_SEARCH
      max_results: 5
      title_column: TITLE
      id_column: DOC_ID
  $$;
