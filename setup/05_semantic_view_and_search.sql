-- Phase 2: semantic view (Cortex Analyst) + policy docs (Cortex Search)
CREATE OR REPLACE SEMANTIC VIEW WHATIF_DEMO.SIM.SCENARIO_SV
  TABLES (
    customers AS WHATIF_DEMO.SIM.CUSTOMERS PRIMARY KEY (CUSTOMER_ID)
      COMMENT = 'Voltara Energy customer accounts (retailers, distributors, OEM)',
    scenarios AS WHATIF_DEMO.SIM.SCENARIO_RESULTS PRIMARY KEY (SCENARIO_ID)
      COMMENT = 'One row per Monte Carlo what-if scenario run (12-month horizon)',
    sales AS WHATIF_DEMO.SIM.SALES_HISTORY
      COMMENT = 'Weekly historical sales by customer and product line',
    price_events AS WHATIF_DEMO.SIM.PRICE_EVENTS PRIMARY KEY (EVENT_ID)
      COMMENT = 'Every past price move of 2%+ with realized 12-week outcomes, competitor response, and model backtest'
  )
  RELATIONSHIPS (
    scenarios_to_customers AS scenarios (CUSTOMER_ID) REFERENCES customers,
    sales_to_customers AS sales (CUSTOMER_ID) REFERENCES customers,
    events_to_customers AS price_events (CUSTOMER_ID) REFERENCES customers
  )
  FACTS (
    scenarios.margin_p10_f AS MARGIN_P10,
    scenarios.margin_p50_f AS MARGIN_P50,
    scenarios.margin_p90_f AS MARGIN_P90,
    scenarios.margin_delta_p50_f AS MARGIN_DELTA_P50,
    scenarios.revenue_p50_f AS REVENUE_P50,
    scenarios.prob_below_base_f AS PROB_MARGIN_BELOW_BASELINE,
    scenarios.prob_share_shift_f AS PROB_SHARE_SHIFT,
    scenarios.runtime_f AS RUNTIME_SEC,
    sales.units_f AS UNITS,
    sales.revenue_f AS UNITS * OUR_PRICE,
    sales.our_price_f AS OUR_PRICE,
    sales.comp_price_f AS COMP_PRICE,
    sales.lead_f AS LEAD_USD_PER_T,
    price_events.ev_move_f AS OUR_PRICE_MOVE,
    price_events.ev_actual_f AS ACTUAL_VOLUME_CHANGE,
    price_events.ev_pred_f AS PREDICTED_VOLUME_CHANGE,
    price_events.ev_err_f AS PREDICTION_ERROR,
    price_events.ev_comp_f AS COMP_PRICE_CHANGE
  )
  DIMENSIONS (
    customers.customer_name AS CUSTOMER_NAME WITH SYNONYMS = ('account', 'retailer') COMMENT = 'Customer account name',
    customers.channel AS CHANNEL COMMENT = 'Aftermarket Retail, Mass Merchant, Club, Distributor, OEM',
    scenarios.scenario_id AS SCENARIO_ID,
    scenarios.scenario_name AS SCENARIO_NAME,
    scenarios.run_ts AS RUN_TS COMMENT = 'When the scenario was run',
    scenarios.scenario_product_line AS SKU_GROUP COMMENT = 'ALL, FLOODED_STD, EFB, AGM, HEAVY_DUTY',
    scenarios.price_change_pct AS PRICE_CHANGE_PCT COMMENT = 'Our price change as a fraction (0.08 = +8%)',
    scenarios.competitor_mode AS COMP_MODE COMMENT = 'MODEL = competitor response sampled, FIXED = assumed move',
    scenarios.competitor_move_pct AS COMP_MOVE_PCT COMMENT = 'Assumed competitor move as fraction when FIXED',
    scenarios.lead_shock_pct AS LEAD_SHOCK_PCT COMMENT = 'LME lead price shock as fraction',
    scenarios.is_sensitivity AS STARTSWITH(SCENARIO_NAME, '[sensitivity]') COMMENT = 'True for automatic sensitivity reruns',
    sales.week_start AS WEEK_START,
    sales.product_line AS SKU_GROUP WITH SYNONYMS = ('sku group', 'battery type'),
    price_events.event_week AS EVENT_WEEK COMMENT = 'Week the historical price move took effect',
    price_events.event_product_line AS SKU_GROUP,
    price_events.competitor_response AS COMP_RESPONSE COMMENT = 'MATCH, UNDERCUT, HOLD (or RAISE after our cuts) within 12 weeks',
    price_events.is_increase AS OUR_PRICE_MOVE > 0 COMMENT = 'True for historical price increases'
  )
  METRICS (
    scenarios.scenario_count AS COUNT(scenarios.SCENARIO_ID),
    scenarios.median_margin AS AVG(scenarios.margin_p50_f) COMMENT = 'P50 12-month gross margin ($), averaged across selected scenarios',
    scenarios.downside_margin AS AVG(scenarios.margin_p10_f) COMMENT = 'P10 (downside) 12-month gross margin ($)',
    scenarios.upside_margin AS AVG(scenarios.margin_p90_f) COMMENT = 'P90 (upside) 12-month gross margin ($)',
    scenarios.margin_change_vs_baseline AS AVG(scenarios.margin_delta_p50_f) COMMENT = 'Median margin change vs no-change baseline ($)',
    scenarios.avg_prob_margin_below_baseline AS AVG(scenarios.prob_below_base_f),
    scenarios.avg_prob_share_shift AS AVG(scenarios.prob_share_shift_f),
    sales.total_units AS SUM(sales.units_f),
    sales.total_revenue AS SUM(sales.revenue_f) COMMENT = 'Historical revenue ($)',
    sales.avg_price AS AVG(sales.our_price_f),
    sales.avg_competitor_price AS AVG(sales.comp_price_f),
    sales.avg_lead_price AS AVG(sales.lead_f) COMMENT = 'Average LME lead $/tonne',
    price_events.price_move_count AS COUNT(price_events.ev_move_f),
    price_events.avg_price_move AS AVG(price_events.ev_move_f) COMMENT = 'Average historical price move (fraction)',
    price_events.avg_actual_volume_change AS AVG(price_events.ev_actual_f) COMMENT = 'Realized 12-week volume change after past moves (fraction)',
    price_events.avg_predicted_volume_change AS AVG(price_events.ev_pred_f) COMMENT = 'Model-predicted volume change for the same past moves',
    price_events.model_mean_abs_error AS AVG(ABS(price_events.ev_err_f)) COMMENT = 'Model backtest mean absolute error on volume change'
  )
  COMMENT = 'Voltara what-if scenario results and historical sales for Cortex Analyst';

CREATE OR REPLACE TABLE WHATIF_DEMO.SIM.POLICY_DOCS (DOC_ID STRING, TITLE STRING, DOC_TYPE STRING, CUSTOMER_NAME STRING, CONTENT STRING);
INSERT INTO WHATIF_DEMO.SIM.POLICY_DOCS VALUES
('CT-AZ','AutoZone Supply Agreement — Pricing Terms','contract','AutoZone','Price increases require 60 days written notice. No more than one list price increase per contract year, capped at 9% absent a commodity surcharge. Lead surcharge clause: if the LME 3-month average moves more than 12% from the base of $2,100/t, either party may invoke a quarterly surcharge/credit at 10.5 kg per unit. Agreement renews each April; volume commitment 85% of prior-year units.'),
('CT-WM','Walmart Vendor Agreement — Pricing Terms','contract','Walmart','Cost increases require 90 days notice and a cost justification package. Most-favored-customer clause: pricing may not exceed the lowest net price offered to any mass merchant or club account for comparable SKUs. No lead surcharge clause; commodity changes must be negotiated in the annual line review (February). Walmart may re-tender the category after any increase above 5%.'),
('CT-OR','O''Reilly Supply Agreement — Pricing Terms','contract','O''Reilly','45 days notice for price changes. Lead surcharge pass-through permitted monthly when LME moves more than 10% from base. Increases above 7% in a contract year trigger a right to request competitive bids.'),
('CT-AA','Advance Auto Supply Agreement — Pricing Terms','contract','Advance Auto','60 days notice. Lead surcharge applies quarterly with a 15% LME threshold. Annual increase cap of 8% excluding surcharge.'),
('CT-NP','NAPA Distribution Agreement','contract','NAPA','30 days notice for list price changes. Distributor pricing follows Voltara list less 22%. Lead surcharge passes through monthly with no threshold.'),
('CT-CO','Costco Vendor Agreement — Pricing Terms','contract','Costco','Costco requires 120 days notice and reserves the right to delist SKUs whose cost increases exceed member value targets. Most-favored-customer clause covers club channel. No surcharge mechanism.'),
('CT-ID','Interstate Dealer Network Agreement','contract','Interstate Dealer Network','30 days notice. Annual price review each January. Surcharge pass-through monthly.'),
('CT-OEM','OEM Assembly Supply Agreement','contract','OEM Assembly (Auto Co.)','Fixed price for the model year with quarterly lead index adjustment at 100% pass-through. Mid-year price increases are not permitted except via the index.'),
('POL-PRICE','Voltara Pricing Governance Policy','policy',NULL,'Any account price increase above 5% requires VP Commercial approval; above 8% requires CFO approval. Every proposal must include a Monte Carlo scenario showing P10/P50/P90 margin and the probability of margin below baseline. Proposals where probability of margin below baseline exceeds 25% or probability of volume loss over 10% exceeds 40% must include a mitigation plan.'),
('POL-SURCH','Voltara Commodity Surcharge Policy','policy',NULL,'Invoke lead surcharges where contracts allow when the LME 3-month average moves more than 10% from contract base. Hedging desk targets 60% of next-quarter lead needs; a sustained 20% lead shock triggers a hedge ratio review within 5 business days.'),
('POL-COMP','Competitive Response Playbook','policy',NULL,'Historical data shows competitors match roughly 30-45% of our increases and undercut 25-35% of the time, typically within 2-12 weeks. When a competitor undercuts by 5% or more at a top-5 account, sales must respond within 10 business days with a value or rebate offer rather than a list price cut.'),
('POL-APPROVE','Decision Approval & Audit Procedure','policy',NULL,'Agent-generated pricing recommendations are proposals only. A named approver must sign off in the decision log with the scenario ID, assumptions, and range of outcomes before any price change is communicated to a customer.');

CREATE OR REPLACE CORTEX SEARCH SERVICE WHATIF_DEMO.SIM.POLICY_SEARCH
  ON CONTENT
  ATTRIBUTES DOC_TYPE, CUSTOMER_NAME
  WAREHOUSE = AICOLLEGE
  TARGET_LAG = '1 day'
  AS SELECT DOC_ID, TITLE, DOC_TYPE, CUSTOMER_NAME, CONTENT FROM WHATIF_DEMO.SIM.POLICY_DOCS;
