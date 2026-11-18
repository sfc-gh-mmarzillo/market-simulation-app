-- Engine v3 (market strategy). Run from setup/:  snow sql -f 12_strategy_engine.sql
CREATE STAGE IF NOT EXISTS WHATIF_DEMO.SIM.CODE;
PUT file://strategy_engine.py @WHATIF_DEMO.SIM.CODE AUTO_COMPRESS = FALSE OVERWRITE = TRUE;

CREATE OR REPLACE PROCEDURE WHATIF_DEMO.SIM.RUN_STRATEGY(SPEC VARIANT, CONFIG VARIANT DEFAULT NULL, QUESTION STRING DEFAULT NULL)
RETURNS VARIANT LANGUAGE PYTHON RUNTIME_VERSION = '3.11' PACKAGES = ('snowflake-snowpark-python', 'numpy', 'pandas')
IMPORTS = ('@WHATIF_DEMO.SIM.CODE/strategy_engine.py') HANDLER = 'strategy_engine.run_strategy'
COMMENT = 'Market strategy Monte Carlo (cut -> competitor pivot -> adapt). Writes STRATEGY_RESULTS.';

CREATE OR REPLACE PROCEDURE WHATIF_DEMO.SIM.RUN_STRATEGY_SWEEP(SPEC VARIANT, CUTS VARIANT, MONTHS VARIANT, CONFIG VARIANT DEFAULT NULL,
    MAX_LOSS_RISK FLOAT DEFAULT 0.25, QUESTION STRING DEFAULT NULL)
RETURNS VARIANT LANGUAGE PYTHON RUNTIME_VERSION = '3.11' PACKAGES = ('snowflake-snowpark-python', 'numpy', 'pandas')
IMPORTS = ('@WHATIF_DEMO.SIM.CODE/strategy_engine.py') HANDLER = 'strategy_engine.run_strategy_sweep';

CREATE OR REPLACE PROCEDURE WHATIF_DEMO.SIM.DESCRIBE_MARKET_MODEL(MARKET STRING DEFAULT NULL, COMPETITOR STRING DEFAULT NULL)
RETURNS VARIANT LANGUAGE PYTHON RUNTIME_VERSION = '3.11' PACKAGES = ('snowflake-snowpark-python', 'numpy', 'pandas')
IMPORTS = ('@WHATIF_DEMO.SIM.CODE/strategy_engine.py') HANDLER = 'strategy_engine.describe_market_model'
COMMENT = 'Agent tool: model card for the market strategy engine (models, fitted parameters, uncertainty sources, assumptions).';

CREATE OR REPLACE PROCEDURE WHATIF_DEMO.SIM.MARKET_OUTLOOK(MARKET STRING)
RETURNS VARIANT LANGUAGE PYTHON RUNTIME_VERSION = '3.11' PACKAGES = ('snowflake-snowpark-python', 'numpy', 'pandas')
IMPORTS = ('@WHATIF_DEMO.SIM.CODE/strategy_engine.py') HANDLER = 'strategy_engine.market_outlook'
COMMENT = 'Agent tool: market size, growth, player shares/margins/capacity and competitor moves in or out.';

CREATE OR REPLACE PROCEDURE WHATIF_DEMO.SIM.FIND_SQUEEZE_PRECEDENTS(MARKET STRING DEFAULT NULL, COMPETITOR STRING DEFAULT NULL, CUT_PCT FLOAT DEFAULT NULL)
RETURNS VARIANT LANGUAGE PYTHON RUNTIME_VERSION = '3.11' PACKAGES = ('snowflake-snowpark-python', 'numpy', 'pandas')
IMPORTS = ('@WHATIF_DEMO.SIM.CODE/strategy_engine.py') HANDLER = 'strategy_engine.find_squeeze_precedents'
COMMENT = 'Agent tool: past price-cut episodes and what each competitor actually did (pivot, destination, capacity change).';

CREATE OR REPLACE PROCEDURE WHATIF_DEMO.SIM.STRATEGY_TOOL(TARGET_MARKET STRING, TARGET_COMPETITOR STRING DEFAULT 'VOLTX',
    CUT_PCT FLOAT DEFAULT 10, CUT_MONTHS INT DEFAULT 6, ADAPT_RULE STRING DEFAULT 'restore_after', EXIT_THRESHOLD_PCT FLOAT DEFAULT 30,
    FOLLOW_CUT_PCT FLOAT DEFAULT 5, HORIZON_MONTHS INT DEFAULT 24, DISABLE_UNCERTAINTY STRING DEFAULT NULL,
    LEAD_SHOCK_PCT FLOAT DEFAULT 0, LABEL STRING DEFAULT NULL)
RETURNS VARIANT LANGUAGE PYTHON RUNTIME_VERSION = '3.11' PACKAGES = ('snowflake-snowpark-python', 'numpy', 'pandas')
IMPORTS = ('@WHATIF_DEMO.SIM.CODE/strategy_engine.py') HANDLER = 'strategy_engine.tool_run_strategy'
COMMENT = 'Agent tool: run a starve-pivot-adapt strategy (whole-number percents).';

CREATE OR REPLACE PROCEDURE WHATIF_DEMO.SIM.STRATEGY_SWEEP_TOOL(TARGET_MARKET STRING, TARGET_COMPETITOR STRING DEFAULT 'VOLTX',
    CUT_PCTS STRING DEFAULT '5,10,15,20', CUT_MONTHS_LIST STRING DEFAULT '3,6,9', ADAPT_RULE STRING DEFAULT 'restore_on_exit',
    MAX_LOSS_RISK_PCT FLOAT DEFAULT 25)
RETURNS VARIANT LANGUAGE PYTHON RUNTIME_VERSION = '3.11' PACKAGES = ('snowflake-snowpark-python', 'numpy', 'pandas')
IMPORTS = ('@WHATIF_DEMO.SIM.CODE/strategy_engine.py') HANDLER = 'strategy_engine.tool_strategy_sweep'
COMMENT = 'Agent tool: sweep cut depth x duration; returns best strategy within the loss-risk limit.';

CREATE OR REPLACE PROCEDURE WHATIF_DEMO.SIM.EXPLAIN_STRATEGY_TOOL(SCENARIO_ID STRING)
RETURNS VARIANT LANGUAGE PYTHON RUNTIME_VERSION = '3.11' PACKAGES = ('snowflake-snowpark-python', 'numpy', 'pandas')
IMPORTS = ('@WHATIF_DEMO.SIM.CODE/strategy_engine.py') HANDLER = 'strategy_engine.tool_explain_strategy'
COMMENT = 'Agent tool: drivers + engine stress tests (no pivots, forced destinations, no minor cuts, lead +20%).';

-- determinism check: all uncertainty off => P10 = P90
CALL WHATIF_DEMO.SIM.RUN_STRATEGY(PARSE_JSON('{"target_market":"AFTERMARKET","target_competitor":"VOLTX","cut_pct":0.12,"cut_months":6,"adapt":"restore_on_exit","label":"[test] deterministic"}'),
  PARSE_JSON('{"n_runs":500,"uncertainty":{"elasticity":false,"share_response":false,"competitor_cost":false,"pivot_timing":false,"pivot_destination":false,"competitor_price_cuts":false,"market_growth":false,"lead":false,"demand_noise":false}}'));
