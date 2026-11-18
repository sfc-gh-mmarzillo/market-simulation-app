-- Engine v2: shared module engine.py (uploaded to @CODE) used by RUN_SCENARIO, RUN_SWEEP, DESCRIBE_MODEL.
-- Run from the setup/ folder:  snow sql -f 03b_engine_v2.sql   (PUT uploads engine.py)
CREATE STAGE IF NOT EXISTS WHATIF_DEMO.SIM.CODE;
PUT file://engine.py @WHATIF_DEMO.SIM.CODE AUTO_COMPRESS = FALSE OVERWRITE = TRUE;

DROP PROCEDURE IF EXISTS WHATIF_DEMO.SIM.RUN_SCENARIO(STRING, STRING, STRING, FLOAT, STRING, FLOAT, FLOAT, INT, INT, INT);

CREATE OR REPLACE PROCEDURE WHATIF_DEMO.SIM.RUN_SCENARIO(
    SCENARIO_NAME STRING, CUSTOMER_ID STRING, SKU_GROUP STRING,
    PRICE_CHANGE_PCT FLOAT, COMP_MODE STRING, COMP_MOVE_PCT FLOAT, LEAD_SHOCK_PCT FLOAT,
    HORIZON_MONTHS INT, N_RUNS INT, SEED INT, CONFIG VARIANT DEFAULT NULL)
RETURNS VARIANT
LANGUAGE PYTHON
RUNTIME_VERSION = '3.11'
PACKAGES = ('snowflake-snowpark-python', 'numpy', 'pandas')
IMPORTS = ('@WHATIF_DEMO.SIM.CODE/engine.py')
HANDLER = 'engine.run_scenario'
COMMENT = 'Monte Carlo pricing / competitor / lead-cost simulation. Fractions (0.08 = +8%). CONFIG (optional): {"uncertainty": {"elasticity","competitor","churn","lead","demand": bool}, "comp_lag_weeks": [lo,hi], "share_lost_mean": x}. Full config + model snapshot stored in SCENARIO_RESULTS.INPUTS.';

CREATE OR REPLACE PROCEDURE WHATIF_DEMO.SIM.RUN_SWEEP(
    CUSTOMER_ID STRING, SKU_GROUP STRING, FROM_PCT FLOAT, TO_PCT FLOAT, STEP_PCT FLOAT,
    COMP_MODE STRING, COMP_MOVE_PCT FLOAT, LEAD_SHOCK_PCT FLOAT, N_RUNS INT DEFAULT 5000, CONFIG VARIANT DEFAULT NULL,
    MAX_PROB_BELOW_BASELINE FLOAT DEFAULT 0.25)
RETURNS VARIANT
LANGUAGE PYTHON
RUNTIME_VERSION = '3.11'
PACKAGES = ('snowflake-snowpark-python', 'numpy', 'pandas')
IMPORTS = ('@WHATIF_DEMO.SIM.CODE/engine.py')
HANDLER = 'engine.run_sweep'
COMMENT = 'Price sweep (max 25 points) in one call with common random numbers; returns per-point P10/P50/P90 and the best price whose P(margin below baseline) <= limit.';

CREATE OR REPLACE PROCEDURE WHATIF_DEMO.SIM.DESCRIBE_MODEL_BY_ID(CUSTOMER_ID STRING, SKU_GROUP STRING)
RETURNS VARIANT
LANGUAGE PYTHON
RUNTIME_VERSION = '3.11'
PACKAGES = ('snowflake-snowpark-python', 'numpy', 'pandas')
IMPORTS = ('@WHATIF_DEMO.SIM.CODE/engine.py')
HANDLER = 'engine.describe_model'
COMMENT = 'Model card: fitted parameters, uncertainty sources, defaults, fixed assumptions, history backtest.';

-- sanity: deterministic run (all uncertainty off) must give P10 = P90
CALL WHATIF_DEMO.SIM.RUN_SCENARIO('[test] all uncertainty off', 'C01', 'ALL', 0.08, 'MODEL', 0, 0, 12, 2000, 7,
  PARSE_JSON('{"uncertainty": {"elasticity": false, "competitor": false, "churn": false, "lead": false, "demand": false}}'));
