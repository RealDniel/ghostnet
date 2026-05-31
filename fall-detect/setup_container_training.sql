-- ============================================================
-- Snowpark Container Services — Fall Detection Training Setup
-- Run this in a Snowflake worksheet as SYSADMIN or ACCOUNTADMIN
-- ============================================================

USE DATABASE ghostnet;
USE SCHEMA fall_detect;


-- ── 1. Image repository ───────────────────────────────────────────────────────
-- This is where you push the Docker image from your local machine.

CREATE IMAGE REPOSITORY IF NOT EXISTS training_repo;

-- Run this to get the full registry URL you need for docker push:
SHOW IMAGE REPOSITORIES IN SCHEMA ghostnet.fall_detect;
-- Look for the "repository_url" column in the output.
-- It will look like:
--   udlcyth-gdb50567.registry.snowflakecomputing.com/ghostnet/fall_detect/training_repo


-- ── 2. Stage for job spec ─────────────────────────────────────────────────────
CREATE STAGE IF NOT EXISTS job_stage;

-- After running the SQL above, upload snowpark_job.yaml:
--   PUT file:///path/to/snowpark_job.yaml @ghostnet.fall_detect.job_stage


-- ── 3. Stage for trained model output ─────────────────────────────────────────
-- The training container uploads models here when done.
CREATE STAGE IF NOT EXISTS model_stage;


-- ── 4. Compute pool ───────────────────────────────────────────────────────────
-- GPU_NV_S  = 1× A10G GPU (16 GB VRAM) — fastest, uses credits
-- CPU_X64_L = 4 vCPU / 16 GB RAM      — no GPU, slower but cheaper
-- Change INSTANCE_FAMILY if GPU is not available on your Snowflake edition.

CREATE COMPUTE POOL IF NOT EXISTS fall_detect_pool
    MIN_NODES = 1
    MAX_NODES = 1
    INSTANCE_FAMILY = GPU_NV_S
    AUTO_SUSPEND_SECS = 300;

-- Wait for the pool to reach IDLE state before running the job (~2-3 min):
DESCRIBE COMPUTE POOL fall_detect_pool;


-- ── 5. Run the training job ───────────────────────────────────────────────────
-- The container will:
--   1. Load cached Snowflake data (or fetch it fresh)
--   2. Train the 1D CNN
--   3. Upload fall_cnn.keras + scaler.npy to @model_stage

EXECUTE JOB SERVICE
    IN COMPUTE POOL fall_detect_pool
    NAME = ghostnet.fall_detect.fall_detect_training
    FROM @ghostnet.fall_detect.job_stage
    SPECIFICATION_FILE = 'snowpark_job.yaml';


-- ── 6. Monitor the job ────────────────────────────────────────────────────────
-- Check status:
CALL SYSTEM$GET_SERVICE_STATUS('ghostnet.fall_detect.fall_detect_training');

-- Stream logs (run repeatedly to see training progress):
CALL SYSTEM$GET_SERVICE_LOGS('ghostnet.fall_detect.fall_detect_training', '0', 'trainer', 1000);


-- ── 7. Download the trained model ─────────────────────────────────────────────
-- After the job completes, download the model files:
--   GET @ghostnet.fall_detect.model_stage/fall_cnn.keras file:///path/to/models/
--   GET @ghostnet.fall_detect.model_stage/scaler.npy     file:///path/to/models/
-- Or via SnowSQL:
--   snowsql -a UDLCYTH-GDB50567 -u martid24 -q \
--     "GET @ghostnet.fall_detect.model_stage file:///Users/danielmartin/Documents/My\ Projects/Hackathons/ghostnet/fall-detect/models/"


-- ── 8. Cleanup (run after training is done) ───────────────────────────────────
-- DROP SERVICE ghostnet.fall_detect.fall_detect_training;
-- ALTER COMPUTE POOL fall_detect_pool SUSPEND;
