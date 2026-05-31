"""
Shared Snowflake connection config. Reads from environment variables.
Copy .env.example to .env and fill in your credentials, then:
  export $(cat .env | xargs) && python3 setup_snowflake.py
"""

import os
import snowflake.connector


def get_connection():
    return snowflake.connector.connect(
        account=os.environ["SNOWFLAKE_ACCOUNT"],
        user=os.environ["SNOWFLAKE_USER"],
        password=os.environ["SNOWFLAKE_PASSWORD"],
        warehouse=os.environ.get("SNOWFLAKE_WAREHOUSE", "compute_wh"),
        database=os.environ.get("SNOWFLAKE_DATABASE", "ghostnet"),
        schema=os.environ.get("SNOWFLAKE_SCHEMA", "fall_detect"),
    )
