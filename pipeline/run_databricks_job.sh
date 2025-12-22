#!/bin/bash
# ABOUTME: Wrapper script to trigger Databricks job via cron
# ABOUTME: Handles environment setup and logging for scheduled execution

# Set PATH to include homebrew binaries (required for cron)
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"

# Change to the script directory
cd /Users/marshall.krassenstein/Desktop/random-projects/detroit-crime-refresh

# Create logs directory if it doesn't exist
mkdir -p databricks/logs

# Log start time
echo "Starting job trigger at $(date)" >> databricks/logs/detroit_job_run_$(date +\%Y\%m\%d).log

# Load environment variables from .env file
if [ -f .env ]; then
    set -a
    . .env
    set +a
fi

# Run the Databricks job and log output
/opt/homebrew/bin/databricks jobs run-now 271950941670414 --no-wait >> databricks/logs/detroit_job_run_$(date +\%Y\%m\%d).log 2>&1

# Log completion
echo "Job triggered at $(date)" >> databricks/logs/detroit_job_run_$(date +\%Y\%m\%d).log