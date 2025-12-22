import os

from databricks.sdk import WorkspaceClient
from dotenv import load_dotenv

load_dotenv()

def run_job(client: WorkspaceClient, job_id: str):
    """
    Run a Databricks job given its job ID.
    """
    client.jobs.run_now(job_id=job_id)


if __name__ == "__main__":
    w = WorkspaceClient(
        host=f"https://{os.getenv('DATABRICKS_SERVER_HOSTNAME')}",
        client_id=os.getenv("DATABRICKS_CLIENT_ID"),
        client_secret=os.getenv("DATABRICKS_CLIENT_SECRET"),
    )
    job_id = "271950941670414"
    run_job(w,job_id)

