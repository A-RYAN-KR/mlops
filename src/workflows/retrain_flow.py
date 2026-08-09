import subprocess
import sys

from prefect import flow, task


@task
def pull_data_from_dvc():
    """
    Simulates or executes a DVC pull command to fetch raw data.
    """
    print("Pulling latest data from DVC remote...")
    try:
        # Check if DVC is initialized and execute pull
        result = subprocess.run(
            ["dvc", "pull"], capture_output=True, text=True, check=False
        )
        print(result.stdout)
        if result.returncode != 0:
            print(
                f"Warning: DVC pull returned status code {result.returncode}: {result.stderr}"
            )
    except Exception as e:
        print(f"Warning: Failed to execute DVC command: {e!s}")
    return True


@task
def run_model_training():
    """
    Executes the training pipeline script.
    """
    print("Executing model training pipeline (src/models/train.py)...")
    interpreter = sys.executable
    result = subprocess.run(
        [interpreter, "src/models/train.py"],
        capture_output=True,
        text=True,
        check=False,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(f"Training script failed: {result.stderr}")
        raise RuntimeError(f"Model retraining failed: {result.stderr}")
    print("Training pipeline executed successfully and registered new model version.")
    return True


@flow(name="Model Retraining Flow")
def retrain_flow():
    """
    Periodic model retraining flow.
    """
    pull_data_from_dvc()
    run_model_training()


if __name__ == "__main__":
    print("Starting Prefect periodic retraining flow local runner...")
    retrain_flow.serve(name="periodic-model-retraining", interval=3600)
