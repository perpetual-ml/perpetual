import os

from dotenv import load_dotenv

env_path = os.path.join(os.path.dirname(__file__), "../../../.env")
load_dotenv(dotenv_path=env_path)

os.environ["KAGGLE_USERNAME"] = os.getenv("KAGGLE_USERNAME")
os.environ["KAGGLE_KEY"] = os.getenv("KAGGLE_API_TOKEN")

import kaggle  # noqa: E402

# Since this script runs in the directory, target is current dir
target_dir = "."

# Authenticate and download dataset
kaggle.api.authenticate()
try:
    print("Downloading march-machine-learning-mania-2026 competition data...")
    kaggle.api.competition_download_files(
        "march-machine-learning-mania-2026", path=target_dir, quiet=False
    )
    print("Download finished. Extracting...")
    # Extract the zip file
    import zipfile

    zip_path = f"{target_dir}/march-machine-learning-mania-2026.zip"
    if os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(target_dir)
        print("Extraction complete.")
        os.remove(zip_path)  # Clean up zip
    else:
        print(f"Zip file not found at {zip_path}")

except Exception as e:
    print(f"Failed to download: {e}")
