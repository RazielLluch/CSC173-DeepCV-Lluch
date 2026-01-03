import shutil
from pathlib import Path
import pandas as pd
import yaml

# ------------------------
# Configuration
# ------------------------
RAW_DATA_DIR = Path("data\\raw")          # contains train/valid/test
YOLO_DATA_DIR = Path("yolo_dataset") # output directory

SPLIT_MAP = {
    "train": "train",
    "valid": "val",
    "test": "test"
}

POSTURE_COLS = [
    " backwardbadposture",
    " forwardbadposture",
    " goodposture"
]

# ------------------------
# Helper functions
# ------------------------
def get_label(row):
    """Return the posture label for a row."""
    for col in POSTURE_COLS:
        if row[col]:
            return col
    return None


def process_split(split_name):
    src_dir = RAW_DATA_DIR / split_name
    dst_split = YOLO_DATA_DIR / SPLIT_MAP[split_name]

    csv_path = src_dir / "_classes.csv"
    df = pd.read_csv(csv_path)

    # Drop unlabeled rows
    df = df[df[" Unlabeled"] == False].copy()
    df["label"] = df.apply(get_label, axis=1)
    df = df.dropna(subset=["label"])

    for _, row in df.iterrows():
        label = row["label"]
        src_img = src_dir / row["filename"]
        dst_dir = dst_split / label
        dst_dir.mkdir(parents=True, exist_ok=True)

        dst_img = dst_dir / row["filename"]
        shutil.copy(src_img, dst_img)


# ------------------------
# Main preprocessing
# ------------------------
def main():
    YOLO_DATA_DIR.mkdir(exist_ok=True)

    for split in SPLIT_MAP:
        process_split(split)

    # Create dataset.yaml
    dataset_yaml = {
        "path": str(YOLO_DATA_DIR.resolve()),
        "train": "train",
        "val": "val",
        "test": "test",
        "names": POSTURE_COLS
    }

    with open(YOLO_DATA_DIR / "dataset.yaml", "w") as f:
        yaml.dump(dataset_yaml, f, sort_keys=False)

    print("YOLOv8 classification dataset created successfully.")


main()
