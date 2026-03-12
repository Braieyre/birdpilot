from pathlib import Path
import pandas as pd
import shutil

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CSV_PATH = PROJECT_ROOT / "data/birds/birds.csv"
DATASET_ROOT = PROJECT_ROOT / "data/birds"
OUT_ROOT = PROJECT_ROOT / "data/birds_1pct"

TRAIN_FRAC = 0.1
VALID_FRAC = 0.5
TEST_FRAC = 0.5
RANDOM_STATE = 42


def stratified_sample(group, frac, min_n=1):
    n = max(min_n, int(len(group) * frac))
    n = min(n, len(group))
    return group.sample(n=n, random_state=RANDOM_STATE)


def main():
    df = pd.read_csv(CSV_PATH)
    sampled_parts = []

    for split, frac in [
        ("train", TRAIN_FRAC),
        ("valid", VALID_FRAC),
        ("test", TEST_FRAC),
    ]:
        split_df = df[df["data set"] == split].copy()
        sampled = (
            split_df.groupby("labels", group_keys=False)
            .apply(lambda g: stratified_sample(g, frac))
            .reset_index(drop=True)
        )
        sampled_parts.append(sampled)

    small_df = pd.concat(sampled_parts, ignore_index=True)

    out_root = Path(OUT_ROOT)
    out_root.mkdir(parents=True, exist_ok=True)
    small_df.to_csv(out_root / "birds_small.csv", index=False)

    missing_files = []

    for _, row in small_df.iterrows():
        rel_path = Path(row["filepaths"])
        src = Path(DATASET_ROOT) / rel_path
        dst = out_root / rel_path
        dst.parent.mkdir(parents=True, exist_ok=True)

        if not src.exists():
            missing_files.append(str(src))
            print(f"[WARNING] Missing file: {src}")
            continue

        shutil.copy2(src, dst)

    print("Done.")
    print(f"Total sampled images: {len(small_df)}")
    print(small_df["data set"].value_counts())
    print(f"Missing files: {len(missing_files)}")


if __name__ == "__main__":
    main()