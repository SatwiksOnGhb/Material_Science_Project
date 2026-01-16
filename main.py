# main.py
import argparse
from pathlib import Path

from src.data_preprocessing import preprocess_dataset
from src.train_random_forest import train_random_forest
from src.train_xgboost import train_xgboost


def main():
    parser = argparse.ArgumentParser(
        description="Materials Strength Prediction Pipeline (Preprocess -> Train RF/XGB)"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # ----------------------------
    # PREPROCESS
    # ----------------------------
    pprep = sub.add_parser("preprocess", help="Clean raw dataset and save to data/processed/")
    pprep.add_argument(
        "--input",
        default="al_data.csv",
        help="Path to raw CSV (default: al_data.csv)",
    )
    pprep.add_argument(
        "--output",
        default="al_data.csv",
        help="Path to save cleaned CSV (default: al_data.csv)",
    )
    pprep.add_argument(
        "--drop",
        nargs="*",
        default=[],
        help="Column names to drop (IDs, names, serial numbers). Example: --drop ID SampleNo AlloyName",
    )
    pprep.add_argument(
        "--targets",
        nargs="*",
        default=[],
        help="(Optional) Target column names to report missing counts for. Example: --targets Tensile_Strength Fatigue_Strength",
    )
    pprep.add_argument(
        "--no_numeric_convert",
        action="store_true",
        help="Disable numeric-like string conversion (e.g., '123 MPa' -> 123).",
    )

    # ----------------------------
    # TRAIN
    # ----------------------------
    ptrain = sub.add_parser("train", help="Train RF and XGBoost models for a given target column")
    ptrain.add_argument(
        "--data",
        default="al_data.csv",
        help="Path to cleaned CSV (default: al_data.csv)",
    )
    ptrain.add_argument(
        "--target",
        required=True,
        help="Target column name to predict (must exist in cleaned CSV).",
    )
    ptrain.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Test split fraction (default: 0.2)",
    )
    ptrain.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )

    args = parser.parse_args()

    if args.cmd == "preprocess":
        preprocess_dataset(
            input_csv=Path(args.input),
            output_csv=Path(args.output),
            drop_cols=args.drop,
            target_cols=args.targets,
            convert_numeric_like=(not args.no_numeric_convert),
        )
        print(f"\n✅ Preprocessing complete -> {args.output}")

    elif args.cmd == "train":
        data_path = Path(args.data)
        if not data_path.exists():
            raise FileNotFoundError(
                f"Cleaned dataset not found: {data_path}\n"
                f"Run:\n  python main.py preprocess --input data/raw/dataset.csv"
            )

        print("=== Training Random Forest ===")
        train_random_forest(
            cleaned_csv=data_path,
            target=args.target,
            test_size=args.test_size,
            seed=args.seed,
        )

        print("\n=== Training XGBoost ===")
        train_xgboost(
            cleaned_csv=data_path,
            target=args.target,
            test_size=args.test_size,
            seed=args.seed,
        )

        print("\n✅ Training complete.")
        print("Check outputs:")
        print("  - models/")
        print("  - results/metrics/")
        print("  - results/plots/")


if __name__ == "__main__":
    main()
