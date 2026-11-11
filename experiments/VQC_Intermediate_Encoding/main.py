"""Single experiment entry point: load and preprocess the dataset once."""
from shared.data import load_academic_performance_data
from shared.outputs import create_run_directory
from models.linear_regression.experiment import run as run_linear_regression
from models.logistic_regression.experiment import run as run_logistic_regression
from models.vqc.experiment import run as run_vqc
from models.vqc.config import NUM_LAYERS, ENCODING_METHOD


def main():
    split = load_academic_performance_data(n_samples=100)
    X_train, X_test = split.X_train, split.X_test
    y_train, y_test = split.y_train, split.y_test

    run_dir = create_run_directory(NUM_LAYERS, ENCODING_METHOD)
    print("\\nShared dataset split")
    print(f"Training samples: {len(X_train)} | Testing samples: {len(X_test)}")
    print(f"Features: {X_train.shape[1]}")
    print(f"Saving this execution's figures under: {run_dir}")

    run_linear_regression(
        X_train, X_test, y_train, y_test,
        output_dir=run_dir / "linear_regression",
    )
    run_logistic_regression(
        X_train, X_test, y_train, y_test,
        output_dir=run_dir / "logistic_regression",
    )
    run_vqc(
        X_train, X_test, y_train, y_test,
        output_dir=run_dir / "vqc",
    )


if __name__ == "__main__":
    main()
