from pathlib import Path
from shared.outputs import create_run_directory

from .logistic_model import init_weights, describe_model
from .training import train, calculate_accuracy
from .visualizations import plot_training_loss, plot_decision_boundary
from .config import FEATURE_DIM


def run(X_train, X_test, y_train_signed, y_test_signed, output_dir=None):
    # Logistic regression uses {0, 1}; convert labels, not the dataset.
    y_train = (y_train_signed > 0).astype(float)
    y_test = (y_test_signed > 0).astype(float)
    output_dir = Path(output_dir) if output_dir is not None else create_run_directory() / "logistic_regression"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\\n" + "=" * 60)
    print(" Logistic Regression Model ")
    print("=" * 60)
    print(f"Training samples: {len(X_train)} | Testing samples: {len(X_test)}")
    print(f"Features: {FEATURE_DIM}")
    weights = init_weights()
    print("\\nModel:")
    print(describe_model(weights, X_train[0]))
    print("\\nStarting Optimization...")
    trained_weights, loss_history = train(weights, X_train, y_train)
    train_acc = calculate_accuracy(trained_weights, X_train, y_train)
    test_acc = calculate_accuracy(trained_weights, X_test, y_test)
    print(f"\\nTraining Accuracy: {train_acc * 100:.2f}%")
    print(f"Testing Accuracy : {test_acc * 100:.2f}%")

    plot_training_loss(
        loss_history, save_path=output_dir / "logistic_training_loss.png"
    )
    plot_decision_boundary(
        trained_weights,
        X_train,
        y_train,
        save_path=output_dir / "logistic_decision_boundary.png",
    )
    return trained_weights, loss_history
