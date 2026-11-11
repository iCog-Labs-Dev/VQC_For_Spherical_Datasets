import numpy as np
from pathlib import Path
from shared.outputs import create_run_directory

from .vqc import init_weights, draw_circuit
from .training import train, calculate_accuracy
from .visualizations import plot_training_loss, plot_decision_boundary
from .config import FEATURE_DIM, NUM_LAYERS


def run(X_train, X_test, y_train, y_test, output_dir=None):
    # Quantum rotation angles require bounded feature values. This
    # model-specific adaptation is deliberately outside shared loading.
    X_train = np.clip(X_train, -np.pi, np.pi)
    X_test = np.clip(X_test, -np.pi, np.pi)

    output_dir = Path(output_dir) if output_dir is not None else create_run_directory() / "vqc"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\\n" + "=" * 60)
    print(" VQC with Intermediate Encoding ")
    print("=" * 60)
    print(f"Training samples: {len(X_train)} | Testing samples: {len(X_test)}")
    print(f"Features: {FEATURE_DIM} | Layers: {NUM_LAYERS}")
    weights = init_weights()
    print("\\nCircuit Architecture:")
    print(draw_circuit(weights, X_train[0]))
    print("\\nStarting Quantum Optimization Pipeline...")
    trained_weights, loss_history = train(weights, X_train, y_train)
    train_acc = calculate_accuracy(trained_weights, X_train, y_train)
    test_acc = calculate_accuracy(trained_weights, X_test, y_test)
    print(f"\\nTraining Accuracy: {train_acc * 100:.2f}%")
    print(f"Testing Accuracy : {test_acc * 100:.2f}%")

    plot_training_loss(loss_history, save_path=output_dir / "vqc_training_loss.png")
    plot_decision_boundary(
        trained_weights,
        X_train,
        y_train,
        save_path=output_dir / "vqc_decision_boundary.png",
    )
    return trained_weights, loss_history
