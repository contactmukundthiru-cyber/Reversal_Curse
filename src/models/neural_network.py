"""
Neural Network Model for Reversal Curse Prediction.

This module implements a deep learning approach to predict curse magnitude
from knowledge structures. The network learns to map pre/post reversal
features to observed curse magnitudes, complementing the Bayesian model
with data-driven pattern recognition.

Architecture:
- Multi-layer perceptron with residual connections
- Separate encoding branches for pre and post states
- Attention mechanism over affected propositions
- Regularization for generalization to new domains
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.special import expit as sigmoid

from .bayesian import KnowledgeStructure, ReversalEvent

logger = logging.getLogger(__name__)


@dataclass
class NetworkConfig:
    """Configuration for the neural network architecture."""

    input_dim: int = 24  # Features from pre/post states
    hidden_dims: List[int] = None
    dropout_rate: float = 0.2
    learning_rate: float = 0.001
    l2_regularization: float = 0.01
    batch_size: int = 32
    max_epochs: int = 100
    early_stopping_patience: int = 10

    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [64, 32, 16]


class ReversalCurseNet:
    """
    Neural network for predicting reversal curse magnitude.

    This is a NumPy-based implementation that doesn't require PyTorch/TensorFlow,
    making it suitable for environments without GPU dependencies.

    The network architecture:
    1. Feature extraction from pre/post knowledge structures
    2. Dense layers with ReLU activation
    3. Residual connections for gradient flow
    4. Output layer with sigmoid for bounded prediction
    """

    def __init__(self, config: Optional[NetworkConfig] = None):
        """
        Initialize the neural network.

        Parameters
        ----------
        config : NetworkConfig, optional
            Network configuration
        """
        self.config = config or NetworkConfig()
        self.weights: Dict[str, np.ndarray] = {}
        self.biases: Dict[str, np.ndarray] = {}
        self.trained = False
        self.training_history: List[Dict[str, float]] = []

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights using He initialization."""
        np.random.seed(42)  # For reproducibility

        dims = [self.config.input_dim] + self.config.hidden_dims + [1]

        for i in range(len(dims) - 1):
            fan_in = dims[i]
            fan_out = dims[i + 1]

            # He initialization
            std = np.sqrt(2.0 / fan_in)
            self.weights[f"W{i}"] = np.random.randn(fan_in, fan_out) * std
            self.biases[f"b{i}"] = np.zeros(fan_out)

    def _extract_features(self, reversal: ReversalEvent) -> np.ndarray:
        """
        Extract feature vector from a reversal event.

        Features include:
        - Pre-state statistics
        - Post-state statistics
        - Reversal characteristics
        - Structural changes
        """
        pre = reversal.pre_state
        post = reversal.post_state

        # Pre-state features
        pre_features = [
            pre.mean_belief_strength,
            pre.mean_uncertainty,
            pre.structural_coherence,
            pre.n_propositions,
            len(pre.connections),
            np.std(list(pre.propositions.values())) if pre.propositions else 0,
            len(pre.acquisition_order),
            np.mean(list(pre.exposure_counts.values())) if pre.exposure_counts else 0,
        ]

        # Post-state features
        post_features = [
            post.mean_belief_strength,
            post.mean_uncertainty,
            post.structural_coherence,
            post.n_propositions,
            len(post.connections),
            np.std(list(post.propositions.values())) if post.propositions else 0,
            len(post.acquisition_order),
            np.mean(list(post.exposure_counts.values())) if post.exposure_counts else 0,
        ]

        # Reversal-specific features
        reversal_features = [
            reversal.reversal_strength,
            len(reversal.affected_propositions),
            reversal.structural_disruption,
            reversal.connection_disruption,
            self._compute_belief_change_asymmetry(reversal),
            self._compute_temporal_centrality(reversal),
            self._encode_reversal_type(reversal.reversal_type),
            self._compute_network_damage(reversal),
        ]

        features = np.array(pre_features + post_features + reversal_features)
        return features

    def _compute_belief_change_asymmetry(self, reversal: ReversalEvent) -> float:
        """Compute asymmetry in belief changes (weakening vs strengthening)."""
        pre = reversal.pre_state.propositions
        post = reversal.post_state.propositions

        weakenings = 0
        strengthenings = 0

        for prop in reversal.affected_propositions:
            if prop in pre and prop in post:
                delta = post[prop] - pre[prop]
                if delta < 0:
                    weakenings += abs(delta)
                else:
                    strengthenings += delta

        total = weakenings + strengthenings
        if total == 0:
            return 0.5

        return weakenings / total

    def _compute_temporal_centrality(self, reversal: ReversalEvent) -> float:
        """Compute how central (early) affected propositions are."""
        order = reversal.pre_state.acquisition_order
        affected = set(reversal.affected_propositions)

        if not order or not affected:
            return 0.5

        positions = []
        for i, prop in enumerate(order):
            if prop in affected:
                positions.append(i / len(order))

        if not positions:
            return 0.5

        # Lower mean position = earlier acquisition = higher centrality
        return 1.0 - np.mean(positions)

    def _encode_reversal_type(self, reversal_type: str) -> float:
        """Encode reversal type as numerical feature."""
        type_mapping = {
            "contradiction": 1.0,
            "paradigm_shift": 0.8,
            "refinement": 0.4,
            "extension": 0.2,
        }
        return type_mapping.get(reversal_type, 0.5)

    def _compute_network_damage(self, reversal: ReversalEvent) -> float:
        """Compute damage to knowledge network structure."""
        pre_connections = reversal.pre_state.connections
        post_connections = reversal.post_state.connections
        affected = set(reversal.affected_propositions)

        if not pre_connections:
            return 0.0

        damaged_connections = 0
        total_relevant = 0

        for (p1, p2), strength in pre_connections.items():
            if p1 in affected or p2 in affected:
                total_relevant += 1
                if (p1, p2) not in post_connections:
                    damaged_connections += 1
                elif post_connections[(p1, p2)] < strength * 0.5:
                    damaged_connections += 0.5

        if total_relevant == 0:
            return 0.0

        return damaged_connections / total_relevant

    def _relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU activation function."""
        return np.maximum(0, x)

    def _relu_derivative(self, x: np.ndarray) -> np.ndarray:
        """Derivative of ReLU for backpropagation."""
        return (x > 0).astype(float)

    def _forward(
        self,
        x: np.ndarray,
        training: bool = False
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Forward pass through the network.

        Parameters
        ----------
        x : np.ndarray
            Input features (batch_size, input_dim)
        training : bool
            Whether in training mode (applies dropout)

        Returns
        -------
        Tuple[np.ndarray, Dict]
            Output predictions and intermediate activations for backprop
        """
        activations = {"a0": x}
        current = x

        n_layers = len(self.config.hidden_dims) + 1

        for i in range(n_layers):
            # Linear transformation
            z = current @ self.weights[f"W{i}"] + self.biases[f"b{i}"]
            activations[f"z{i}"] = z

            if i < n_layers - 1:
                # Hidden layer: ReLU activation
                current = self._relu(z)

                # Dropout during training
                if training and self.config.dropout_rate > 0:
                    mask = np.random.binomial(
                        1, 1 - self.config.dropout_rate, size=current.shape
                    ) / (1 - self.config.dropout_rate)
                    current = current * mask
                    activations[f"dropout_mask{i}"] = mask
            else:
                # Output layer: sigmoid for bounded output
                current = sigmoid(z)

            activations[f"a{i+1}"] = current

        return current, activations

    def _backward(
        self,
        y_true: np.ndarray,
        activations: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Backward pass to compute gradients.

        Parameters
        ----------
        y_true : np.ndarray
            True labels
        activations : Dict
            Intermediate activations from forward pass

        Returns
        -------
        Dict[str, np.ndarray]
            Gradients for weights and biases
        """
        gradients = {}
        m = y_true.shape[0]  # Batch size

        n_layers = len(self.config.hidden_dims) + 1

        # Output layer gradient (BCE loss derivative)
        y_pred = activations[f"a{n_layers}"]
        dz = y_pred - y_true.reshape(-1, 1)

        for i in range(n_layers - 1, -1, -1):
            a_prev = activations[f"a{i}"]

            # Weight gradient
            gradients[f"dW{i}"] = (a_prev.T @ dz) / m

            # Add L2 regularization
            gradients[f"dW{i}"] += self.config.l2_regularization * self.weights[f"W{i}"]

            # Bias gradient
            gradients[f"db{i}"] = np.mean(dz, axis=0)

            if i > 0:
                # Propagate gradient to previous layer
                dz = dz @ self.weights[f"W{i}"].T
                dz = dz * self._relu_derivative(activations[f"z{i-1}"])

        return gradients

    def _update_weights(self, gradients: Dict[str, np.ndarray]):
        """Update weights using gradient descent."""
        n_layers = len(self.config.hidden_dims) + 1

        for i in range(n_layers):
            self.weights[f"W{i}"] -= self.config.learning_rate * gradients[f"dW{i}"]
            self.biases[f"b{i}"] -= self.config.learning_rate * gradients[f"db{i}"]

    def fit(
        self,
        reversals: List[ReversalEvent],
        magnitudes: np.ndarray,
        validation_split: float = 0.2,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Train the neural network on reversal data.

        Parameters
        ----------
        reversals : List[ReversalEvent]
            Training reversal events
        magnitudes : np.ndarray
            Observed curse magnitudes
        validation_split : float
            Fraction of data for validation
        verbose : bool
            Whether to print training progress

        Returns
        -------
        Dict[str, Any]
            Training results
        """
        # Extract features
        X = np.array([self._extract_features(r) for r in reversals])
        y = magnitudes.astype(float)

        # Normalize features
        self.feature_mean = X.mean(axis=0)
        self.feature_std = X.std(axis=0) + 1e-8
        X = (X - self.feature_mean) / self.feature_std

        # Split into train/validation
        n_val = int(len(X) * validation_split)
        indices = np.random.permutation(len(X))

        val_idx = indices[:n_val]
        train_idx = indices[n_val:]

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        self.training_history = []

        for epoch in range(self.config.max_epochs):
            # Shuffle training data
            perm = np.random.permutation(len(X_train))
            X_train_shuffled = X_train[perm]
            y_train_shuffled = y_train[perm]

            # Mini-batch training
            epoch_loss = 0
            n_batches = max(1, len(X_train) // self.config.batch_size)

            for batch_idx in range(n_batches):
                start = batch_idx * self.config.batch_size
                end = start + self.config.batch_size

                X_batch = X_train_shuffled[start:end]
                y_batch = y_train_shuffled[start:end]

                # Forward pass
                y_pred, activations = self._forward(X_batch, training=True)

                # Compute loss (MSE)
                batch_loss = np.mean((y_pred.flatten() - y_batch) ** 2)
                epoch_loss += batch_loss

                # Backward pass
                gradients = self._backward(y_batch, activations)

                # Update weights
                self._update_weights(gradients)

            epoch_loss /= n_batches

            # Validation
            val_pred, _ = self._forward(X_val, training=False)
            val_loss = np.mean((val_pred.flatten() - y_val) ** 2)

            self.training_history.append({
                "epoch": epoch,
                "train_loss": epoch_loss,
                "val_loss": val_loss
            })

            if verbose and epoch % 10 == 0:
                logger.info(
                    f"Epoch {epoch}: train_loss={epoch_loss:.4f}, val_loss={val_loss:.4f}"
                )

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best weights
                self.best_weights = {k: v.copy() for k, v in self.weights.items()}
                self.best_biases = {k: v.copy() for k, v in self.biases.items()}
            else:
                patience_counter += 1
                if patience_counter >= self.config.early_stopping_patience:
                    if verbose:
                        logger.info(f"Early stopping at epoch {epoch}")
                    break

        # Restore best weights
        if hasattr(self, 'best_weights'):
            self.weights = self.best_weights
            self.biases = self.best_biases

        self.trained = True

        # Final evaluation
        train_pred, _ = self._forward(X_train, training=False)
        val_pred, _ = self._forward(X_val, training=False)

        train_r2 = 1 - np.sum((train_pred.flatten() - y_train) ** 2) / np.sum((y_train - y_train.mean()) ** 2)
        val_r2 = 1 - np.sum((val_pred.flatten() - y_val) ** 2) / np.sum((y_val - y_val.mean()) ** 2)

        return {
            "train_loss": np.mean((train_pred.flatten() - y_train) ** 2),
            "val_loss": best_val_loss,
            "train_r2": train_r2,
            "val_r2": val_r2,
            "epochs_trained": len(self.training_history),
            "history": self.training_history
        }

    def predict(
        self,
        reversals: Union[ReversalEvent, List[ReversalEvent]]
    ) -> np.ndarray:
        """
        Predict curse magnitude for reversal events.

        Parameters
        ----------
        reversals : ReversalEvent or List[ReversalEvent]
            Events to predict

        Returns
        -------
        np.ndarray
            Predicted curse magnitudes
        """
        if not self.trained:
            logger.warning("Model not trained, using random initialization")

        if isinstance(reversals, ReversalEvent):
            reversals = [reversals]

        X = np.array([self._extract_features(r) for r in reversals])

        # Normalize using training statistics
        if hasattr(self, 'feature_mean'):
            X = (X - self.feature_mean) / self.feature_std

        predictions, _ = self._forward(X, training=False)
        return predictions.flatten()

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Estimate feature importance based on weight magnitudes.

        Returns
        -------
        Dict[str, float]
            Feature importance scores
        """
        feature_names = [
            "pre_mean_belief", "pre_uncertainty", "pre_coherence",
            "pre_n_props", "pre_n_connections", "pre_belief_std",
            "pre_order_len", "pre_exposure_mean",
            "post_mean_belief", "post_uncertainty", "post_coherence",
            "post_n_props", "post_n_connections", "post_belief_std",
            "post_order_len", "post_exposure_mean",
            "reversal_strength", "n_affected", "structural_disruption",
            "connection_disruption", "belief_asymmetry", "temporal_centrality",
            "reversal_type", "network_damage"
        ]

        # Use first layer weights as importance proxy
        first_layer_weights = np.abs(self.weights["W0"]).mean(axis=1)

        importance = {}
        for i, name in enumerate(feature_names[:len(first_layer_weights)]):
            importance[name] = float(first_layer_weights[i])

        # Normalize
        total = sum(importance.values())
        if total > 0:
            importance = {k: v / total for k, v in importance.items()}

        return importance


def train_predictor(
    reversals: List[ReversalEvent],
    magnitudes: np.ndarray,
    config: Optional[NetworkConfig] = None,
    **kwargs
) -> Tuple[ReversalCurseNet, Dict[str, Any]]:
    """
    Convenience function to train a curse magnitude predictor.

    Parameters
    ----------
    reversals : List[ReversalEvent]
        Training data
    magnitudes : np.ndarray
        Observed magnitudes
    config : NetworkConfig, optional
        Network configuration
    **kwargs
        Additional arguments passed to fit()

    Returns
    -------
    Tuple[ReversalCurseNet, Dict]
        Trained model and training results
    """
    model = ReversalCurseNet(config)
    results = model.fit(reversals, magnitudes, **kwargs)
    return model, results


def evaluate_model(
    model: ReversalCurseNet,
    test_reversals: List[ReversalEvent],
    test_magnitudes: np.ndarray
) -> Dict[str, float]:
    """
    Evaluate model performance on test data.

    Parameters
    ----------
    model : ReversalCurseNet
        Trained model
    test_reversals : List[ReversalEvent]
        Test reversal events
    test_magnitudes : np.ndarray
        True magnitudes

    Returns
    -------
    Dict[str, float]
        Evaluation metrics
    """
    predictions = model.predict(test_reversals)

    # MSE
    mse = np.mean((predictions - test_magnitudes) ** 2)

    # RMSE
    rmse = np.sqrt(mse)

    # MAE
    mae = np.mean(np.abs(predictions - test_magnitudes))

    # R-squared
    ss_res = np.sum((test_magnitudes - predictions) ** 2)
    ss_tot = np.sum((test_magnitudes - test_magnitudes.mean()) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    # Correlation
    correlation = np.corrcoef(predictions, test_magnitudes)[0, 1]

    return {
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
        "correlation": float(correlation),
        "n_samples": len(test_magnitudes)
    }


class EnsemblePredictor:
    """
    Ensemble combining Bayesian and neural network predictions.

    Uses weighted combination of both models for more robust predictions.
    """

    def __init__(
        self,
        bayesian_weight: float = 0.5,
        neural_config: Optional[NetworkConfig] = None
    ):
        """
        Initialize ensemble predictor.

        Parameters
        ----------
        bayesian_weight : float
            Weight for Bayesian model (neural gets 1 - this)
        neural_config : NetworkConfig, optional
            Configuration for neural network
        """
        from .bayesian import BayesianCurseModel

        self.bayesian_weight = bayesian_weight
        self.bayesian_model = BayesianCurseModel()
        self.neural_model = ReversalCurseNet(neural_config)
        self.trained = False

    def fit(
        self,
        reversals: List[ReversalEvent],
        magnitudes: np.ndarray,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Fit both models.

        Parameters
        ----------
        reversals : List[ReversalEvent]
            Training data
        magnitudes : np.ndarray
            Observed magnitudes

        Returns
        -------
        Dict[str, Any]
            Training results for both models
        """
        # Fit Bayesian model
        bayesian_results = self.bayesian_model.fit(reversals, magnitudes)

        # Fit neural model
        neural_results = self.neural_model.fit(reversals, magnitudes, **kwargs)

        self.trained = True

        return {
            "bayesian": bayesian_results,
            "neural": neural_results
        }

    def predict(
        self,
        reversals: Union[ReversalEvent, List[ReversalEvent]]
    ) -> np.ndarray:
        """
        Make ensemble predictions.

        Parameters
        ----------
        reversals : ReversalEvent or List[ReversalEvent]
            Events to predict

        Returns
        -------
        np.ndarray
            Ensemble predictions
        """
        if isinstance(reversals, ReversalEvent):
            reversals = [reversals]

        # Bayesian predictions
        bayesian_preds = np.array([
            self.bayesian_model.predict_curse_magnitude(r)
            for r in reversals
        ])

        # Neural predictions
        neural_preds = self.neural_model.predict(reversals)

        # Weighted combination
        ensemble_preds = (
            self.bayesian_weight * bayesian_preds +
            (1 - self.bayesian_weight) * neural_preds
        )

        return ensemble_preds

    def optimize_weights(
        self,
        val_reversals: List[ReversalEvent],
        val_magnitudes: np.ndarray
    ) -> float:
        """
        Optimize ensemble weights on validation data.

        Parameters
        ----------
        val_reversals : List[ReversalEvent]
            Validation events
        val_magnitudes : np.ndarray
            True magnitudes

        Returns
        -------
        float
            Optimal Bayesian weight
        """
        from scipy.optimize import minimize_scalar

        bayesian_preds = np.array([
            self.bayesian_model.predict_curse_magnitude(r)
            for r in val_reversals
        ])
        neural_preds = self.neural_model.predict(val_reversals)

        def loss(weight):
            combined = weight * bayesian_preds + (1 - weight) * neural_preds
            return np.mean((combined - val_magnitudes) ** 2)

        result = minimize_scalar(loss, bounds=(0, 1), method='bounded')
        self.bayesian_weight = result.x

        return self.bayesian_weight
