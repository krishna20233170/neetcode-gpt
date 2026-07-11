"""Neural network primitives built from scratch (NeetCode ML course).

Each submodule exposes a ``Solution`` class matching the course problem API.
Import from the specific module, e.g.::

    from foundations.gradient_descent import Solution
"""

from . import (
    activations,
    backprop,
    dead_relu_detector,
    digit_classifier,
    gradient_descent,
    linear_regression,
    linear_regression_training,
    loss,
    mlp,
    multi_layer_backprop,
    neuron,
    pytorch_basics,
    sentiment,
    softmax,
    training_diagnostics,
    training_loop,
    weight_init,
)

__all__ = [
    "activations",
    "backprop",
    "dead_relu_detector",
    "digit_classifier",
    "gradient_descent",
    "linear_regression",
    "linear_regression_training",
    "loss",
    "mlp",
    "multi_layer_backprop",
    "neuron",
    "pytorch_basics",
    "sentiment",
    "softmax",
    "training_diagnostics",
    "training_loop",
    "weight_init",
]
