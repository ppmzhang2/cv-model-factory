"""Functions Demo."""
from collections.abc import Callable
from importlib import import_module

import haiku as hk
import jax.numpy as jnp


def layer_factory(
    config: dict,
) -> Callable[[dict[str, jnp.ndarray]], dict[str, jnp.ndarray]]:
    """Create a layer.

    Args:
        config (dict):
          - source: library
          - func: function / module name
          - from: list of input tensors
          - init: parameters to initialize a function; None means no
            initialization is needed
          - call: parameters to invoke the function
          - e.g.:
            {
                "source": "haiku",
                "func": "BatchNorm",
                "from": ["x"],
                "to": "x",
                "init": {
                    "create_scale": True,
                    "create_offset": True,
                    "decay_rate": 0.9,
                },
                "call": {
                    "is_training": True,
                }
            }

    Returns:
        dict[str, jnp.ndarray]: dictionary of inputs / intermediate / output
        tensors
    """

    def _func(config: dict) -> Callable:
        module = import_module(config["source"])
        return getattr(module, config["func"])

    def _init_args(config: dict) -> dict | None:
        """Get parameters to initialize a layer function.

        Returns `None` if initialization is not needed.
        """
        return config.get("init", None)

    def _call_args(config: dict) -> dict:
        """Get parameters to call a layer function."""
        return config.get("call", {})

    def infer(dc: dict[str, jnp.ndarray]) -> dict[str, jnp.ndarray]:
        """e.g. invoke the haiku.Module.__call__ function."""
        func = _func(config)
        init_args = _init_args(config)
        call_args = _call_args(config)
        inputs = [dc[i] for i in config["from"]]
        if init_args is None:  # function, not Module
            return {config["to"]: func(*inputs, **call_args)}
        return {config["to"]: func(**init_args)(*inputs, **call_args)}

    return infer


def model_factory(
    kwargs: dict,
) -> Callable[[jnp.ndarray], jnp.ndarray | list[jnp.ndarray]]:
    """Create model from a `dict` configuration."""
    seq_cfg = kwargs["layer"]
    output = kwargs["meta"]["output"]

    def model(x: jnp.ndarray) -> jnp.ndarray | list[jnp.ndarray]:
        res = {"x": x}
        for d in seq_cfg:
            res.update(layer_factory(d)(res))
        if isinstance(output, list):
            return [res[i] for i in output]
        return res[output]

    return model


def get_xfm(model: hk.Module) -> hk.TransformedWithState:
    """Get Transformed model."""
    return hk.transform_with_state(lambda x: model(x))
