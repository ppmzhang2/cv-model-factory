"""Test YOLOv3 Modles."""
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import pytest

from cvmodel import utils

N = 2
N_CLASS = 80
N_BBOX = 5

# with open("cnn.toml", "rb") as f:
#    dc = tomllib.load(f)
dc = {
    "meta": {
        "name": "CNNBlock",
        "version": "0",
        "output": "x",
    },
    "layer": [{
        "source": "haiku",
        "func": "Conv2D",
        "from": ["x"],
        "to": "x",
        "init": {
            "output_channels": 32,
            "kernel_shape": 3,
            "stride": 2,
            "padding": "SAME",
            "with_bias": False,
        },
    }, {
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
        },
    }, {
        "source": "jax.nn",
        "func": "leaky_relu",
        "from": ["x"],
        "to": "x",
    }],
}


@dataclass(frozen=True)
class Data:
    """Input dataset and output shapes."""
    x: jnp.ndarray
    cfg: dict
    shape: tuple[int, int, int, int, int]


in_shape = (N, 416, 416, 3)
seed = 0
key = jax.random.PRNGKey(seed)
x = jax.random.normal(key, in_shape, dtype=jnp.float32)

dataset = [
    Data(x=x, cfg=dc, shape=(2, 208, 208, 32)),
]


@pytest.mark.parametrize("data", dataset)
def test_resnet_model(data: Data) -> None:
    """Test module `cvmodel.utils`."""
    model = utils.model_factory(dc)
    xfm = utils.get_xfm(model)

    params, states = xfm.init(key, x)
    y, _ = xfm.apply(params, states, key, x)

    assert y.shape == data.shape
