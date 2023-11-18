from pathlib import Path

import numpy as np
import torch

# import torchvision
import ujson

PROPERTIES_FOLDER = Path.cwd().parent.joinpath("data", "properties")


def to_numpy(x_t):
    return x_t.clone().detach().cpu().numpy()


def predict(ort_session, x_t):
    ort_inputs = {ort_session.get_inputs()[0].name: x_t}
    ort_outs = ort_session.run(None, ort_inputs)
    return np.argmax(ort_outs[0]), ort_outs[0]
