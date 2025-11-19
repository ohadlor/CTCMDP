from typing import Iterable
import os

import torch as th
import numpy as np


def polyak_update(
    params: Iterable[th.Tensor],
    target_params: Iterable[th.Tensor],
    tau: float,
) -> None:
    with th.no_grad():
        for param, target_param in zip(params, target_params):
            target_param.data.mul_(1 - tau)
            th.add(target_param.data, param.data, alpha=tau, out=target_param.data)


def safe_mean(a_list: Iterable):
    if not a_list:
        return np.array([])
    else:
        return np.array(a_list).mean()
