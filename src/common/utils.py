from typing import Iterable

import torch as th
import numpy as np


def polyak_update(
    params: Iterable[th.Tensor],
    target_params: Iterable[th.Tensor],
    tau: float,
) -> None:
    """
    Polyak-averaging update of target parameters.

    Parameters
    ----------
    params : Iterable[th.Tensor]
        The parameters to update from.
    target_params : Iterable[th.Tensor]
        The parameters to update.
    tau : float
        The update rate.
    """
    with th.no_grad():
        for param, target_param in zip(params, target_params):
            target_param.data.mul_(1 - tau)
            th.add(target_param.data, param.data, alpha=tau, out=target_param.data)


def safe_mean(a_list: Iterable):
    """
    Compute the mean of a list, returning an empty array if the list is empty.

    Parameters
    ----------
    a_list : Iterable
        The list to compute the mean of.

    Returns
    -------
    np.ndarray
        The mean of the list, or an empty array if the list is empty.
    """
    if not a_list:
        return np.array([])
    else:
        return np.array(a_list).mean()
