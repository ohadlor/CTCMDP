from typing import Iterable

import torch as th


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
