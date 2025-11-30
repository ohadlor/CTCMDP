from datetime import datetime

import numpy as np
from omegaconf import DictConfig, OmegaConf
from coolname import generate_slug


def custom_dir_resolver(agent_cfg: DictConfig, env_cfg: DictConfig):
    """
    Generate a custom directory name for the run.

    Parameters
    ----------
    agent_cfg : DictConfig
        The agent configuration.
    env_cfg : DictConfig
        The environment configuration.

    Returns
    -------
    str
        The custom directory name.
    """
    cool_name = generate_slug(2)
    timestamp = datetime.now().strftime("%Y%m%d")
    run_name = f"{timestamp}_{cool_name}_{env_cfg.name}"
    if "variant" in agent_cfg:
        run_name += f"_{agent_cfg.variant}"
    run_name += f"_{agent_cfg.model._target_.split('.')[-1]}"
    shrink_factor = int(10 * agent_cfg.get("shrink_factor", 0))
    if shrink_factor or agent_cfg.get("boot_with_shrink_factor", False):
        run_name += "_shrink"
    return run_name


def int_product_resolver(a: float, b: float) -> int:
    return int(a * b)


def floor_div_resolver(a: float, b: float) -> int:
    return a // b


def seed_sequence_resolver(n: int, entropy: int) -> str:
    seed_sequence = np.random.SeedSequence(entropy)
    seeds = seed_sequence.generate_state(n)
    return seeds.tolist()
    return ",".join(map(str, seeds))


def linspace_resolver(start: float, end: float, steps: int) -> str:
    interval = np.linspace(start, end, steps)
    return ",".join(map(str, interval))


def logspace_resolver(start: float, end: float, steps: int) -> str:
    interval = np.logspace(start, end, steps)
    return ",".join(map(str, interval))


resolvers = {
    "seed_sequence": seed_sequence_resolver,
    "int_product": int_product_resolver,
    "floor_div": floor_div_resolver,
    "custom_dir": custom_dir_resolver,
    "linspace": linspace_resolver,
    "logspace": logspace_resolver,
}
for resolver_name, resolver in resolvers.items():
    if not OmegaConf.has_resolver(resolver_name):
        OmegaConf.register_new_resolver(resolver_name, resolver)
