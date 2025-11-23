from datetime import datetime

import numpy as np
from omegaconf import DictConfig, OmegaConf
from coolname import generate_slug


def custom_dir_resolver(agent_cfg: DictConfig, env_cfg: DictConfig):
    cool_name = generate_slug(2)
    timestamp = datetime.now().strftime("%Y%m%d")
    run_name = f"{timestamp}_{cool_name}_{env_cfg.name}"
    if "variant" in agent_cfg:
        run_name += f"_{agent_cfg.variant}"
    run_name += f"_{agent_cfg.model._target_.split('.')[-1]}"
    return run_name


def int_product_resolver(a: float, b: float) -> int:
    return int(a * b)


def floor_div_resolver(a: float, b: float) -> int:
    return a // b


def seed_sequence_resolver(n: int, entropy: int) -> str:
    """
    Generates a comma-separated string of n seeds using numpy.random.SeedSequence.
    This is compatible with Hydra's sweeper.
    """
    seed_sequence = np.random.SeedSequence(entropy)
    seeds = seed_sequence.generate_state(n)
    return ",".join(map(str, seeds))


resolvers = {
    "seed_sequence": seed_sequence_resolver,
    "int_product": int_product_resolver,
    "floor_div": floor_div_resolver,
    "custom_dir": custom_dir_resolver,
}
for resolver_name, resolver in resolvers.items():
    if not OmegaConf.has_resolver(resolver_name):
        OmegaConf.register_new_resolver(resolver_name, resolver)
