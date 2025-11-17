from datetime import datetime

import hydra
import numpy as np
from omegaconf import DictConfig

from src.agents.continual_td3 import ContinualTD3
from src.environments import create_env
from src.schedules import HiddenActionSelector
from src.common.evaluation import evaluate_policy


# TODO: implment the following agent classes: continual_td3, vanilla_td3, tc-m2td3


@hydra.main(config_path="configs", config_name="main_config", version_base=None)
def main(cfg: DictConfig):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = (
        f"{timestamp}_{cfg.agent.variant}_{cfg.schedule._target_.split('.')[-1]}"
        + f"_{cfg.reward._target_.split('.')[-1]}"
    )
    run_dir = f"runs/{run_name}"
    rng = np.random.default_rng(cfg.seed)

    env, simulator = create_env(cfg, run_dir)

    # Define the hidden action selection policy/schedule
    hidden_policy: HiddenActionSelector = hydra.utils.instantiate(
        cfg.schedule, rng=rng, hidden_dim=env.action_space["hidden"].shape[0]
    )

    agent: ContinualTD3 = hydra.utils.instantiate(
        cfg.agent,
        env=env,
        policy="MlpPolicy",
        simulator_env=simulator,
        seed=cfg.seed,
        _init_setup_model=True,
    )

    # Start the agent with a pretrained policy
    if cfg.bootstrap_model_path:
        print(f"Bootstrapping agent from: {cfg.bootstrap_model_path}")
        agent.load(cfg.bootstrap_model_path)

    print("Setup complete. Starting training loop...")

    evaluate_policy(
        agent,
        env,
        adversary_policy=hidden_policy,
        is_continual=cfg.agent.is_continual,
        n_eval_episodes=cfg.n_eval_episodes,
        render=cfg.render,
    )


if __name__ == "__main__":
    main()
