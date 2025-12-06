import os

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from src.common.managment import set_torch_gpu, update_bootstrap_path


@hydra.main(config_path="configs", config_name="local_test_config", version_base=None)
def main(cfg: DictConfig):
    """
    Main function for testing the agent.

    Parameters
    ----------
    cfg : DictConfig
        The configuration object.
    """
    hydra_cfg = HydraConfig.get()
    job_id = hydra_cfg.job.get("num", None)
    if job_id is not None:
        set_torch_gpu(job_id, cfg.num_gpus)
    # Imports in main to make multiprocessing easier, and after setting gpu
    from src.agents.td3 import TD3
    from src.environments import create_env
    from src.schedules import BaseActionSchedule
    from src.common.evaluation import evaluate_policy_hidden_state

    output_dir = hydra_cfg.runtime.output_dir
    cool_name = os.path.basename(output_dir).split("_")[1]

    print(f"Results will be saved to {output_dir}")

    env_name = cfg.env.id.split("-")[0]
    env = create_env(cfg)

    agent_params = {"seed": cfg.master_seed, "env": env, "tensorboard_log": output_dir}
    agent_params = update_bootstrap_path(agent_params, cfg)
    agent: TD3 = hydra.utils.instantiate(cfg.agent.model, **agent_params, _convert_="all")

    hidden_action_schedule: BaseActionSchedule = hydra.utils.instantiate(
        cfg.schedule,
        action_space=env.hidden_action_space,
        observation_space=env.hidden_observation_space,
    )

    setup_string = (
        f"Starting evaluation of {cool_name}," + f"\nEnv: {env_name}\nAgent: {cfg.agent.model._target_.split('.')[-1]}"
    )
    if cfg.agent.bootstrap is not None:
        setup_string += f", bootstrapped from: {cfg.agent.bootstrap}"
    print(setup_string)

    print("Setup complete. Starting continual learning...")

    mean_reward, std_reward = evaluate_policy_hidden_state(
        model=agent,
        env=env,
        total_timesteps=cfg.total_timesteps,
        adversary_policy=hidden_action_schedule,
        seeds=cfg.seeds,
    )

    print(f"Time average reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    # Save evaluation results
    with open(f"{output_dir}/evaluation.txt", "w") as f:
        f.write(f"Time average reward: {mean_reward:.5f} +/- {std_reward:.5f}")


if __name__ == "__main__":
    main()
