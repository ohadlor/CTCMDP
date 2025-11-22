import os

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from src.common.managment import set_torch_gpu


# TODO: toggle continuous learning
@hydra.main(config_path="configs", config_name="main_config", version_base=None)
def main(cfg: DictConfig):
    hydra_cfg = HydraConfig.get()
    job_id = hydra_cfg.job.get("num", None)
    if job_id is not None:
        set_torch_gpu(job_id, cfg.num_gpus, cfg.gpu_slot_size)
    # Imports in main to make multiprocessing easier, and after setting gpu
    from src.agents.td3 import TD3
    from src.environments import create_env
    from src.schedules import BaseActionSchedule
    from src.common.evaluation import evaluate_policy_hidden_state

    output_dir = hydra_cfg.runtime.output_dir
    cool_name = os.path.basename(output_dir).split("_")[1]

    print(f"Results will be saved to {output_dir}")

    env = create_env(cfg, output_dir)

    agent_params = {"seed": cfg.seed, "env": env, "tensorboard_log": output_dir}
    if "continual" in cfg.agent.name and cfg.agent.get("bootstrap", None) is not None:
        agent_params["actor_path"] = os.path.join("pretrained_models", cfg.env.name, cfg.agent.bootstrap)
        agent_params["critic_path"] = os.path.join("pretrained_models", cfg.env.name, "td3")
    elif cfg.agent.get("bootstrap", None) is not None:
        bootstrap_path = os.path.join("pretrained_models", cfg.env.name, cfg.agent.bootstrap)
        agent_params["actor_path"] = bootstrap_path

    agent: TD3 = hydra.utils.instantiate(cfg.agent.model, **agent_params, _convert_="all")

    hidden_action_schedule: BaseActionSchedule = hydra.utils.instantiate(
        cfg.schedule, action_space=env.hidden_action_space, state_space=env.state_space, rng=agent.rng
    )

    setup_string = (
        f"Starting evaluation of {cool_name},"
        + f"\nEnv: {cfg.env.name}\nAgent: {cfg.agent.model._target_.split('.')[-1]}"
    )
    if cfg.actor.bootstrap_from is not None:
        setup_string += f", bootstrapped from: {cfg.actor.bootstrap_from}"
    print(setup_string)

    print("Setup complete. Starting continual learning...")

    mean_reward, std_reward = evaluate_policy_hidden_state(
        model=agent,
        env=env,
        adversary_policy=hidden_action_schedule,
        iterations=cfg.iterations,
        seed=cfg.seed,
    )

    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    # Save evaluation results
    with open(f"{output_dir}/evaluation.txt", "w") as f:
        f.write(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")


if __name__ == "__main__":
    main()
