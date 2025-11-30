import os

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from src.common.managment import set_torch_gpu


# Training function to create pretrained networks for TD3, vanilla-TC-M2TD3, stacked-TC-M2TD3, oracle-TC-M2TD3
@hydra.main(config_path="configs", config_name="local_pretrain_config", version_base=None)
def main(cfg: DictConfig):
    """
    Main function for training the agent.

    Parameters
    ----------
    cfg : DictConfig
        The configuration object.
    """
    hydra_cfg = HydraConfig.get()
    job_id = hydra_cfg.job.get("num", None)
    if job_id is not None:
        set_torch_gpu(job_id, cfg.num_gpus, cfg.gpu_slot_size)
    # Imports in main to make multiprocessing easier, and after setting gpu
    from src.environments import create_env
    from src.agents.base_algorithm import BaseAlgorithm
    from src.common.evaluation import evaluate_policy

    output_dir = hydra_cfg.runtime.output_dir
    cool_name = os.path.basename(output_dir).split("_")[1]

    print(f"Results will be saved to {output_dir}")

    env = create_env(cfg)

    agent_params = {"seed": cfg.seed, "env": env, "tensorboard_log": output_dir}
    agent: BaseAlgorithm = hydra.utils.instantiate(cfg.agent.model, **agent_params, _convert_="all")

    setup_string = (
        f"Starting training of {cool_name},"
        + f"\nEnv: {cfg.env.name}\nAgent: {cfg.agent.model._target_.split('.')[-1]}"
    )
    if "variant" in cfg.agent:
        setup_string += f", variant: {cfg.agent.variant}"
    print(setup_string)
    agent.learn(total_timesteps=int(cfg.total_timesteps), progress_bar=True)

    print("Training finished. Saving model...")
    save_path = f"pretrained_models/{cfg.env.name}/"
    agent_name = ""
    if cfg.agent.get("variant", None) is not None:
        agent_name += f"_{cfg.agent.variant}"
    agent_name += f"_{cfg.agent.model._target_.split('.')[-1]}"
    agent_name = agent_name[1:]
    if cfg.env.get("shrink_factor", 0) > 0:
        agent_name += "_shrink"
    save_path += agent_name + ".pth"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    agent.save(save_path)

    print("Evaluation...")
    mean_reward, std_reward = evaluate_policy(agent, env, n_eval_episodes=10, seed=cfg.seed)
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    # Save evaluation results
    with open(f"{output_dir}/evaluation.txt", "w") as f:
        f.write(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")


if __name__ == "__main__":
    main()
