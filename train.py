from datetime import datetime
import os

import hydra
from hydra.core.hydra_config import HydraConfig
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


def product_resolver(a, b) -> float:
    return float(a) * float(b)


OmegaConf.register_new_resolver("product", product_resolver)

OmegaConf.register_new_resolver("train_dir", custom_dir_resolver)


def set_torch_gpu(job_num: int, n_gpus: int = 1, jobs_per_gpu: int = 1):
    gpu_id = (job_num // jobs_per_gpu) % n_gpus
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)


# Training function to create pretrained networks for TD3, vanilla-TC-M2TD3, stacked-TC-M2TD3, oracle-TC-M2TD3
@hydra.main(config_path="configs", config_name="test_config", version_base=None)
def main(cfg: DictConfig):
    if cfg.get("job", None) is not None:
        job_id = cfg.job.num
        set_torch_gpu(job_id, cfg.n_gpus, cfg.gpu_slot_size)
    # Imports in main to make multiprocessing easier, and after setting gpu
    from src.environments import create_env
    from src.agents.base_algorithm import BaseAlgorithm
    from src.common.evaluation import evaluate_policy

    output_dir = HydraConfig.get().runtime.output_dir
    cool_name = os.path.basename(output_dir).split("_")[1]

    print(f"Results will be saved to {output_dir}")

    env, _ = create_env(cfg, output_dir)

    agent_params = {"seed": cfg.seed, "env": env, "tensorboard_log": output_dir}
    agent: BaseAlgorithm = hydra.utils.instantiate(cfg.agent.model, **agent_params, _convert_="all")
    print(agent.device)

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
    if "variant" in cfg.agent:
        agent_name += f"_{cfg.agent.variant}"
    agent_name += f"_{cfg.agent.model._target_.split('.')[-1]}.pt"
    agent_name = agent_name[1:]
    save_path += agent_name
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
