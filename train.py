from datetime import datetime

import hydra
import tensorboard
from omegaconf import DictConfig

from coolname import generate_slug

from src.environments import create_env
from src.agents.base_algorithm import BaseAlgorithm
from src.common.evaluation import evaluate_policy


# Training function to create pretrained networks for TD3, vanilla-TC-M2TD3, stacked-TC-M2TD3, oracle-TC-M2TD3
@hydra.main(config_path="configs", config_name="pretrain_config", version_base=None)
def main(cfg: DictConfig):
    cool_name = generate_slug(2)
    timestamp = datetime.now().strftime("%Y%m%d")
    run_name = f"{timestamp}_{cool_name}_pretrain_{cfg.agent.model._target_.split('.')[-1]}_{cfg.env.env_id}"
    if "variant" in cfg.agent:
        run_name += f"_{cfg.agent.variant}"
    run_dir = f"runs/{run_name}"

    env, _ = create_env(cfg, run_dir)

    agent_params = {"seed": cfg.seed, "env": env, "tensorboard_log": run_dir}
    agent: BaseAlgorithm = hydra.utils.instantiate(cfg.agent.model, **agent_params, _convert_="all")

    print(f"Setup complete. Starting training of {cool_name}")
    agent.learn(total_timesteps=int(cfg.total_timesteps), progress_bar=True)

    print("Training finished. Saving model...")
    save_path = f"pretrained_models/{cfg.env.env_id}_{cfg.agent.model._target_.split('.')[-1]}"
    if "variant" in cfg.agent:
        save_path += f"_{cfg.agent.variant}"
    agent.save(save_path)

    print("Evaluation...")
    mean_reward, std_reward = evaluate_policy(env, agent, n_eval_episodes=10)
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")


if __name__ == "__main__":
    main()
