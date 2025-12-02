import os

from omegaconf import DictConfig


def set_torch_gpu(job_num: int, n_gpus: int = 1):
    """
    Set the CUDA_VISIBLE_DEVICES environment variable to a specific GPU.

    This is useful for running multiple jobs on a multi-GPU machine.

    Parameters
    ----------
    job_num : int
        The job number.
    n_gpus : int, optional
        The number of GPUs available, by default 1.
    """
    gpu_id = job_num % n_gpus
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)


def update_bootstrap_path(agent_params: dict, cfg: DictConfig) -> dict:
    if "continual" in cfg.agent.model._target_ and cfg.agent.get("bootstrap", None) is not None:
        agent_params["actor_path"] = os.path.join("pretrained_models", cfg.env.name, cfg.agent.bootstrap + ".pth")
        agent_params["critic_path"] = os.path.join("pretrained_models", cfg.env.name, "TD3" + ".pth")
        if cfg.agent.get("boot_with_shrink_factor", False):
            agent_params["actor_path"] = agent_params["actor_path"].removesuffix(".pth") + "_shrink.pth"
    elif cfg.agent.get("bootstrap", None) is not None:
        agent_params["policy_path"] = os.path.join("pretrained_models", cfg.env.name, cfg.agent.bootstrap + ".pth")
        if cfg.agent.get("boot_with_shrink_factor", False):
            agent_params["policy_path"] = agent_params["policy_path"].removesuffix(".pth") + "_shrink.pth"
    return agent_params
