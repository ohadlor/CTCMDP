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
    env_name = cfg.env.id.split("-")[0]
    boot_step = f"_{cfg.agent.bootstrap_step}.pth"
    if cfg.agent.get("bootstrap", None) is not None:
        agent_params["actor_path"] = os.path.join("pretrained_models", env_name, cfg.agent.bootstrap + boot_step)
        boot_shrink_factor = cfg.agent.get("boot_with_shrink_factor", False)
        if boot_shrink_factor:
            agent_params["actor_path"] = (
                agent_params["actor_path"].removesuffix(boot_step)
                + f"_shrink-{str(boot_shrink_factor).replace('.', '')}"
                + boot_step
            )
        if "continual" in cfg.agent.model._target_:
            agent_params["critic_path"] = os.path.join("pretrained_models", env_name, "TD3" + boot_step)
    return agent_params
