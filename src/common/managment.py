import os

import psutil
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


def set_affinity(job_id: int, cores_per_job: int):

    # Designed for specific server architecture
    n_nodes = psutil.cpu_count(logical=False)
    print(f"Found {n_nodes} nodes")
    if n_nodes != 2:
        return
    # 2. Define your specific 2-Node hardware map
    # Node 0: 0-63 and 128-191
    # Node 1: 64-127 and 192-255
    node0_pool = list(range(0, 64)) + list(range(128, 192))
    node1_pool = list(range(64, 128)) + list(range(192, 256))

    node_id = job_id % n_nodes
    target_pool = node0_pool if node_id == 0 else node1_pool

    jobs_on_this_node = job_id // n_nodes
    start_idx = jobs_on_this_node * cores_per_job
    end_idx = start_idx + cores_per_job

    cores = target_pool[start_idx:end_idx]
    p = psutil.Process(os.getpid())
    p.cpu_affinity(cores)

    # Outside of loop such that os.environ["CUDA_VISIBLE_DEVICES"] is set before importing torch
    # os.environ["OMP_NUM_THREADS"] = str(cores_per_job)
    # os.environ["MKL_NUM_THREADS"] = str(cores_per_job)
    # th.set_num_threads(cores_per_job)

    print(f"Job {job_id} -> Node {node_id} | Cores: {cores} | Threads: {cores_per_job}")


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
