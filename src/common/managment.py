import os


def set_torch_gpu(job_num: int, n_gpus: int = 1, jobs_per_gpu: int = 1):
    """
    Set the CUDA_VISIBLE_DEVICES environment variable to a specific GPU.

    This is useful for running multiple jobs on a multi-GPU machine.

    Parameters
    ----------
    job_num : int
        The job number.
    n_gpus : int, optional
        The number of GPUs available, by default 1.
    jobs_per_gpu : int, optional
        The number of jobs to run on each GPU, by default 1.
    """
    gpu_id = (job_num // jobs_per_gpu) % n_gpus
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
