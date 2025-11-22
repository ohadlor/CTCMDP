import os


def set_torch_gpu(job_num: int, n_gpus: int = 1, jobs_per_gpu: int = 1):
    gpu_id = (job_num // jobs_per_gpu) % n_gpus
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
