import io
import os
import socket

import blobfile as bf
import torch as th
import torch.distributed as dist

# Constants for multi-GPU environments
GPUS_PER_NODE = 8
SETUP_RETRY_COUNT = 3

def setup_dist(args):
    """
    Setup a distributed process group using the Gloo backend.
    """
    if dist.is_initialized():
        return
    
    # Set CUDA_VISIBLE_DEVICES to the selected GPU, or leave it for CPU usage
    if not args.multi_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_dev

    # Force the use of the "gloo" backend for distributed processing
    backend = "gloo"

    # Set the master address and port
    hostname = "localhost"
    os.environ["MASTER_ADDR"] = '127.0.0.1'
    os.environ["RANK"] = '0'  # Single process rank
    os.environ["WORLD_SIZE"] = '1'  # Single process, so world size is 1

    # Find and set a free port for communication
    port = _find_free_port()
    os.environ["MASTER_PORT"] = str(port)

    # Initialize the process group with the gloo backend
    dist.init_process_group(backend=backend, init_method="env://")


def dev():
    """
    Get the device to use for torch.distributed. Returns CPU when using gloo.
    """
    return th.device("cpu")  # Force CPU usage since we're using the gloo backend


def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file without redundant fetches across MPI ranks.
    """
    mpigetrank = 0  # Assuming single process (rank 0)
    if mpigetrank == 0:
        with bf.BlobFile(path, "rb") as f:
            data = f.read()
    else:
        data = None
    
    return th.load(io.BytesIO(data), **kwargs)


def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    This can be omitted if you're not using distributed training.
    """
    for p in params:
        with th.no_grad():
            dist.broadcast(p, 0)


def _find_free_port():
    """
    Find a free port on the system to use for distributed communication.
    """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()
