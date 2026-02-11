"""
References
----------
.. [1] https://docs.jax.dev/en/latest/gpu_memory_allocation.html
"""

import os

os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

from .evaluation import Evaluation

if __name__ == "__main__":
    Evaluation().run()
