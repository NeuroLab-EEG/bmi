"""
References
----------
.. [1] https://docs.jax.dev/en/latest/gpu_memory_allocation.html
.. [2] https://github.com/jax-ml/jax/discussions/10674#discussioncomment-7214817
"""

import os

os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_ops=true"

from .evaluation import Evaluation

if __name__ == "__main__":
    Evaluation().run()
