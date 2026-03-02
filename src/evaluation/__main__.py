"""
References
----------
.. [1] https://github.com/jax-ml/jax/discussions/10674#discussioncomment-7214817
"""

import os

os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_ops=true"

import subprocess
from .evaluation import Evaluation


if __name__ == "__main__":
    for dataset in Evaluation.DATASETS.keys():
        for pipeline in Evaluation.PIPELINES.keys():
            subprocess.run(
                [
                    "python",
                    "-m",
                    "src.evaluation.run",
                    "--dataset",
                    dataset,
                    "--pipeline",
                    pipeline,
                ]
            )
