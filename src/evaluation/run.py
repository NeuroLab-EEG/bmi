import argparse
from .evaluation import Evaluation


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--pipeline", required=True)
    args = parser.parse_args()
    evaluation = Evaluation(dataset=args.dataset, pipeline=args.pipeline)
    evaluation.run()
