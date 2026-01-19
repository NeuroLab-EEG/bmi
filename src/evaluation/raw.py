"""
Save raw results from evaluations to disk as readable CSV.

References
----------
.. [1] https://github.com/NeuroTechX/moabb/blob/develop/moabb/analysis/results.py
"""

from argparse import ArgumentParser
from os import path, getenv
from dotenv import load_dotenv
from moabb.analysis.results import Results
from moabb.evaluations import CrossSubjectEvaluation
from src.paradigm.paradigm import LogLossLeftRightImagery


class Raw:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        self.data_path = getenv("DATA_PATH")

        # Read raw results
        self.results = Results(
            CrossSubjectEvaluation, LogLossLeftRightImagery, hdf5_path=self.data_path
        ).to_dataframe()

    def csv(self):
        # Optionally filter raw results
        parser = ArgumentParser()
        parser.add_argument("--subject", type=str, default=None)
        parser.add_argument("--session", type=str, default=None)
        parser.add_argument("--dataset", type=str, default=None)
        parser.add_argument("--pipeline", type=str, default=None)
        args = parser.parse_args()
        if args.subject:
            self.results = self.results[self.results["subject"] == args.subject]
        if args.session:
            self.results = self.results[self.results["session"] == args.session]
        if args.dataset:
            self.results = self.results[self.results["dataset"] == args.dataset]
        if args.pipeline:
            self.results = self.results[self.results["pipeline"] == args.pipeline]

        # Save raw results to disk
        self.results.to_csv(path.join(self.data_path, "raw.csv"), index=False)


Raw().csv()
