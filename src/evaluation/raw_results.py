"""
Save raw results from evaluations to disk as readable CSV.

References
----------
.. [1] https://github.com/NeuroTechX/moabb/blob/develop/moabb/analysis/results.py
"""

from os import path, getenv
from dotenv import load_dotenv
from moabb.analysis.results import Results
from moabb.evaluations import CrossSubjectEvaluation
from src.paradigm.paradigm import LogLossLeftRightImagery

# Load environment variables
load_dotenv()
data_path = getenv("DATA_PATH")

# Read raw results
df = Results(
    CrossSubjectEvaluation, LogLossLeftRightImagery, hdf5_path=data_path
).to_dataframe()

# Save raw results to disk
df.to_csv(path.join(data_path, "raw_results.csv"), index=False)
