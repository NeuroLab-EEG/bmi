"""
Perform cross-subject evaluation with binary classification.

References
----------
.. [1] https://moabb.neurotechx.com/docs/generated/moabb.evaluations.CrossSubjectEvaluation.html#moabb.evaluations.CrossSubjectEvaluation  # noqa: E501
.. [2] https://moabb.neurotechx.com/docs/auto_examples/advanced_examples/plot_select_electrodes_resample.html  # noqa: E501
"""

from os import getenv
from dotenv import load_dotenv
from moabb.utils import set_download_dir
from moabb.evaluations import CrossSubjectEvaluation
from codecarbon import EmissionsTracker
from src.evaluation.evaluate_parameters import parameters


# Load environment variables
load_dotenv()
data_path = getenv("DATA_PATH")

# Change download directory
set_download_dir(data_path)

for name, (pipeline, grid), paradigm, dataset, epochs, n_splits in parameters:
    # Inititalize compute cost monitoring
    tracker = EmissionsTracker(
        experiment_id=name,
        output_dir=data_path,
        save_to_file=True,
        log_level="critical",
        tracking_mode="process"
    )

    # Initialize evaluation
    evaluation = CrossSubjectEvaluation(
        paradigm=paradigm,
        datasets=[dataset],
        return_epochs=epochs,
        hdf5_path=data_path,
        save_model=True,
        n_splits=n_splits,
    )

    # Execute evaluation
    tracker.start()
    evaluation.process(pipeline, grid)
    tracker.stop()
