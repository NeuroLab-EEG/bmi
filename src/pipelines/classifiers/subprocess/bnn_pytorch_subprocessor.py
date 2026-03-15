from os import path
from .pytorch_subprocessor import PyTorchSubprocessor


class BNNPyTorchSubprocessor(PyTorchSubprocessor):
    def load_backbone(self):
        classifier = self.estimator.build_classifier()
        classifier.initialize()
        classifier.load_params(f_params=path.join(self.save_dir, "params.pt"))
        classifier.module_.to("cpu")
        return classifier.module_
