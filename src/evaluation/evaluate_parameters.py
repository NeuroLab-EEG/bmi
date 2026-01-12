"""
ML pipelines evaluation parameters.
"""

from moabb.datasets import (
    PhysionetMI,
    Lee2019_MI,
    Cho2017,
    Schirrmeister2017,
    Shin2017A,
    BNCI2014_001,
)
from src.paradigm.paradigm import LogLossLeftRightImagery
from src.pipelines.raw_signal import csp_lda, csp_svm
from src.pipelines.riemannian import ts_lr, ts_svm
from src.pipelines.deep_learning import scnn, dcnn


physionetmi = [
    ("CSP+LDA+PhysionetMI", csp_lda(), LogLossLeftRightImagery(resample=160), PhysionetMI(), False, 10),
    ("CSP+SVM+PhysionetMI", csp_svm(), LogLossLeftRightImagery(resample=160), PhysionetMI(), False, 10),
    ("TS+LR+PhysionetMI", ts_lr(), LogLossLeftRightImagery(resample=160), PhysionetMI(), False, 10),
    ("TS+SVM+PhysionetMI", ts_svm(), LogLossLeftRightImagery(resample=160), PhysionetMI(), False, 10),
    ("SCNN+PhysionetMI", scnn(), LogLossLeftRightImagery(resample=160), PhysionetMI(), True, 10),
    ("DCNN+PhysionetMI", dcnn(), LogLossLeftRightImagery(resample=160), PhysionetMI(), True, 10),
]

lee2019_mi = [
    ("CSP+LDA+Lee2019_MI", csp_lda(), LogLossLeftRightImagery(resample=1000), Lee2019_MI(), False, 10),
    ("CSP+SVM+Lee2019_MI", csp_svm(), LogLossLeftRightImagery(resample=1000), Lee2019_MI(), False, 10),
    ("TS+LR+Lee2019_MI", ts_lr(), LogLossLeftRightImagery(resample=1000), Lee2019_MI(), False, 10),
    ("TS+SVM+Lee2019_MI", ts_svm(), LogLossLeftRightImagery(resample=1000), Lee2019_MI(), False, 10),
    ("SCNN+Lee2019_MI", scnn(), LogLossLeftRightImagery(resample=1000), Lee2019_MI(), True, 10),
    ("DCNN+Lee2019_MI", dcnn(), LogLossLeftRightImagery(resample=1000), Lee2019_MI(), True, 10),
]

cho2017 = [
    ("CSP+LDA+Cho2017", csp_lda(), LogLossLeftRightImagery(resample=512), Cho2017(), False, 10),
    ("CSP+SVM+Cho2017", csp_svm(), LogLossLeftRightImagery(resample=512), Cho2017(), False, 10),
    ("TS+LR+Cho2017", ts_lr(), LogLossLeftRightImagery(resample=512), Cho2017(), False, 10),
    ("TS+SVM+Cho2017", ts_svm(), LogLossLeftRightImagery(resample=512), Cho2017(), False, 10),
    ("SCNN+Cho2017", scnn(), LogLossLeftRightImagery(resample=512), Cho2017(), True, 10),
    ("DCNN+Cho2017", dcnn(), LogLossLeftRightImagery(resample=512), Cho2017(), True, 10),
]

schirrmeister2017 = [
    ("CSP+LDA+Schirrmeister2017", csp_lda(), LogLossLeftRightImagery(resample=500), Schirrmeister2017(), False, 5),
    ("CSP+SVM+Schirrmeister2017", csp_svm(), LogLossLeftRightImagery(resample=500), Schirrmeister2017(), False, 5),
    ("TS+LR+Schirrmeister2017", ts_lr(), LogLossLeftRightImagery(resample=500), Schirrmeister2017(), False, 5),
    ("TS+SVM+Schirrmeister2017", ts_svm(), LogLossLeftRightImagery(resample=500), Schirrmeister2017(), False, 5),
    ("SCNN+Schirrmeister2017", scnn(), LogLossLeftRightImagery(resample=500), Schirrmeister2017(), True, 5),
    ("DCNN+Schirrmeister2017", dcnn(), LogLossLeftRightImagery(resample=500), Schirrmeister2017(), True, 5),
]

shin2017a = [
    ("CSP+LDA+Shin2017A", csp_lda(), LogLossLeftRightImagery(resample=200), Shin2017A(), False, 5),
    ("CSP+SVM+Shin2017A", csp_svm(), LogLossLeftRightImagery(resample=200), Shin2017A(), False, 5),
    ("TS+LR+Shin2017A", ts_lr(), LogLossLeftRightImagery(resample=200), Shin2017A(), False, 5),
    ("TS+SVM+Shin2017A", ts_svm(), LogLossLeftRightImagery(resample=200), Shin2017A(), False, 5),
    ("SCNN+Shin2017A", scnn(), LogLossLeftRightImagery(resample=200), Shin2017A(), True, 5),
    ("DCNN+Shin2017A", dcnn(), LogLossLeftRightImagery(resample=200), Shin2017A(), True, 5),
]

bnci2014_001 = [
    ("CSP+LDA+BNCI2014_001", csp_lda(), LogLossLeftRightImagery(resample=250), BNCI2014_001(), False, 9),
    ("CSP+SVM+BNCI2014_001", csp_svm(), LogLossLeftRightImagery(resample=250), BNCI2014_001(), False, 9),
    ("TS+LR+BNCI2014_001", ts_lr(), LogLossLeftRightImagery(resample=250), BNCI2014_001(), False, 9),
    ("TS+SVM+BNCI2014_001", ts_svm(), LogLossLeftRightImagery(resample=250), BNCI2014_001(), False, 9),
    ("SCNN+BNCI2014_001", scnn(), LogLossLeftRightImagery(resample=250), BNCI2014_001(), True, 9),
    ("DCNN+BNCI2014_001", dcnn(), LogLossLeftRightImagery(resample=250), BNCI2014_001(), True, 9),
]

parameters = (
    physionetmi
    + lee2019_mi
    + cho2017
    + schirrmeister2017
    + shin2017a
    + bnci2014_001
)
