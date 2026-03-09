from moabb.datasets import (
    PhysionetMI,
    Lee2019_MI,
    Cho2017,
    Schirrmeister2017,
    Shin2017A,
    BNCI2014_001,
    BNCI2014_004,
    Dreyer2023,
    Weibo2014,
    GrosseWentrup2009,
    Stieger2021,
)
from src.datasets import Liu2024


N_SPLITS = {
    BNCI2014_001: 9,
    Liu2024: 10,
    Stieger2021: 10,
    PhysionetMI: 10,
    Lee2019_MI: 10,
    Cho2017: 10,
    Schirrmeister2017: 5,
    Shin2017A: 5,
    BNCI2014_004: 9,
    Dreyer2023: 10,
    Weibo2014: 5,
    GrosseWentrup2009: 5,
}

CHANNELS = {
    BNCI2014_001: ["FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "CP3", "CP1", "CPz", "CP2", "CP4"],
    Liu2024: ["FC3", "FCz", "FC4", "C3", "Cz", "C4", "CP3", "CP4"],
    Stieger2021: ["FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6"],
    PhysionetMI: ["FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6"],
    Lee2019_MI: ["FC5", "FC3", "FC1", "FC2", "FC4", "FC6", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6"],
    Cho2017: ["FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6"],
    Schirrmeister2017: ["FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "FCC5h", "FCC3h", "FFC1h", "FCC2h", "FCC4h", "FCC6h", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "CCP5h", "CCP3h", "CCP1h", "CCP2h", "CCP4h", "CCP6h", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6"],
    Shin2017A: ["FCC5h", "FCC3h", "FCC4h", "FCC6h", "Cz", "CCP5h", "CCP3h", "CCP4h", "CCP6h"],
    BNCI2014_004: ["C3", "Cz", "C4"],
    Dreyer2023: ["FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6"],
    Weibo2014: ["FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6"],
    GrosseWentrup2009: ["FC5", "FC3", "FC1", "FC2", "FC4", "FC6", "FCC5h", "FCC3h", "FFC1h", "FCC2h", "FCC4h", "FCC6h", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "CCP5h", "CCP3h", "CCP1h", "CCP2h", "CCP4h", "CCP6h", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6"],
}
