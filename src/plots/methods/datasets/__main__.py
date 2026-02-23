from .montage import Montage
from .bubble_plot import BubblePlot
from .psd import PSD
from .edf import EDF

if __name__ == "__main__":
    Montage().run()
    BubblePlot().run()
    PSD().run()
    EDF().run()
