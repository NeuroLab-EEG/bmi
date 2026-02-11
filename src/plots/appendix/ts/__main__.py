from .geodesic import Geodesic
from .ode import ODE
from .pipeline import Pipeline
from .tangent_space import TangentSpace

if __name__ == "__main__":
    Geodesic().run()
    ODE().run()
    Pipeline().run()
    TangentSpace().run()
