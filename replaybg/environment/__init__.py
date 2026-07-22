from numba import njit
from numba.experimental import jitclass

class Environment:
    """
    A class that represents the hyperparameters to be used by ReplayBG.

    ...
    Attributes
    ----------
    ts : int
        An integer that specifies the integration time (in minutes).
    seed : int
        An integer that specifies the random seed. For reproducibility.
    plot_mode : bool
        A boolean that specifies whether to show the plot of the results or not.
    verbose : bool
        A boolean that specifies the verbosity of ReplayBG.

    Methods
    -------
    None
    """

    def __init__(self,
                 ts: int = 1,
                 seed: int = 42,
                 plot_mode: bool = True,
                 verbose: bool = True
                 ):
        """
        Constructs all the necessary attributes for the Environment object.

        Parameters
        ----------

        ts : int, optional, default : 1
            An integer that specifies the integration time (in minutes).
        seed : int, optional, default : 42
            An integer that specifies the random seed. For reproducibility.
        plot_mode : boolean, optional, default : True
            A boolean that specifies whether to show the plot of the results or not.
        verbose : boolean, optional, default : True
            A boolean that specifies the verbosity of ReplayBG.
        """

        # Set integration time
        self.ts = ts


        # Set the seed
        self.seed = seed

        # Set plot mode and verbosity
        self.plot_mode = plot_mode
        self.verbose = verbose

DEBUG = False

def identity(x=None, **kwargs):
    if callable(x):
        return x
    def wrapper(func_or_cls):
        return func_or_cls
    return wrapper

njit_ = identity if DEBUG else njit
jitclass_ = identity if DEBUG else jitclass