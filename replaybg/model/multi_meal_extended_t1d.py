import numpy as np
from numba import float64, int16, int64
from numba.typed import Dict
from numba import types

from environment import jitclass_

theta0_type = types.DictType(types.unicode_type, float64)
theta_prev_type = types.DictType(types.unicode_type, float64)
x0_type = types.DictType(types.unicode_type, float64)
JITCLASS_SPEC = [
    ("_G0", float64),
    ("_X0", float64),
    ("_Qsto1_B_0", float64),
    ("_Qsto2_B_0", float64),
    ("_Qgut_B_0", float64),
    ("_Qsto1_L_0", float64),
    ("_Qsto2_L_0", float64),
    ("_Qgut_L_0", float64),
    ("_Qsto1_D_0", float64),
    ("_Qsto2_D_0", float64),
    ("_Qgut_D_0", float64),
    ("_Qsto1_S_0", float64),
    ("_Qsto2_S_0", float64),
    ("_Qgut_S_0", float64),
    ("_Qsto1_H_0", float64),
    ("_Qsto2_H_0", float64),
    ("_Qgut_H_0", float64),
    ("_Qsto1_B2_0", float64),
    ("_Qsto2_B2_0", float64),
    ("_Qgut_B2_0", float64),
    ("_Qsto1_L2_0", float64),
    ("_Qsto2_L2_0", float64),
    ("_Qgut_L2_0", float64),
    ("_Qsto1_S2_0", float64),
    ("_Qsto2_S2_0", float64),
    ("_Qgut_S2_0", float64),
    ("_Isc10", float64),
    ("_Isc20", float64),
    ("_Ip0", float64),
    ("_IG0", float64),
    ("G", float64[:]),
    ("X", float64[:]),
    ("Qsto1_B", float64[:]),
    ("Qsto2_B", float64[:]),
    ("Qgut_B", float64[:]),
    ("Qsto1_L", float64[:]),
    ("Qsto2_L", float64[:]),
    ("Qgut_L", float64[:]),
    ("Qsto1_D", float64[:]),
    ("Qsto2_D", float64[:]),
    ("Qgut_D", float64[:]),
    ("Qsto1_S", float64[:]),
    ("Qsto2_S", float64[:]),
    ("Qgut_S", float64[:]),
    ("Qsto1_H", float64[:]),
    ("Qsto2_H", float64[:]),
    ("Qgut_H", float64[:]),
    ("Qsto1_B2", float64[:]),
    ("Qsto2_B2", float64[:]),
    ("Qgut_B2", float64[:]),
    ("Qsto1_L2", float64[:]),
    ("Qsto2_L2", float64[:]),
    ("Qgut_L2", float64[:]),
    ("Qsto1_S2", float64[:]),
    ("Qsto2_S2", float64[:]),
    ("Qgut_S2", float64[:]),
    ("Isc1", float64[:]),
    ("Isc2", float64[:]),
    ("Ip", float64[:]),
    ("IG", float64[:]),
    ("SI_B", float64),
    ("SI_L", float64),
    ("SI_D", float64),
    ("SI_B2", float64),
    ("SG", float64),
    ("Gb", float64),
    ("p2", float64),
    ("r1", float64),
    ("r2", float64),
    ("tau", int16),
    ("ka2", float64),
    ("kd", float64),
    ("ke", float64),
    ("kabs_B", float64),
    ("kabs_L", float64),
    ("kabs_D", float64),
    ("kabs_S", float64),
    ("kabs_H", float64),
    ("kabs_B2", float64),
    ("kabs_L2", float64),
    ("kabs_S2", float64),
    ("beta_B", int16),
    ("beta_L", int16),
    ("beta_D", int16),
    ("beta_S", int16),
    ("beta_B2", int16),
    ("beta_L2", int16),
    ("beta_S2", int16),
    ("kempt", float64),
    ("f", float64),
    ("VG", float64),
    ("VI", float64),
    ("alpha", int16),
    ("Ipb", float64),
    ("u2ss", float64),
    ("theta0", theta0_type),
    ("theta_prev", theta_prev_type),
    ("x0", x0_type),
    ("tsteps", int16),
    ("t_start", int16),
    ("n_u", int16),
    ("u", float64[:,:]),
    ("previous_ra", float64[:]),
]


@jitclass_(JITCLASS_SPEC)
class MultiMealExtendedT1DModel:
    """An extended multi-meal physiological model of glucose-insulin dynamics for T1D.

    The model is a system of ODEs integrated with Backward Euler at a one-minute
    resolution. It extends :class:`~model.multi_meal_t1d.MultiMealT1DModel` with
    second-occurrence meal slots (``B2``, ``L2``, ``S2``) and a second breakfast
    insulin sensitivity ``SI_B2`` so that simulations spanning more than one day
    can be twinned in a single segment. The interstitial glucose is the
    observable output. The class is compiled with Numba's ``@jitclass``; all
    parameters are passed in through Numba typed dicts.

    ...
    Attributes
    ----------
    u2ss : float
        Steady-state basal insulin (the basal insulin rate at equilibrium).
    f : float
        Fraction of intestinal absorption that appears in plasma.
    VG : float
        Glucose distribution volume.
    VI : float
        Insulin distribution volume.
    alpha : int
        Rate governing the lag between plasma and interstitial glucose.
    SI_B, SI_L, SI_D, SI_B2 : float
        Insulin sensitivity for the breakfast, lunch, dinner/night and
        second-day breakfast windows.
    SG : float
        Glucose effectiveness.
    Gb : float
        Basal (target) glucose concentration.
    p2 : float
        Rate constant of the remote insulin (action) compartment.
    r1, r2 : float
        Parameters of the non-symmetric hypoglycaemia risk function.
    ka2, kd, ke : float
        Insulin pharmacokinetic rate constants.
    tau : int
        Integer delay (minutes) applied to the subcutaneous insulin input.
    kabs_B, kabs_L, kabs_D, kabs_S, kabs_H, kabs_B2, kabs_L2, kabs_S2 : float
        Per-meal gut-absorption rate constants (including second occurrences).
    kempt : float
        Gastric emptying rate constant shared across meal slots.
    beta_B, beta_L, beta_D, beta_S, beta_B2, beta_L2, beta_S2 : int
        Per-meal announcement delays (integer minutes).
    Ipb : float
        Basal plasma insulin derived from ``u2ss``.
    G, X, Isc1, Isc2, Ip, IG : numpy.ndarray
        State trajectories for plasma glucose, insulin action, the two
        subcutaneous insulin compartments, plasma insulin and interstitial
        glucose.
    Qsto1_*, Qsto2_*, Qgut_* : numpy.ndarray
        Per-meal gastric (1, 2) and gut compartment trajectories, one set per
        meal label (including second occurrences).
    theta0 : numba.typed.Dict
        The initial parameter dict the model was constructed with.
    theta_prev : numba.typed.Dict
        Parameters of the previous segment, used to scale carry-over initial
        conditions across segments.
    x0 : numba.typed.Dict
        Initial-condition dict (carry-over state from a previous segment).
    tsteps : int
        Number of integration steps.
    t_start : int
        Minute-of-day at which this segment starts (drives the time-of-day SI).
    n_u : int
        Number of input channels (13).
    u : numpy.ndarray
        Input history buffer used to retrieve delayed inputs during ``step``.
    previous_ra : numpy.ndarray
        Rate of appearance carried over from the previous segment.

    Methods
    -------
    reset(theta0):
        Resets parameters and re-initialises the state arrays.
    step(u, t):
        Advances the model by one minute.
    output(t):
        Returns the model output (interstitial glucose) at step ``t``.
    get_final_x0():
        Returns the end-of-segment state as a typed dict.
    get_theta():
        Returns the current parameters as a typed dict.
    """

    def __init__(self,
                 u2ss,
                 theta0=Dict.empty(key_type=types.unicode_type, value_type=float64),
                 x0=Dict.empty(key_type=types.unicode_type, value_type=float64),
                 tsteps=1440,
                 t_start=240,
                 theta_prev=Dict.empty(key_type=types.unicode_type, value_type=float64),
                 ):
        """Constructs the model and allocates the state arrays.

        Parameters
        ----------
        u2ss : float
            Steady-state basal insulin (the basal insulin rate at equilibrium).
        theta0 : numba.typed.Dict, optional
            Initial model parameters. Missing entries fall back to default values.
        x0 : numba.typed.Dict, optional
            Initial conditions (carry-over state). Missing entries fall back to
            the steady state.
        tsteps : int, optional, default : 1440
            Number of integration steps to allocate.
        t_start : int, optional, default : 240
            Minute-of-day at which this segment starts, driving the time-of-day
            insulin sensitivity selection.
        theta_prev : numba.typed.Dict, optional
            Parameters of the previous segment, used to scale the carry-over
            initial conditions. Empty for a cold start.
        """

        # Initial values of the model parameters (just a subset) and states
        self.theta0 = theta0
        self.x0 = x0
        self.tsteps = np.int16(tsteps)
        self.t_start = np.int16(t_start)

        # Number of inputs (8 meal channels + bolus + basal + t_hour + forcing_ip + forcing_ra = 13)
        self.n_u = np.int16(13)

        # Steady-state basal insulin (u2ss)
        self.u2ss = u2ss

        # Set previous "segment" parameters (needed to scale initial conditions if x0 is provided)
        self.theta_prev = theta_prev

        # Overwrite x0 of stomach before resetting the first time
        if len(self.theta_prev) > 0:
            self._override_stomach()

        # Launch reset() to initialise state arrays and model parameters
        self.reset(theta0)

    def reset(self,
              theta0=Dict.empty(key_type=types.unicode_type, value_type=float64)
              ):
        """Resets all model parameters and re-initialises the state arrays.

        Parameters are set to those provided in ``theta0``, or to default values
        when not provided.

        Parameters
        ----------
        theta0 : numba.typed.Dict, optional
            The model parameters to set. Missing entries fall back to default
            values.
        """

        # Reset parameters first
        self._reset_theta(theta0)

        # Reset state initial conditions
        self._reset_x0()

        # Reset inputs
        self._reset_u()


    def _reset_theta(self,
              theta0=Dict.empty(key_type=types.unicode_type, value_type=float64)
              ):
        """Sets the model parameters from ``theta0``, falling back to defaults.

        Parameters
        ----------
        theta0 : numba.typed.Dict, optional
            The model parameters to set. Missing entries fall back to default
            values.
        """
        # --- Structural / volume parameters ---
        self.f = theta0["f"] if "f" in theta0 else np.float64(0.9)
        self.VG = theta0["VG"] if "VG" in theta0 else np.float64(1.45)
        self.VI = theta0["VI"] if "VI" in theta0 else np.float64(0.135)
        # Alpha governs the lag between plasma and interstitial glucose
        self.alpha = np.int16(theta0["alpha"]) if "alpha" in theta0 else np.int16(7)

        # --- Insulin sensitivity (time-of-day dependent) ---
        self.SI_B = theta0["SI_B"] if "SI_B" in theta0 else np.float64(10.35e-4 / self.VG)
        self.SI_L = theta0["SI_L"] if "SI_L" in theta0 else np.float64(10.35e-4 / self.VG)
        self.SI_D = theta0["SI_D"] if "SI_D" in theta0 else np.float64(10.35e-4 / self.VG)
        self.SI_B2 = theta0["SI_B2"] if "SI_B2" in theta0 else np.float64(10.35e-4 / self.VG)

        # --- Glucose kinetics ---
        self.SG = theta0["SG"] if "SG" in theta0 else np.float64(2.5e-2)
        self.Gb = theta0["Gb"] if "Gb" in theta0 else np.float64(119.13)
        # p2: rate constant of the remote insulin compartment
        self.p2 = theta0["p2"] if "p2" in theta0 else np.float64(0.012)
        # Risk function parameters (non-symmetric hypoglycaemia penalty)
        self.r1 = theta0["r1"] if "r1" in theta0 else np.float64(1.4407)
        self.r2 = theta0["r2"] if "r2" in theta0 else np.float64(0.8124)

        # --- Insulin pharmacokinetics ---
        self.ka2 = theta0["ka2"] if "ka2" in theta0 else np.float64(0.014)
        self.kd = theta0["kd"] if "kd" in theta0 else np.float64(0.026)
        self.ke = theta0["ke"] if "ke" in theta0 else np.float64(0.127)
        # tau: integer delay (minutes) applied to the subcutaneous insulin input
        self.tau = np.int16(theta0["tau"]) if "tau" in theta0 else np.int16(8)

        # --- Meal gut-absorption rates ---
        self.kabs_B = theta0["kabs_B"] if "kabs_B" in theta0 else np.float64(0.012)
        self.kabs_L = theta0["kabs_L"] if "kabs_L" in theta0 else np.float64(0.012)
        self.kabs_D = theta0["kabs_D"] if "kabs_D" in theta0 else np.float64(0.012)
        self.kabs_S = theta0["kabs_S"] if "kabs_S" in theta0 else np.float64(0.012)
        self.kabs_H = theta0["kabs_H"] if "kabs_H" in theta0 else np.float64(0.012)
        self.kabs_B2 = theta0["kabs_B2"] if "kabs_B2" in theta0 else np.float64(0.012)
        self.kabs_L2 = theta0["kabs_L2"] if "kabs_L2" in theta0 else np.float64(0.012)
        self.kabs_S2 = theta0["kabs_S2"] if "kabs_S2" in theta0 else np.float64(0.012)
        # Shared gastric emptying rate constant across all meal slots
        self.kempt = theta0["kempt"] if "kempt" in theta0 else np.float64(0.18)

        # --- Meal announcement delays (integer minutes) ---
        self.beta_B = np.int16(theta0["beta_B"]) if "beta_B" in theta0 else np.int16(0)
        self.beta_L = np.int16(theta0["beta_L"]) if "beta_L" in theta0 else np.int16(0)
        self.beta_D = np.int16(theta0["beta_D"]) if "beta_D" in theta0 else np.int16(0)
        self.beta_S = np.int16(theta0["beta_S"]) if "beta_S" in theta0 else np.int16(0)
        self.beta_B2 = np.int16(theta0["beta_B2"]) if "beta_B2" in theta0 else np.int16(0)
        self.beta_L2 = np.int16(theta0["beta_L2"]) if "beta_L2" in theta0 else np.int16(0)
        self.beta_S2 = np.int16(theta0["beta_S2"]) if "beta_S2" in theta0 else np.int16(0)

    def _reset_x0(self):
        """Initialises the state arrays from ``x0``, falling back to the steady state.

        Carry-over insulin and insulin-action initial conditions are rescaled by
        the ratio between the current and previous segment parameters when
        ``theta_prev`` is provided.
        """

        # --- Initial conditions (fall back to steady state if not provided) ---

        # Glucose compartments
        self._G0 = self.x0["G0"] if "G0" in self.x0 else np.float64(self.Gb)
        self.G = np.empty((self.tsteps,), dtype=np.float64)
        self.G[0] = self._G0

        self._IG0 = self.x0["IG0"] if "IG0" in self.x0 else np.float64(self.Gb)
        self.IG = np.empty((self.tsteps,), dtype=np.float64)
        self.IG[0] = self._IG0

        # Gut compartments start empty (no meal in progress at t=0)
        self._Qsto1_B_0 = self.x0["Qsto1_B_0"] if "Qsto1_B_0" in self.x0 else np.float64(0)
        self.Qsto1_B = np.empty((self.tsteps,), dtype=np.float64)
        self.Qsto1_B[0] = self._Qsto1_B_0
        self._Qsto2_B_0 = self.x0["Qsto2_B_0"] if "Qsto2_B_0" in self.x0 else np.float64(0)
        self.Qsto2_B = np.empty((self.tsteps,), dtype=np.float64)
        self.Qsto2_B[0] = self._Qsto2_B_0
        self._Qgut_B_0 = self.x0["Qgut_B_0"] if "Qgut_B_0" in self.x0 else np.float64(0)
        self.Qgut_B = np.empty((self.tsteps,), dtype=np.float64)
        self.Qgut_B[0] = self._Qgut_B_0

        self._Qsto1_L_0 = self.x0["Qsto1_L_0"] if "Qsto1_L_0" in self.x0 else np.float64(0)
        self.Qsto1_L = np.empty((self.tsteps,), dtype=np.float64)
        self.Qsto1_L[0] = self._Qsto1_L_0
        self._Qsto2_L_0 = self.x0["Qsto2_L_0"] if "Qsto2_L_0" in self.x0 else np.float64(0)
        self.Qsto2_L = np.empty((self.tsteps,), dtype=np.float64)
        self.Qsto2_L[0] = self._Qsto2_L_0
        self._Qgut_L_0 = self.x0["Qgut_L_0"] if "Qgut_L_0" in self.x0 else np.float64(0)
        self.Qgut_L = np.empty((self.tsteps,), dtype=np.float64)
        self.Qgut_L[0] = self._Qgut_L_0

        self._Qsto1_D_0 = self.x0["Qsto1_D_0"] if "Qsto1_D_0" in self.x0 else np.float64(0)
        self.Qsto1_D = np.empty((self.tsteps,), dtype=np.float64)
        self.Qsto1_D[0] = self._Qsto1_D_0
        self._Qsto2_D_0 = self.x0["Qsto2_D_0"] if "Qsto2_D_0" in self.x0 else np.float64(0)
        self.Qsto2_D = np.empty((self.tsteps,), dtype=np.float64)
        self.Qsto2_D[0] = self._Qsto2_D_0
        self._Qgut_D_0 = self.x0["Qgut_D_0"] if "Qgut_D_0" in self.x0 else np.float64(0)
        self.Qgut_D = np.empty((self.tsteps,), dtype=np.float64)
        self.Qgut_D[0] = self._Qgut_D_0

        self._Qsto1_S_0 = self.x0["Qsto1_S_0"] if "Qsto1_S_0" in self.x0 else np.float64(0)
        self.Qsto1_S = np.empty((self.tsteps,), dtype=np.float64)
        self.Qsto1_S[0] = self._Qsto1_S_0
        self._Qsto2_S_0 = self.x0["Qsto2_S_0"] if "Qsto2_S_0" in self.x0 else np.float64(0)
        self.Qsto2_S = np.empty((self.tsteps,), dtype=np.float64)
        self.Qsto2_S[0] = self._Qsto2_S_0
        self._Qgut_S_0 = self.x0["Qgut_S_0"] if "Qgut_S_0" in self.x0 else np.float64(0)
        self.Qgut_S = np.empty((self.tsteps,), dtype=np.float64)
        self.Qgut_S[0] = self._Qgut_S_0

        self._Qsto1_H_0 = self.x0["Qsto1_H_0"] if "Qsto1_H_0" in self.x0 else np.float64(0)
        self.Qsto1_H = np.empty((self.tsteps,), dtype=np.float64)
        self.Qsto1_H[0] = self._Qsto1_H_0
        self._Qsto2_H_0 = self.x0["Qsto2_H_0"] if "Qsto2_H_0" in self.x0 else np.float64(0)
        self.Qsto2_H = np.empty((self.tsteps,), dtype=np.float64)
        self.Qsto2_H[0] = self._Qsto2_H_0
        self._Qgut_H_0 = self.x0["Qgut_H_0"] if "Qgut_H_0" in self.x0 else np.float64(0)
        self.Qgut_H = np.empty((self.tsteps,), dtype=np.float64)
        self.Qgut_H[0] = self._Qgut_H_0

        self._Qsto1_B2_0 = self.x0["Qsto1_B2_0"] if "Qsto1_B2_0" in self.x0 else np.float64(0)
        self.Qsto1_B2 = np.empty((self.tsteps,), dtype=np.float64)
        self.Qsto1_B2[0] = self._Qsto1_B2_0
        self._Qsto2_B2_0 = self.x0["Qsto2_B2_0"] if "Qsto2_B2_0" in self.x0 else np.float64(0)
        self.Qsto2_B2 = np.empty((self.tsteps,), dtype=np.float64)
        self.Qsto2_B2[0] = self._Qsto2_B2_0
        self._Qgut_B2_0 = self.x0["Qgut_B2_0"] if "Qgut_B2_0" in self.x0 else np.float64(0)
        self.Qgut_B2 = np.empty((self.tsteps,), dtype=np.float64)
        self.Qgut_B2[0] = self._Qgut_B2_0

        self._Qsto1_L2_0 = self.x0["Qsto1_L2_0"] if "Qsto1_L2_0" in self.x0 else np.float64(0)
        self.Qsto1_L2 = np.empty((self.tsteps,), dtype=np.float64)
        self.Qsto1_L2[0] = self._Qsto1_L2_0
        self._Qsto2_L2_0 = self.x0["Qsto2_L2_0"] if "Qsto2_L2_0" in self.x0 else np.float64(0)
        self.Qsto2_L2 = np.empty((self.tsteps,), dtype=np.float64)
        self.Qsto2_L2[0] = self._Qsto2_L2_0
        self._Qgut_L2_0 = self.x0["Qgut_L2_0"] if "Qgut_L2_0" in self.x0 else np.float64(0)
        self.Qgut_L2 = np.empty((self.tsteps,), dtype=np.float64)
        self.Qgut_L2[0] = self._Qgut_L2_0

        self._Qsto1_S2_0 = self.x0["Qsto1_S2_0"] if "Qsto1_S2_0" in self.x0 else np.float64(0)
        self.Qsto1_S2 = np.empty((self.tsteps,), dtype=np.float64)
        self.Qsto1_S2[0] = self._Qsto1_S2_0
        self._Qsto2_S2_0 = self.x0["Qsto2_S2_0"] if "Qsto2_S2_0" in self.x0 else np.float64(0)
        self.Qsto2_S2 = np.empty((self.tsteps,), dtype=np.float64)
        self.Qsto2_S2[0] = self._Qsto2_S2_0
        self._Qgut_S2_0 = self.x0["Qgut_S2_0"] if "Qgut_S2_0" in self.x0 else np.float64(0)
        self.Qgut_S2 = np.empty((self.tsteps,), dtype=np.float64)
        self.Qgut_S2[0] = self._Qgut_S2_0

        # Insulin action

        # X = (SI/VI)*(Ip - Ipb): when SI or VI change across segments, X0 must be
        # rescaled by the ratio of (SI_first/VI) for the current segment to
        # (SI_last/VI) for the previous segment, where SI_last is the sensitivity
        # active during the last minute of the previous segment (hour t_start-1).
        h_first = int(self.t_start // 60) % 24
        if h_first < 4 or h_first >= 17:
            SI_first = self.SI_D
        elif h_first < 11:
            SI_first = self.SI_B
        else:
            SI_first = self.SI_L
        if "X0" in self.x0 and len(self.theta_prev) > 0:
            self._X0 = (SI_first / self.VI) / (self.theta_prev["SI_last"] / self.theta_prev["VI"]) * self.x0["X0"]
        else:
            self._X0 = np.float64(0)
        self.X = np.empty((self.tsteps,), dtype=np.float64)
        self.X[0] = self._X0

        # Subcutaneous insulin absorption

        # Insulin compartments start at the basal steady state derived from u2ss.
        # When x0 carries values from the previous day, they are scaled by the ratio of
        # the current trial's ki1/ki2 to the previous day's ki1_prev/ki2_prev, mirroring
        # the py_replay_bg approach. When _kd_prev == kd (cold start default) the ratio
        # is 1 and behaviour is unchanged.
        ki1 = self.u2ss / self.kd
        ki2 = self.kd / self.ka2 * ki1
        self.Ipb = self.ka2 / self.ke * ki2  # basal plasma insulin

        if len(self.theta_prev) > 0:
            ki1_prev = self.u2ss / self.theta_prev["kd"]
            ki2_prev = self.theta_prev["kd"] / self.theta_prev["ka2"] * ki1_prev

        if "Isc10" in self.x0 and len(self.theta_prev) > 0:
            self._Isc10 = ki1 / ki1_prev * self.x0["Isc10"]
        else:
            self._Isc10 = ki1
        self.Isc1 = np.empty((self.tsteps,), dtype=np.float64)
        self.Isc1[0] = self._Isc10

        if "Isc20" in self.x0 and len(self.theta_prev) > 0:
            self._Isc20 = ki2 / ki2_prev * self.x0["Isc20"]
        else:
            self._Isc20 = ki2
            
        self.Isc2 = np.empty((self.tsteps,), dtype=np.float64)
        self.Isc2[0] = self._Isc20

        # Ipb = u2ss/ke regardless of kd/ka2 — no scaling needed
        self._Ip0 = self.x0["Ip0"] if "Ip0" in self.x0 else np.float64(self.Ipb)
        self.Ip = np.empty((self.tsteps,), dtype=np.float64)
        self.Ip[0] = self._Ip0

    def _reset_u(self):
        """Allocates the input history buffer and seeds the basal channel."""
        # Input buffer used by step() to retrieve delayed input values
        self.u = np.zeros((self.n_u, self.tsteps), dtype=np.float64)
        self.u[8, 0] = self.u2ss  # slot 8 = bolus in data convention, but set to u2ss
        self.u[9, 0] = 0.0  # slot 9 = basal in data convention, but set to 0
        if len(self.theta_prev) == 0:
            self.previous_ra = np.zeros(self.tsteps)


    def step(self, u: float64[:], t: int64):
        """Advances the model by one minute using Backward Euler integration.

        Parameters
        ----------
        u : numpy.ndarray
            The input vector for this step (13 channels: the eight meal slots
            B/L/D/S/H/B2/L2/S2, bolus, basal, hour-of-day, forcing_ip and
            forcing_ra).
        t : int
            The integration step index (minute) to compute.
        """
        # Store current inputs in the history buffer so delayed values can be
        # retrieved at this or future timesteps via index arithmetic.
        self.u[0, t] = u[0]
        # Apply meal announcement delay beta_B: use the carb input from beta_B
        # minutes ago. If we are still within the first beta_B minutes of the
        # simulation (t < beta_B), treat the delayed input as zero.
        u_m_b = self.u[0, t - self.beta_B] if (t - self.beta_B) >= 0 else 0
        self.u[1, t] = u[1]
        u_m_l = self.u[1, t - self.beta_L] if (t - self.beta_L) >= 0 else 0
        self.u[2, t] = u[2]
        u_m_d = self.u[2, t - self.beta_D] if (t - self.beta_D) >= 0 else 0
        self.u[3, t] = u[3]
        u_m_s = self.u[3, t - self.beta_S] if (t - self.beta_S) >= 0 else 0
        self.u[4, t] = u[4]
        # Hypo-treatment has no announcement delay
        u_m_h = self.u[4, t]
        self.u[5, t] = u[5]
        u_m_b2 = self.u[5, t - self.beta_B2] if (t - self.beta_B2) >= 0 else 0
        self.u[6, t] = u[6]
        u_m_l2 = self.u[6, t - self.beta_L2] if (t - self.beta_L2) >= 0 else 0
        self.u[7, t] = u[7]
        u_m_s2 = self.u[7, t - self.beta_S2] if (t - self.beta_S2) >= 0 else 0
        self.u[8, t] = u[8]
        self.u[9, t] = u[9]
        # Apply insulin absorption delay tau: use the insulin input from tau
        # minutes ago. Before tau minutes have elapsed, fall back to u2ss so
        # the plasma insulin chain starts at its basal steady state.
        u_i = self.u[8, t - self.tau] + self.u[9, t - self.tau] if (t - self.tau) >= 0 else self.u2ss
        self.u[10, t] = u[10]
        u_h = self.u[10, t]
        self.u[11, t] = u[11]
        u_fip = self.u[11, t] # forcing ip
        self.u[12, t] = u[12]
        u_fra = self.u[12, t] # forcing ra

        # Select insulin sensitivity based on hour of day:
        # breakfast window 04:00–10:59, lunch 11:00–16:59, dinner/night otherwise
        if u_h < 4 or u_h >= 17:
            SI = self.SI_D
        elif 4 <= u_h < 11:
            if (t + self.t_start) < 1440:
                SI = self.SI_B
            else:
                SI = self.SI_B2
        else:
            SI = self.SI_L

        # Non-symmetric risk function evaluated at the previous glucose value
        # (semi-implicit treatment of the nonlinear term to avoid a nonlinear
        # solve). Risk is zero above Gb, increasing below it, and constant
        # below 60 mg/dL.
        g_prev = self.G[t-1]
        logGb = np.log(self.Gb)
        log60 = np.log(60.0)
        if (g_prev < self.Gb) and (g_prev >= 60.0):
            lg = np.log(g_prev)
            diff = lg ** self.r2 - logGb ** self.r2
            risk = 1.0 + 10 * self.r1 * diff * diff
        elif g_prev < 60.0:
            diff = log60 ** self.r2 - logGb ** self.r2  # constant below 60 mg/dL
            risk = 1.0 + 10 * self.r1 * diff * diff
        else:
            risk = 1.0

        # Pre-compute implicit denominators.
        # For dQ/dt = -k*Q + input the BE update is Q[t] = (Q[t-1] + input) / (1 + k).
        k1 = 1.0 / (1.0 + self.kempt)   # Qsto1: -kempt * Qsto1
        k2 = 1.0 / (1.0 + self.kempt)   # Qsto2: -kempt * Qsto2
        kd_fac = 1.0 / (1.0 + self.kd)  # Isc1:  -kd * Isc1

        # --- Gut absorption chains (Backward Euler, Gauss-Seidel order) ---
        # Each meal slot: solid stomach (Qsto1) → liquid stomach (Qsto2) → intestine (Qgut).
        # Each compartment uses the current-step value of its upstream input,
        # which is valid because the system is a strict cascade (no feedback).
        self.Qsto1_B[t] = (self.Qsto1_B[t-1] + u_m_b) * k1
        self.Qsto2_B[t] = (self.Qsto2_B[t-1] + self.kempt * self.Qsto1_B[t]) * k2
        self.Qgut_B[t] = (self.Qgut_B[t-1] + self.kempt * self.Qsto2_B[t]) / (1 + self.kabs_B)

        self.Qsto1_L[t] = (self.Qsto1_L[t-1] + u_m_l) * k1
        self.Qsto2_L[t] = (self.Qsto2_L[t-1] + self.kempt * self.Qsto1_L[t]) * k2
        self.Qgut_L[t] = (self.Qgut_L[t-1] + self.kempt * self.Qsto2_L[t]) / (1 + self.kabs_L)

        self.Qsto1_D[t] = (self.Qsto1_D[t-1] + u_m_d) * k1
        self.Qsto2_D[t] = (self.Qsto2_D[t-1] + self.kempt * self.Qsto1_D[t]) * k2
        self.Qgut_D[t] = (self.Qgut_D[t-1] + self.kempt * self.Qsto2_D[t]) / (1 + self.kabs_D)

        self.Qsto1_S[t] = (self.Qsto1_S[t-1] + u_m_s) * k1
        self.Qsto2_S[t] = (self.Qsto2_S[t-1] + self.kempt * self.Qsto1_S[t]) * k2
        self.Qgut_S[t] = (self.Qgut_S[t-1] + self.kempt * self.Qsto2_S[t]) / (1 + self.kabs_S)

        self.Qsto1_H[t] = (self.Qsto1_H[t-1] + u_m_h) * k1
        self.Qsto2_H[t] = (self.Qsto2_H[t-1] + self.kempt * self.Qsto1_H[t]) * k2
        self.Qgut_H[t] = (self.Qgut_H[t-1] + self.kempt * self.Qsto2_H[t]) / (1 + self.kabs_H)

        self.Qsto1_B2[t] = (self.Qsto1_B2[t - 1] + u_m_b2) * k1
        self.Qsto2_B2[t] = (self.Qsto2_B2[t - 1] + self.kempt * self.Qsto1_B2[t]) * k2
        self.Qgut_B2[t] = (self.Qgut_B2[t - 1] + self.kempt * self.Qsto2_B2[t]) / (1 + self.kabs_B2)

        self.Qsto1_L2[t] = (self.Qsto1_L2[t - 1] + u_m_l2) * k1
        self.Qsto2_L2[t] = (self.Qsto2_L2[t - 1] + self.kempt * self.Qsto1_L2[t]) * k2
        self.Qgut_L2[t] = (self.Qgut_L2[t - 1] + self.kempt * self.Qsto2_L2[t]) / (1 + self.kabs_L2)

        self.Qsto1_S2[t] = (self.Qsto1_S2[t - 1] + u_m_s2) * k1
        self.Qsto2_S2[t] = (self.Qsto2_S2[t - 1] + self.kempt * self.Qsto1_S2[t]) * k2
        self.Qgut_S2[t] = (self.Qgut_S2[t - 1] + self.kempt * self.Qsto2_S2[t]) / (1 + self.kabs_S2)

        # --- Insulin pharmacokinetic chain (Backward Euler, Gauss-Seidel order) ---
        # Subcutaneous compartment 1 → compartment 2 → plasma
        self.Isc1[t] = (self.Isc1[t-1] + u_i) * kd_fac
        self.Isc2[t] = (self.Isc2[t-1] + self.kd * self.Isc1[t]) / (1 + self.ka2)
        self.Ip[t] = (self.Ip[t-1] + self.ka2 * self.Isc2[t] + u_fip) / (1 + self.ke)

        # --- Insulin action and glucose (Backward Euler) ---
        # X depends on Ip[t] (just computed above)
        self.X[t] = (self.X[t-1] + self.p2 * (SI / self.VI) * (self.Ip[t] - self.Ipb)) / (1 + self.p2)
        # G depends on X[t] and all Qgut[t] (just computed). The risk coefficient
        # is frozen at G[t-1] (semi-implicit) to keep the update linear in G[t].
        self.G[t] = (self.G[t-1] + self.SG * self.Gb + self.f * (
                self.kabs_B * self.Qgut_B[t] + self.kabs_L * self.Qgut_L[t] + self.kabs_D * self.Qgut_D[t] +
                self.kabs_S * self.Qgut_S[t] + self.kabs_H * self.Qgut_H[t] +
                self.kabs_B2 * self.Qgut_B2[t] + self.kabs_L2 * self.Qgut_L2[t] + self.kabs_S2 * self.Qgut_S2[t]) / self.VG + self.previous_ra[t-1] / self.VG + u_fra / self.VG) / (1 + self.SG + risk * self.X[t])
        # Interstitial glucose: first-order low-pass filter on plasma glucose
        self.IG[t] = (self.alpha * self.IG[t-1] + self.G[t]) / (1 + self.alpha)

    def output(self, t: float64):
        """Returns the model output at time step ``t``.

        The observable output is the interstitial glucose concentration, which
        corresponds to the signal measured by a continuous glucose monitor.

        Parameters
        ----------
        t : int
            Time step index (integer minute).

        Returns
        -------
        float
            The interstitial glucose concentration at time ``t`` (mg/dL).
        """
        return self.IG[t]

    def get_final_x0(self):
        """Returns the final state of the model as a Numba typed dict.

        Reads the last time-step of every state array and packs the values
        into a typed dict suitable for passing as ``x0`` to a subsequent
        segment's ``reset()`` call.

        Returns
        -------
        numba.typed.Dict
            A typed dict (unicode -> float64) with keys matching the ``x0``
            contract expected by ``reset()``.
        """
        final_x0 = Dict.empty(key_type=types.unicode_type, value_type=float64)
        final_x0["G0"]        = self.G[-1]
        final_x0["X0"]        = self.X[-1]
        final_x0["IG0"]       = self.IG[-1]
        final_x0["Isc10"]     = self.Isc1[-1]
        final_x0["Isc20"]     = self.Isc2[-1]
        final_x0["Ip0"]       = self.Ip[-1]
        final_x0["Qsto1_B_0"] = self.Qsto1_B[-1]
        final_x0["Qsto2_B_0"] = self.Qsto2_B[-1]
        final_x0["Qgut_B_0"]  = self.Qgut_B[-1]
        final_x0["Qsto1_L_0"] = self.Qsto1_L[-1]
        final_x0["Qsto2_L_0"] = self.Qsto2_L[-1]
        final_x0["Qgut_L_0"]  = self.Qgut_L[-1]
        final_x0["Qsto1_D_0"] = self.Qsto1_D[-1]
        final_x0["Qsto2_D_0"] = self.Qsto2_D[-1]
        final_x0["Qgut_D_0"]  = self.Qgut_D[-1]
        final_x0["Qsto1_S_0"] = self.Qsto1_S[-1]
        final_x0["Qsto2_S_0"] = self.Qsto2_S[-1]
        final_x0["Qgut_S_0"]  = self.Qgut_S[-1]
        final_x0["Qsto1_H_0"] = self.Qsto1_H[-1]
        final_x0["Qsto2_H_0"] = self.Qsto2_H[-1]
        final_x0["Qgut_H_0"]  = self.Qgut_H[-1]
        final_x0["Qsto1_B2_0"] = self.Qsto1_B2[-1]
        final_x0["Qsto2_B2_0"] = self.Qsto2_B2[-1]
        final_x0["Qgut_B2_0"] = self.Qgut_B2[-1]
        final_x0["Qsto1_L2_0"] = self.Qsto1_L2[-1]
        final_x0["Qsto2_L2_0"] = self.Qsto2_L2[-1]
        final_x0["Qgut_L2_0"] = self.Qgut_L2[-1]
        final_x0["Qsto1_S2_0"] = self.Qsto1_S2[-1]
        final_x0["Qsto2_S2_0"] = self.Qsto2_S2[-1]
        final_x0["Qgut_S2_0"] = self.Qgut_S2[-1]
        return final_x0

    def get_theta(self):
        """Returns the current model parameters as a Numba typed dict.

        Packs all parameters set by :meth:`_reset_theta` into a typed dict
        suitable for passing as ``theta0`` to a subsequent ``reset()`` call.

        Returns
        -------
        numba.typed.Dict
            A typed dict (unicode -> float64) with keys matching the
            ``unknown_parameters_prior`` naming convention.
        """
        theta = Dict.empty(key_type=types.unicode_type, value_type=float64)
        theta["f"]      = self.f
        theta["VG"]     = self.VG
        theta["VI"]     = self.VI
        theta["alpha"]  = np.float64(self.alpha)
        theta["SI_B"]   = self.SI_B
        theta["SI_L"]   = self.SI_L
        theta["SI_D"]   = self.SI_D
        theta["SI_B2"] = self.SI_B2
        theta["SG"]     = self.SG
        theta["Gb"]     = self.Gb
        theta["p2"]     = self.p2
        theta["r1"]     = self.r1
        theta["r2"]     = self.r2
        theta["ka2"]    = self.ka2
        theta["kd"]     = self.kd
        theta["ke"]     = self.ke
        theta["tau"]    = np.float64(self.tau)
        theta["kabs_B"] = self.kabs_B
        theta["kabs_L"] = self.kabs_L
        theta["kabs_D"] = self.kabs_D
        theta["kabs_S"] = self.kabs_S
        theta["kabs_H"] = self.kabs_H
        theta["kabs_B2"] = self.kabs_B2
        theta["kabs_L2"] = self.kabs_L2
        theta["kabs_S2"] = self.kabs_S2
        theta["kempt"]  = self.kempt
        theta["beta_B"] = np.float64(self.beta_B)
        theta["beta_L"] = np.float64(self.beta_L)
        theta["beta_D"] = np.float64(self.beta_D)
        theta["beta_S"] = np.float64(self.beta_S)
        theta["beta_B2"] = np.float64(self.beta_B2)
        theta["beta_L2"] = np.float64(self.beta_L2)
        theta["beta_S2"] = np.float64(self.beta_S2)
        theta["SI_last"] = self.SI_D # IMPORTANT: It is always the last SI value since it is extended mode (user is supposed to cut before 4 AM of the second day)
        return theta


    def _override_stomach(self):
        """Converts carry-over meal compartments into a forcing rate of appearance.

        Evolves the previous segment's per-meal gut compartments forward (with no
        new meal input) under the previous parameters, accumulates the resulting
        rate of appearance into ``previous_ra``, and zeroes the meal entries of
        ``x0`` so the carry-over is handled entirely through the forcing term.
        """
        x0 = self.x0

        # Free Backward-Euler evolution of meal gut compartments (no new meal input).
        # Layout: [Qsto1_B, Qsto2_B, Qgut_B, Qsto1_L, ..., Qgut_H]
        xk = np.zeros(24)
        xk[0]  = x0["Qsto1_B_0"] if "Qsto1_B_0" in x0 else np.float64(0.0)
        xk[1]  = x0["Qsto2_B_0"] if "Qsto2_B_0" in x0 else np.float64(0.0)
        xk[2]  = x0["Qgut_B_0"]  if "Qgut_B_0"  in x0 else np.float64(0.0)
        xk[3]  = x0["Qsto1_L_0"] if "Qsto1_L_0" in x0 else np.float64(0.0)
        xk[4]  = x0["Qsto2_L_0"] if "Qsto2_L_0" in x0 else np.float64(0.0)
        xk[5]  = x0["Qgut_L_0"]  if "Qgut_L_0"  in x0 else np.float64(0.0)
        xk[6]  = x0["Qsto1_D_0"] if "Qsto1_D_0" in x0 else np.float64(0.0)
        xk[7]  = x0["Qsto2_D_0"] if "Qsto2_D_0" in x0 else np.float64(0.0)
        xk[8]  = x0["Qgut_D_0"]  if "Qgut_D_0"  in x0 else np.float64(0.0)
        xk[9]  = x0["Qsto1_S_0"] if "Qsto1_S_0" in x0 else np.float64(0.0)
        xk[10] = x0["Qsto2_S_0"] if "Qsto2_S_0" in x0 else np.float64(0.0)
        xk[11] = x0["Qgut_S_0"]  if "Qgut_S_0"  in x0 else np.float64(0.0)
        xk[12] = x0["Qsto1_H_0"] if "Qsto1_H_0" in x0 else np.float64(0.0)
        xk[13] = x0["Qsto2_H_0"] if "Qsto2_H_0" in x0 else np.float64(0.0)
        xk[14] = x0["Qgut_H_0"]  if "Qgut_H_0"  in x0 else np.float64(0.0)
        xk[15] = x0["Qsto1_B2_0"] if "Qsto1_B2_0" in x0 else np.float64(0.0)
        xk[16] = x0["Qsto2_B2_0"] if "Qsto2_B2_0" in x0 else np.float64(0.0)
        xk[17] = x0["Qgut_B2_0"] if "Qgut_B2_0" in x0 else np.float64(0.0)
        xk[18] = x0["Qsto1_L2_0"] if "Qsto1_L2_0" in x0 else np.float64(0.0)
        xk[19] = x0["Qsto2_L2_0"] if "Qsto2_L2_0" in x0 else np.float64(0.0)
        xk[20] = x0["Qgut_L2_0"] if "Qgut_L2_0" in x0 else np.float64(0.0)
        xk[21] = x0["Qsto1_S2_0"] if "Qsto1_S2_0" in x0 else np.float64(0.0)
        xk[22] = x0["Qsto2_S2_0"] if "Qsto2_S2_0" in x0 else np.float64(0.0)
        xk[23] = x0["Qgut_S2_0"] if "Qgut_S2_0" in x0 else np.float64(0.0)
        previous_ra = np.zeros(self.tsteps)
        for k in range(self.tsteps):
            xk[0]  = xk[0]  / (1 + self.theta_prev["kempt"])
            xk[1]  = (xk[1]  + self.theta_prev["kempt"] * xk[0])  / (1 + self.theta_prev["kempt"])
            xk[2]  = (xk[2]  + self.theta_prev["kempt"] * xk[1])  / (1 + self.theta_prev["kabs_B"])
            xk[3]  = xk[3]  / (1 + self.theta_prev["kempt"])
            xk[4]  = (xk[4]  + self.theta_prev["kempt"] * xk[3])  / (1 + self.theta_prev["kempt"])
            xk[5]  = (xk[5]  + self.theta_prev["kempt"] * xk[4])  / (1 + self.theta_prev["kabs_L"])
            xk[6]  = xk[6]  / (1 + self.theta_prev["kempt"])
            xk[7]  = (xk[7]  + self.theta_prev["kempt"] * xk[6])  / (1 + self.theta_prev["kempt"])
            xk[8]  = (xk[8]  + self.theta_prev["kempt"] * xk[7])  / (1 + self.theta_prev["kabs_D"])
            xk[9]  = xk[9]  / (1 + self.theta_prev["kempt"])
            xk[10] = (xk[10] + self.theta_prev["kempt"] * xk[9])  / (1 + self.theta_prev["kempt"])
            xk[11] = (xk[11] + self.theta_prev["kempt"] * xk[10]) / (1 + self.theta_prev["kabs_S"])
            xk[12] = xk[12] / (1 + self.theta_prev["kempt"])
            xk[13] = (xk[13] + self.theta_prev["kempt"] * xk[12]) / (1 + self.theta_prev["kempt"])
            xk[14] = (xk[14] + self.theta_prev["kempt"] * xk[13]) / (1 + self.theta_prev["kabs_H"])
            xk[15] = xk[15] / (1 + self.theta_prev["kempt"])
            xk[16] = (xk[16] + self.theta_prev["kempt"] * xk[15]) / (1 + self.theta_prev["kempt"])
            xk[17] = (xk[17] + self.theta_prev["kempt"] * xk[16]) / (1 + self.theta_prev["kabs_B2"])
            xk[18] = xk[18] / (1 + self.theta_prev["kempt"])
            xk[19] = (xk[19] + self.theta_prev["kempt"] * xk[18]) / (1 + self.theta_prev["kempt"])
            xk[20] = (xk[20] + self.theta_prev["kempt"] * xk[19]) / (1 + self.theta_prev["kabs_L2"])
            xk[21] = xk[21] / (1 + self.theta_prev["kempt"])
            xk[22] = (xk[22] + self.theta_prev["kempt"] * xk[21]) / (1 + self.theta_prev["kempt"])
            xk[23] = (xk[23] + self.theta_prev["kempt"] * xk[22]) / (1 + self.theta_prev["kabs_S2"])
            previous_ra[k] = self.theta_prev["f"] * (
                self.theta_prev["kabs_B"] * xk[2]  + self.theta_prev["kabs_L"] * xk[5]  + self.theta_prev["kabs_D"] * xk[8] +
                self.theta_prev["kabs_S"] * xk[11] + self.theta_prev["kabs_H"] * xk[14] + self.theta_prev["kabs_B2"] * xk[17] +
                self.theta_prev["kabs_L2"] * xk[20] + self.theta_prev["kabs_S2"] * xk[23]
            )

        # set previous_ra.
        self.previous_ra = previous_ra

        # Reset meal compartments in x0 — carry-over is now handled via previous_ra.
        self.x0["Qsto1_B_0"] = np.float64(0.0)
        self.x0["Qsto2_B_0"] = np.float64(0.0)
        self.x0["Qgut_B_0"]  = np.float64(0.0)
        self.x0["Qsto1_L_0"] = np.float64(0.0)
        self.x0["Qsto2_L_0"] = np.float64(0.0)
        self.x0["Qgut_L_0"]  = np.float64(0.0)
        self.x0["Qsto1_D_0"] = np.float64(0.0)
        self.x0["Qsto2_D_0"] = np.float64(0.0)
        self.x0["Qgut_D_0"]  = np.float64(0.0)
        self.x0["Qsto1_S_0"] = np.float64(0.0)
        self.x0["Qsto2_S_0"] = np.float64(0.0)
        self.x0["Qgut_S_0"]  = np.float64(0.0)
        self.x0["Qsto1_H_0"] = np.float64(0.0)
        self.x0["Qsto2_H_0"] = np.float64(0.0)
        self.x0["Qgut_H_0"]  = np.float64(0.0)
        self.x0["Qsto1_B2_0"] = np.float64(0.0)
        self.x0["Qsto2_B2_0"] = np.float64(0.0)
        self.x0["Qgut_B2_0"] = np.float64(0.0)
        self.x0["Qsto1_L2_0"] = np.float64(0.0)
        self.x0["Qsto2_L2_0"] = np.float64(0.0)
        self.x0["Qgut_L2_0"] = np.float64(0.0)
        self.x0["Qsto1_S2_0"] = np.float64(0.0)
        self.x0["Qsto2_S2_0"] = np.float64(0.0)
        self.x0["Qgut_S2_0"] = np.float64(0.0)
