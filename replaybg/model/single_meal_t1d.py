import numpy as np
from numba import float64, int16
from numba.typed import Dict
from numba import types

from environment import jitclass_

theta0_type = types.DictType(types.unicode_type, float64)
theta_prev_type = types.DictType(types.unicode_type, float64)
x0_type = types.DictType(types.unicode_type, float64)
JITCLASS_SPEC = [
    ("_G0", float64),
    ("_X0", float64),
    ("_Qsto1_0", float64),
    ("_Qsto2_0", float64),
    ("_Qgut_0", float64),
    ("_Isc10", float64),
    ("_Isc20", float64),
    ("_Ip0", float64),
    ("_IG0", float64),
    ("G", float64[:]),
    ("X", float64[:]),
    ("Qsto1", float64[:]),
    ("Qsto2", float64[:]),
    ("Qgut", float64[:]),
    ("Isc1", float64[:]),
    ("Isc2", float64[:]),
    ("Ip", float64[:]),
    ("IG", float64[:]),
    ("SI", float64),
    ("SG", float64),
    ("Gb", float64),
    ("p2", float64),
    ("r1", float64),
    ("r2", float64),
    ("tau", int16),
    ("ka2", float64),
    ("kd", float64),
    ("ke", float64),
    ("kabs", float64),
    ("beta", int16),
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
    ("n_u", int16),
    ("u", float64[:,:]),
    ("previous_ra", float64[:]),
]


@jitclass_(JITCLASS_SPEC)
class SingleMealT1DModel:
    """A physiological model of glucose-insulin dynamics for a single-meal T1D scenario.

    The model is a system of ODEs integrated with Backward Euler at a one-minute
    resolution. It chains a meal gut-absorption sub-system, a subcutaneous insulin
    pharmacokinetic sub-system, an insulin-action compartment and a glucose
    sub-system, and exposes the interstitial glucose as the observable output.
    The class is compiled with Numba's ``@jitclass``; all parameters are passed
    in through Numba typed dicts.

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
    SI : float
        Insulin sensitivity.
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
    kabs : float
        Meal gut-absorption rate constant.
    kempt : float
        Gastric emptying rate constant.
    beta : int
        Meal announcement delay (integer minutes).
    Ipb : float
        Basal plasma insulin derived from ``u2ss``.
    G, X, Qsto1, Qsto2, Qgut, Isc1, Isc2, Ip, IG : numpy.ndarray
        State trajectories (one entry per integration step) for plasma glucose,
        insulin action, the two gastric and the gut compartments, the two
        subcutaneous insulin compartments, plasma insulin and interstitial
        glucose.
    theta0 : numba.typed.Dict
        The initial parameter dict the model was constructed with.
    theta_prev : numba.typed.Dict
        Parameters of the previous segment, used to scale carry-over initial
        conditions across segments.
    x0 : numba.typed.Dict
        Initial-condition dict (carry-over state from a previous segment).
    tsteps : int
        Number of integration steps.
    n_u : int
        Number of input channels (5: meal, bolus, basal, forcing_ip, forcing_ra).
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
        theta_prev : numba.typed.Dict, optional
            Parameters of the previous segment, used to scale the carry-over
            initial conditions. Empty for a cold start.
        """
        # Initial values of the model parameters (just a subset) and states
        self.theta0 = theta0
        self.x0 = x0
        self.tsteps = np.int16(tsteps)

        # Number of inputs
        self.n_u = np.int16(5)

        # Steady-state basal insulin (u2ss)
        self.u2ss = np.float64(u2ss)

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

        # --- Insulin sensitivity ---
        self.SI = theta0["SI"] if "SI" in theta0 else np.float64(10.35e-4 / self.VG)

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

        # --- Meal gut-absorption rate ---
        self.kabs = theta0["kabs"] if "kabs" in theta0 else np.float64(0.012)
        # Shared gastric emptying rate constant
        self.kempt = theta0["kempt"] if "kempt" in theta0 else np.float64(0.18)

        # --- Meal announcement delay (integer minutes) ---
        self.beta = np.int16(theta0["beta"]) if "beta" in theta0 else np.int16(0)

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
        self._Qsto1_0 = self.x0["Qsto1_0"] if "Qsto1_0" in self.x0 else np.float64(0)
        self.Qsto1 = np.empty((self.tsteps,), dtype=np.float64)
        self.Qsto1[0] = self._Qsto1_0
        self._Qsto2_0 = self.x0["Qsto2_0"] if "Qsto2_0" in self.x0 else np.float64(0)
        self.Qsto2 = np.empty((self.tsteps,), dtype=np.float64)
        self.Qsto2[0] = self._Qsto2_0
        self._Qgut_0 = self.x0["Qgut_0"] if "Qgut_0" in self.x0 else np.float64(0)
        self.Qgut = np.empty((self.tsteps,), dtype=np.float64)
        self.Qgut[0] = self._Qgut_0

        # Insulin action

        # X = (SI/VI)*(Ip - Ipb): when SI or VI change across segments, X0 must be
        # rescaled by the ratio of (SI/VI) for the current segment to
        # (SI/VI) for the previous segment.
        if "X0" in self.x0 and len(self.theta_prev) > 0:
            self._X0 = (self.SI / self.VI) / (self.theta_prev["SI"] / self.theta_prev["VI"]) * self.x0["X0"]
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
        self.u[1, 0] = self.u2ss
        self.u[2, 0] = 0.0  # no bolus at t=0
        if len(self.theta_prev) == 0:
            self.previous_ra = np.zeros(self.tsteps)

    def step(self, u: float64[:], t: float64):
        """Advances the model by one minute using Backward Euler integration.

        Parameters
        ----------
        u : numpy.ndarray
            The input vector for this step (5 channels: meal, bolus, basal,
            forcing_ip, forcing_ra).
        t : int
            The integration step index (minute) to compute.
        """
        # Store current inputs in the history buffer so delayed values can be
        # retrieved at this or future timesteps via index arithmetic.
        self.u[0, t] = u[0]
        # Apply meal announcement delay beta: use the carb input from beta
        # minutes ago. If we are still within the first beta minutes of the
        # simulation (t < beta), treat the delayed input as zero.
        u_m = self.u[0, t - self.beta] if (t - self.beta) >= 0 else 0
        self.u[1, t] = u[1]
        self.u[2, t] = u[2]
        # Apply insulin absorption delay tau: use the insulin input from tau
        # minutes ago. Before tau minutes have elapsed, fall back to u2ss so
        # the plasma insulin chain starts at its basal steady state.
        u_i = self.u[1, t - self.tau] + self.u[2, t - self.tau] if (t - self.tau) >= 0 else self.u2ss
        self.u[3, t] = u[3]
        u_fip = self.u[3, t]  # forcing ip
        self.u[4, t] = u[4]
        u_fra = self.u[4, t]  # forcing ra

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

        # --- Gut absorption chain (Backward Euler, Gauss-Seidel order) ---
        # Solid stomach (Qsto1) → liquid stomach (Qsto2) → intestine (Qgut).
        self.Qsto1[t] = (self.Qsto1[t-1] + u_m) * k1
        self.Qsto2[t] = (self.Qsto2[t-1] + self.kempt * self.Qsto1[t]) * k2
        self.Qgut[t] = (self.Qgut[t-1] + self.kempt * self.Qsto2[t]) / (1 + self.kabs)

        # --- Insulin pharmacokinetic chain (Backward Euler, Gauss-Seidel order) ---
        # Subcutaneous compartment 1 → compartment 2 → plasma
        self.Isc1[t] = (self.Isc1[t-1] + u_i) * kd_fac
        self.Isc2[t] = (self.Isc2[t-1] + self.kd * self.Isc1[t]) / (1 + self.ka2)
        self.Ip[t] = (self.Ip[t-1] + self.ka2 * self.Isc2[t] + u_fip) / (1 + self.ke)

        # --- Insulin action and glucose (Backward Euler) ---
        # X depends on Ip[t] (just computed above)
        self.X[t] = (self.X[t-1] + self.p2 * (self.SI / self.VI) * (self.Ip[t] - self.Ipb)) / (1 + self.p2)
        # G depends on X[t] and Qgut[t] (just computed). The risk coefficient
        # is frozen at G[t-1] (semi-implicit) to keep the update linear in G[t].
        self.G[t] = (self.G[t-1] + self.SG * self.Gb + self.f * self.kabs * self.Qgut[t] / self.VG + self.previous_ra[t-1] / self.VG + u_fra / self.VG) / (1 + self.SG + risk * self.X[t])
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
        final_x0["G0"]      = self.G[-1]
        final_x0["X0"]      = self.X[-1]
        final_x0["IG0"]     = self.IG[-1]
        final_x0["Isc10"]   = self.Isc1[-1]
        final_x0["Isc20"]   = self.Isc2[-1]
        final_x0["Ip0"]     = self.Ip[-1]
        final_x0["Qsto1_0"] = self.Qsto1[-1]
        final_x0["Qsto2_0"] = self.Qsto2[-1]
        final_x0["Qgut_0"]  = self.Qgut[-1]
        return final_x0

    def get_theta(self):
        """Returns the current model parameters as a Numba typed dict.

        Packs all parameters set by :meth:`_reset_theta` into a typed dict
        suitable for passing as ``theta_prev`` to a subsequent segment's
        ``__init__`` call.

        Returns
        -------
        numba.typed.Dict
            A typed dict (unicode -> float64) with keys matching the
            ``unknown_parameters_prior`` naming convention.
        """
        theta = Dict.empty(key_type=types.unicode_type, value_type=float64)
        theta["f"]     = self.f
        theta["VG"]    = self.VG
        theta["VI"]    = self.VI
        theta["alpha"] = np.float64(self.alpha)
        theta["SI"]    = self.SI
        theta["SG"]    = self.SG
        theta["Gb"]    = self.Gb
        theta["p2"]    = self.p2
        theta["r1"]    = self.r1
        theta["r2"]    = self.r2
        theta["ka2"]   = self.ka2
        theta["kd"]    = self.kd
        theta["ke"]    = self.ke
        theta["tau"]   = np.float64(self.tau)
        theta["kabs"]  = self.kabs
        theta["kempt"] = self.kempt
        theta["beta"]  = np.float64(self.beta)
        return theta

    def _override_stomach(self):
        """Converts carry-over meal compartments into a forcing rate of appearance.

        Evolves the previous segment's meal gut compartments forward (with no new
        meal input) under the previous parameters, accumulates the resulting rate
        of appearance into ``previous_ra``, and zeroes the meal entries of ``x0``
        so the carry-over is handled entirely through the forcing term.
        """
        x0 = self.x0

        # Free Backward-Euler evolution of the single meal gut compartment.
        xk = np.zeros(3)
        xk[0] = x0["Qsto1_0"] if "Qsto1_0" in x0 else np.float64(0.0)
        xk[1] = x0["Qsto2_0"] if "Qsto2_0" in x0 else np.float64(0.0)
        xk[2] = x0["Qgut_0"]  if "Qgut_0"  in x0 else np.float64(0.0)
        previous_ra = np.zeros(self.tsteps)
        for k in range(self.tsteps):
            xk[0] = xk[0] / (1 + self.theta_prev["kempt"])
            xk[1] = (xk[1] + self.theta_prev["kempt"] * xk[0]) / (1 + self.theta_prev["kempt"])
            xk[2] = (xk[2] + self.theta_prev["kempt"] * xk[1]) / (1 + self.theta_prev["kabs"])
            previous_ra[k] = self.theta_prev["f"] * self.theta_prev["kabs"] * xk[2]

        # set previous_ra.
        self.previous_ra = previous_ra

        # Reset meal compartments in x0 — carry-over is now handled via previous_ra.
        self.x0["Qsto1_0"] = np.float64(0.0)
        self.x0["Qsto2_0"] = np.float64(0.0)
        self.x0["Qgut_0"]  = np.float64(0.0)
