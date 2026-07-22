from typing import Dict

import numpy as np

from callbacks.context import ReplayContext
from environment import Environment
from twinner.twinner import Twinner
from utils.save_results import save_results


class ReplayBG:
    """A class that represents the core entry point of the ReplayBG framework.

    ReplayBG is a digital-twin framework for Type 1 Diabetes glucose dynamics.
    It fits a physiological model to recorded data (the twinning step, exposed by
    :meth:`twin`) and then simulates counterfactual scenarios with the fitted
    model (the replay step, exposed by :meth:`replay`).

    ...
    Attributes
    ----------
    environment : Environment
        An Environment object collecting the hyperparameters (integration time,
        random seed, plot mode, verbosity) shared by the twinning and replay
        procedures.

    Methods
    -------
    twin(rbg_data, model, unknown_parameters_prior, ...):
        Runs the ReplayBG twinning (model identification) procedure.
    replay(rbg_data, model, theta, ...):
        Runs the ReplayBG replay (forward simulation) procedure.
    """

    def __init__(self,
                 ts: int = 1,
                 seed: int = 1,
                 plot_mode: bool = True, verbose: bool = True
                 ):
        """Constructs all the necessary attributes for the ReplayBG object.

        Parameters
        ----------
        ts : int, optional, default : 1
            An integer that specifies the integration time (in minutes).
        seed : int, optional, default : 1
            An integer that specifies the random seed. For reproducibility.
        plot_mode : bool, optional, default : True
            A boolean that specifies whether to show the plot of the results or not.
        verbose : bool, optional, default : True
            A boolean that specifies the verbosity of ReplayBG.
        """

        # TODO: Validate input

        # Initialize the environment parameters
        self.environment = Environment(ts=ts,
                                       seed=seed,
                                       plot_mode=plot_mode,
                                       verbose=verbose)

    def twin(self, rbg_data,
             model: object = None,
             unknown_parameters_prior: Dict = None,
             correlations: Dict = None,
             n_starts: int = 64,
             parallelize: bool = False, n_jobs: int | None = None,
             log_history: bool = False,
             path: str | None = None, save_name: str | None = None,
             ) -> Dict:
        """Runs the ReplayBG twinning procedure.

        Estimates the unknown model parameters ``theta`` from the recorded data
        via MAP estimation (multi-start Powell optimisation), delegating the work
        to a :class:`~twinner.twinner.Twinner`.

        If ``path`` is provided, the twinning results (``theta``, ``history`` and
        ``rbg_data``) are pickled to ``<path>/<name>.pkl``. When ``name`` is
        omitted it defaults to ``twin_YYYY_mm_dd.pkl``.

        Parameters
        ----------
        rbg_data : object
            A data object (e.g. ``MultiMealT1DData``) holding the inputs, output
            timestamps and observed glucose values consumed by the model.
        model : object, optional, default : None
            A model instance implementing the model interface contract
            (``reset(theta_dict)``, ``step(u, t)``, ``output(t)``).
        unknown_parameters_prior : dict, optional, default : None
            A dictionary defining which parameters to estimate. Each entry maps a
            parameter name to a dict with keys ``prior`` (a distribution with
            ``evaluate`` and ``sample``), ``min``, ``max`` and optional
            ``integer``.
        correlations : dict, optional, default : None
            Optional pairwise prior correlations between parameters, specified as
            ``{(name_a, name_b): rho}`` with ``rho`` in ``[-1, 1]`` (unlisted
            pairs default to 0). When provided, the joint prior becomes a
            Gaussian copula over the named parameters: their marginals stay
            exactly as declared in ``unknown_parameters_prior`` and only the
            dependence structure is added. Correlated parameters must use a
            distribution that provides a ``cdf`` method. When ``None`` the
            parameters are treated as independent (the previous behaviour).
        n_starts : int, optional, default : 64
            An integer that specifies the number of multi-start optimisations.
        parallelize : bool, optional, default : False
            A boolean that specifies whether to run the multi-start optimisation
            in parallel using multiprocessing.
        n_jobs : int or None, optional, default : None
            The number of parallel jobs to use. If ``None``, all available CPUs
            are used.
        log_history : bool, optional, default : False
            A boolean that specifies whether to record the optimisation history.
        path : str or None, optional, default : None
            The directory where the results are pickled. If ``None``, results are
            not saved to disk.
        save_name : str or None, optional, default : None
            The file name used when saving the results. If ``None``, it defaults
            to ``twin_YYYY_mm_dd.pkl``.

        Returns
        -------
        dict
            A dictionary with keys ``theta`` (the estimated parameters as a
            name->value mapping), ``correlations`` (the prior correlations used,
            or ``None``) and ``history`` (the optimisation history when
            ``log_history`` is ``True``, otherwise ``None``).
        """
        # TODO: validate the inputs

        # Initialize the twinner
        twinner = Twinner(parallelize=parallelize, n_jobs=n_jobs, n_starts=n_starts, log_history=log_history,
                          verbose=self.environment.verbose)

        # Run the twinning procedure
        theta_estimated = twinner.twin(model=model,
                                       rbg_data=rbg_data,
                                       unknown_parameters_prior=unknown_parameters_prior,
                                       correlations=correlations)

        ret = {
            'theta': dict(zip(unknown_parameters_prior.keys(), theta_estimated['x'])),
            'correlations': correlations,
            'history': twinner.history if log_history else None,
        }

        # Save results if a destination path was provided
        if path is not None:
            save_results({**ret, 'rbg_data': rbg_data}, path, save_name, prefix='twin')

        return ret

    def replay(self, rbg_data,
               model: object = None,
               callbacks: list | None = None,
               sensor: object = None,
               path: str | None = None, save_name: str | None = None,
               ) -> Dict:
        """Runs ReplayBG replay procedure.

        Replays the recorded inputs through the model, optionally letting
        user-supplied control policies act at every integration minute. Each
        ``callback`` (a :class:`~callbacks.callback.ReplayCallback`) is invoked
        before the model steps and may modify the inputs for the current step via
        the :class:`~callbacks.context.ReplayContext` it receives.

        An optional ``sensor`` (a :class:`~sensors.sensor.Sensor`) observes the model
        output and produces a measurement of it at its own cadence (``sensor.ts``). The
        measurement is exposed to the callbacks via ``ctx.measurement`` (held between
        samples), so closed-loop policies act on the realistic, noisy signal rather than
        the true output. The sensor is output-agnostic: it sees only ``model.output(k)``.
        When no sensor is supplied, ``ctx.measurement`` carries the true output, so
        behaviour is unchanged.

        If ``path`` is provided, the replay results (the returned dict plus
        ``rbg_data``) are pickled to ``<path>/<name>.pkl``. When ``name`` is
        omitted it defaults to ``replay_YYYY_mm_dd.pkl``.

        Parameters
        ----------
        rbg_data : object
            A data object (e.g. ``MultiMealT1DData``) holding the inputs to be
            replayed through the model.
        model : object, optional, default : None
            A model instance implementing the model interface contract
            (``reset(theta_dict)``, ``step(u, t)``, ``output(t)``).
        callbacks : list or None, optional, default : None
            A list of :class:`~callbacks.callback.ReplayCallback` control policies
            invoked before each model step. Each may modify the current-step
            inputs to implement a closed-loop policy.
        sensor : object, optional, default : None
            An optional :class:`~sensors.sensor.Sensor` that observes the model
            output and produces a (noisy, sub-sampled) measurement at its own
            cadence. When ``None``, callbacks observe the true model output.
        path : str or None, optional, default : None
            The directory where the results are pickled. If ``None``, results are
            not saved to disk.
        save_name : str or None, optional, default : None
            The file name used when saving the results. If ``None``, it defaults
            to ``replay_YYYY_mm_dd.pkl``.

        Returns
        -------
        dict
            A dictionary with keys:
            ``output`` (predicted interstitial glucose at integration resolution),
            ``input`` (applied inputs at integration resolution, shape
            ``(tsteps, n)``),
            ``data_to_input`` (channel index -> name mapping),
            ``actions`` (flat list of action records logged by callbacks) and,
            only when a sensor is supplied,
            ``measurement`` (sensor samples at sensor cadence) and
            ``measurement_time`` (integration-step index of each sample).
        """
        # TODO: validate the inputs

        n_ch = rbg_data.u.shape[1]
        out = np.zeros(rbg_data.tsteps, )
        replayed_u = np.zeros((rbg_data.tsteps, n_ch))
        out[0] = model.output(0)
        replayed_u[0] = rbg_data.u[0]

        callbacks = callbacks or []
        for cb in callbacks:
            cb.rbg_data = rbg_data
        ctx = ReplayContext(rbg_data=rbg_data, model=model,
                            output_history=out, input_history=replayed_u)

        use_sensor = sensor is not None
        if use_sensor:
            # Seed the global RNG and (re)connect the sensor under that seed so its
            # whole error realization (sampled parameters + per-step noise) is
            # reproducible, regardless of when the sensor object was constructed.
            np.random.seed(self.environment.seed)
            sensor.connect_new(connected_at=0)
            ts = sensor.ts
            n_samp = (rbg_data.tsteps - 1) // ts + 1
            measurement = np.full(n_samp, np.nan)
            measurement_time = np.zeros(n_samp, dtype=int)
            measurement[0] = sensor.measure(out[0], past_values=out[:0], t=0.0)
            measurement_time[0] = 0
            # Zero-order-held measurement at integration resolution (for ctx history).
            meas_hold = out.copy()
            meas_hold[0] = measurement[0]
            ctx.measurement_history = meas_hold

        for k in range(1, rbg_data.tsteps):
            ctx._advance(k, rbg_data.u[k].copy())
            # Latest measurement available to a controller acting at step k.
            ctx.measurement = meas_hold[k - 1] if use_sensor else out[k - 1]
            for cb in callbacks:
                ctx._active_cb = cb.name or type(cb).__name__
                cb.action(ctx)
            model.step(ctx.u, k)
            out[k] = model.output(k)
            replayed_u[k] = ctx.u

            if use_sensor:
                if k % ts == 0:
                    if (k + sensor.t_offset) % sensor.max_lifetime == 0:
                        sensor.connect_new(connected_at=k)
                    j = k // ts
                    measurement[j] = sensor.measure(
                        out[k], past_values=out[:k],
                        t=(k - sensor.connected_at) / (24 * 60))
                    measurement_time[j] = k
                    meas_hold[k] = measurement[j]
                else:
                    meas_hold[k] = meas_hold[k - 1]

        ret = {
            'output': out,
            'input': replayed_u,
            'data_to_input': rbg_data.data_to_input,
            'actions': ctx._actions,
        }
        if use_sensor:
            ret['measurement'] = measurement
            ret['measurement_time'] = measurement_time

        # Save results if a destination path was provided
        if path is not None:
            save_results({**ret}, path, save_name, prefix='replay')

        return ret
