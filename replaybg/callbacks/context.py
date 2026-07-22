class ReplayContext:
    """Per-step view of a replay simulation passed to every callback.

    A single instance is created once per ``replay()`` call and refreshed in
    place each minute (no per-step allocation). Callbacks address input channels
    by *name* via ``get_input`` / ``set_input`` / ``add_input`` and read history
    directly from ``output_history`` / ``input_history`` using ``k``.

    The context is model-agnostic: inputs are read and written in the model's own
    units. Any human-unit conversion (e.g. insulin U, carbs grams) is the
    callback's responsibility, using data it carries itself (e.g. body weight).

    Note
    ----
    The data layer holds a bolus/meal value across the whole ``yts`` window.
    A callback that sets an input for a single minute therefore delivers a
    one-minute pulse, which is an acceptable approximation for correction
    boluses; spread the value over several steps yourself if it matters.

    Note
    ----
    ``shared`` records only values, not when they were written. Because
    callbacks run in the order they are passed to ``replay()``, a reader placed
    *before* a writer sees the value that writer published at an *earlier* step.
    When freshness matters, store the step alongside the value (e.g.
    ``ctx.shared["last_bolus"] = {"u": cb_u, "k": ctx.k}``) and compare against
    ``ctx.k`` on read — see ``CorrectionBolus`` / ``HypoTreatment`` for a worked
    example.

    ...
    Attributes
    ----------
    k : int
        Current integration minute (the step being decided).
    t_hour : float
        Hour-of-day at step ``k``.
    yts : int
        Data/sensor cadence in minutes (e.g. 5); use it to self-gate.
    u : numpy.ndarray
        Live, mutable input vector for this step (model units). Callbacks edit it
        through the helpers; later callbacks see earlier edits.
    output_history : numpy.ndarray
        Interstitial glucose outputs; valid up to index ``k-1``.
    input_history : numpy.ndarray
        Applied input rows; valid up to index ``k-1``.
    measurement : float
        Latest sensor measurement available at step ``k`` (held between sensor
        samples). When ``replay()`` is called without a sensor this carries the
        true model output, so policies reading it behave as if perfectly
        observed. Prefer this over ``output_history`` in control policies.
    measurement_history : numpy.ndarray
        Sensor measurement at integration resolution (zero-order held between
        samples); valid up to index ``k-1``. Mirrors ``output_history`` but
        reflects the noisy sensor signal when a sensor is supplied.
    data_to_input : dict
        Mapping from channel index to channel name.
    model : object
        The live model (read-only by convention) for advanced state such as
        plasma insulin (``model.Ip``) to derive insulin-on-board.
    shared : dict
        Free-form store for callbacks to exchange data by name. Empty at the
        start of the run; entries persist until overwritten and are never
        cleared between steps. Keys are a contract between cooperating
        callbacks: any callback able to produce a quantity may publish it, and
        readers need not know which class did.

    Methods
    -------
    get_input(name):
        Returns the current value of an input channel.
    set_input(name, value):
        Sets an input channel to a value.
    add_input(name, value):
        Adds a value to an input channel.
    log(**fields):
        Appends an action record to the action log.
    """

    def __init__(self, rbg_data, model, output_history, input_history):
        """Constructs all the necessary attributes for the ReplayContext object.

        Parameters
        ----------
        rbg_data : object
            The replay data object providing the inputs, time grid and channel
            mapping.
        model : object
            The live model being replayed.
        output_history : numpy.ndarray
            The output trajectory buffer (filled as the replay progresses).
        input_history : numpy.ndarray
            The applied-input trajectory buffer (filled as the replay progresses).
        """
        self.k = 0
        self.t_hour = rbg_data.t_hour[0]
        self.yts = rbg_data.yts
        self.u = rbg_data.u[0].copy()
        self.output_history = output_history
        self.input_history = input_history
        self.measurement = output_history[0]
        self.measurement_history = output_history
        self.data_to_input = rbg_data.data_to_input
        self.model = model
        self.shared = {}

        self._name_to_idx = {name: idx for idx, name in rbg_data.data_to_input.items()}
        self._actions = []
        self._active_cb = None

    # -- lifecycle (called by the replay loop) --------------------------------

    def _advance(self, k, u):
        """Refresh the context for integration minute ``k`` with baseline ``u``."""
        self.k = k
        self.u = u

    # -- input accessors (name-based) -----------------------------------------

    def get_input(self, name) -> float:
        """Returns the current value of an input channel (model units).

        Parameters
        ----------
        name : str
            The channel name (as in ``data_to_input``).

        Returns
        -------
        float
            The current value of input channel ``name``.
        """
        return float(self.u[self._name_to_idx[name]])

    def set_input(self, name, value) -> None:
        """Sets an input channel to a value (model units).

        Parameters
        ----------
        name : str
            The channel name (as in ``data_to_input``).
        value : float
            The value to assign to the channel.
        """
        self.u[self._name_to_idx[name]] = value

    def add_input(self, name, value) -> None:
        """Adds a value (model units) to an input channel.

        Parameters
        ----------
        name : str
            The channel name (as in ``data_to_input``).
        value : float
            The value to add to the channel.
        """
        self.u[self._name_to_idx[name]] += value

    # -- action logging -------------------------------------------------------

    def log(self, **fields) -> None:
        """Appends an action record to the action log (``k``, ``callback`` auto-filled).

        Pass whatever fields make the action reconstructable later, e.g. the
        previous and new input values if you want to track what changed.

        Parameters
        ----------
        **fields
            Arbitrary keyword fields stored on the record alongside the
            auto-filled ``k`` (current step) and ``callback`` (active callback).
        """
        record = {
            "k": self.k,
            "callback": self._active_cb,
        }
        record.update(fields)
        self._actions.append(record)
