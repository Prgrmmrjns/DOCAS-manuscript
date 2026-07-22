class ReplayCallback:
    """Base class for replay control policies.

    A callback is invoked once per integration minute, *before* the model steps,
    so it can inspect the simulation so far and modify the inputs the model is
    about to consume. Subclasses:

    - store **hyperparameters** in ``__init__`` (supplied at the ``replay()`` call
      site when the instance is created);
    - keep **memory** in instance attributes, which persist across all timesteps
      of a single ``replay()`` call and remain inspectable on the instance after
      the run finishes;
    - share **state with other callbacks** through ``ctx.shared``, a free-form
      dict that persists across every step of the run; prefer it to holding
      references to other callback objects;
    - override :meth:`action` to read the simulation state from the context and
      modify the current-step inputs via ``ctx.set_input`` / ``ctx.add_input``.

    Callbacks run in the order they are passed to ``replay()``; each mutates the
    same input vector in place, so later callbacks observe earlier edits. The
    return value of :meth:`action` is ignored.

    ...
    Attributes
    ----------
    name : str
        Identifies this callback in the action log. Defaults to the class name.
        Set a distinct ``name`` when using two instances of the same class.
    rbg_data : object
        The replay ``rbg_data``, injected before the run starts so callbacks can
        access data-specific quantities (e.g. ``rbg_data.body_weight``). ``None``
        until ``replay()`` is called.

    Methods
    -------
    action(ctx):
        Inspects the context and modifies the current-step inputs in place.
    """

    #: Identifies this callback in the action log. Defaults to the class name.
    #: Set a distinct ``name`` when using two instances of the same class.
    name: str = None

    #: The replay ``rbg_data`` is injected here before the run starts, so
    #: callbacks can access data-specific quantities (e.g. ``rbg_data.body_weight``)
    #: for their own unit conversions. ``None`` until ``replay()`` is called.
    rbg_data = None

    def action(self, ctx) -> None:
        """Inspects ``ctx`` and modifies the current-step inputs in place.

        Parameters
        ----------
        ctx : ReplayContext
            The :class:`~callbacks.context.ReplayContext` for the current step.
        """
        raise NotImplementedError
