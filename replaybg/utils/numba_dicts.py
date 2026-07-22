import numpy as np
from numba import float64, types
from numba.typed import Dict


def to_typed_f64_dict(src=None, *, strict=False) -> Dict:
    """Convert a mapping to a Numba typed dictionary with float64 values.

    This helper normalizes a Python dict-like object into a
    ``numba.typed.Dict`` with keys of type ``unicode_type`` and values of type
    ``float64``. It is useful when passing parameter dictionaries into Numba-
    compiled code or APIs that expect typed containers.

    Parameters
    ----------
    src : mapping, optional, default : None
        A mapping or mapping-like object to convert. If ``None``, an empty typed
        dictionary is returned.
    strict : bool, optional, default : False
        If ``True``, raise ``TypeError`` when a value cannot be cast to
        ``float64``. If ``False``, invalid entries are skipped silently.

    Returns
    -------
    numba.typed.Dict
        A typed dictionary with ``str`` keys and ``float64`` values.

    Raises
    ------
    TypeError
        If ``strict`` is ``True`` and a key/value pair cannot be converted to
        ``str``/``float64``.
    """
    out = Dict.empty(key_type=types.unicode_type, value_type=float64)
    if src is None:
        return out

    for k, v in src.items():
        try:
            out[str(k)] = np.float64(v)
        except Exception:
            if strict:
                raise TypeError(f"Cannot cast key={k!r}, value={v!r} to float32")
            # non-strict: ignore invalid entries
    return out