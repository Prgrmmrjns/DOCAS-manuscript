import os
import pickle
from datetime import datetime


def save_results(results: dict, path: str, name: str | None, prefix: str) -> str:
    """Pickles ``results`` to ``<path>/<name>.pkl``, creating ``path`` if needed.

    Parameters
    ----------
    results : dict
        The dictionary of objects to persist.
    path : str
        Directory where the file will be written. Created if missing.
    name : str or None
        File name (with or without the ``.pkl`` extension). If ``None``, defaults
        to ``f"{prefix}_{YYYY_mm_dd}"``.
    prefix : str
        Prefix used to build the default name when ``name`` is ``None`` (e.g.
        ``"twin"`` or ``"replay"``).

    Returns
    -------
    str
        The full path of the file written.
    """
    os.makedirs(path, exist_ok=True)
    if name is None:
        name = f"{datetime.now().strftime('%Y_%m_%d')}"
    if not name.endswith('.pkl'):
        name += '.pkl'
    name = prefix + '_' + name
    file_path = os.path.join(path, name)
    with open(file_path, 'wb') as f:
        pickle.dump(results, f)
    return file_path
