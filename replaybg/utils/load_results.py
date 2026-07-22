import os
import pickle


def load_results(path: str, name: str, prefix: str) -> dict:
    """Loads results previously pickled by :func:`save_results`.

    Parameters
    ----------
    path : str
        Directory where the file lives.
    name : str
        File name (with or without the ``.pkl`` extension), matching the ``name``
        passed to :func:`save_results`.
    prefix : str
        Prefix used when the file was saved (e.g. ``"twin"`` or ``"replay"``).

    Returns
    -------
    dict
        The unpickled results dictionary.
    """
    if not name.endswith('.pkl'):
        name += '.pkl'
    name = prefix + '_' + name
    file_path = os.path.join(path, name)
    with open(file_path, 'rb') as f:
        return pickle.load(f)
