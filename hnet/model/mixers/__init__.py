"""Sequence mixers (registered under the "mixer" kind of hnet.registry).

Submodules register themselves on import; ``hnet.model`` imports them. This file stays import-free so that
importing one mixer never pulls in another (and to avoid import cycles with ``hnet.model``).
"""
