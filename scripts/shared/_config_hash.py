"""
_config_hash.py — Shared pvlib_config hash helper.

Imported by:
  - scripts/shared/tb_config_loader.py  (writes pvlib_config_hash SERVER_SCOPE attr)
  - scripts/shared/find_config_drift.py (compares master hash vs TB attr)

The algorithm must be identical in both callers — co-locating it here is the
single-source-of-truth guarantee.

Algorithm
---------
  SHA-1( json.dumps(pvlib_config_dict, sort_keys=True, separators=(",",":")), utf-8 )
  → hexdigest()[:12]

The sort_keys=True and compact separators ensure the hash is stable regardless
of insertion order in the source dict.

DO NOT change this algorithm without bumping all existing pvlib_config_hash
SERVER_SCOPE attributes across the fleet (force-overwrite with tb_config_loader.py).
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict


def compute_hash(pvlib_config: Dict[str, Any]) -> str:
    """Compute a stable 12-char SHA-1 fingerprint of a pvlib_config dict.

    Parameters
    ----------
    pvlib_config : dict
        The pvlib_config blob as returned by tb_config_loader.build_pvlib_config().

    Returns
    -------
    str
        First 12 hex characters of the SHA-1 digest.
    """
    payload = json.dumps(pvlib_config, sort_keys=True, separators=(",", ":"),
                         default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
