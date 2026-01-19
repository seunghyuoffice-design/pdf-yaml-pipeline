"""HWP encryption detection module.

HWP 파일의 암호화 여부를 OLE 구조 분석으로 탐지.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Union

import olefile


def probe_hwp_encryption(file_path: Path) -> Dict[str, Union[str, bool]]:
    """Probe HWP file for encryption.

    Conservative approach:
    - OLE inspection failures are treated as 'possibly encrypted/corrupted'
    - 'reason' is meant for logs/metadata, not end-user display

    Args:
        file_path: Path to HWP file

    Returns:
        Dict with 'is_encrypted' (bool) and 'reason' (str)
    """
    if not olefile.isOleFile(str(file_path)):
        return {"is_encrypted": False, "reason": "Not an OLE container"}

    try:
        with olefile.OleFileIO(str(file_path)) as ole:
            streams = ["/".join(s) for s in ole.listdir(streams=True, storages=True)]
            suspicious = [s for s in streams if ("Encrypted" in s or "Password" in s or "Encryption" in s)]
            if suspicious:
                return {
                    "is_encrypted": True,
                    "reason": f"Suspicious streams: {suspicious[:5]}",
                }
    except Exception:
        return {
            "is_encrypted": True,
            "reason": "OLE inspection failed (encrypted/corrupted)",
        }

    return {"is_encrypted": False, "reason": "No encryption hints"}


__all__ = ["probe_hwp_encryption"]
