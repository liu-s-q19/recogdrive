from __future__ import annotations

from typing import Optional

def resolve_reference_policy_checkpoint(
    checkpoint_path: Optional[str],
    reference_policy_checkpoint: Optional[str],
) -> str:
    """Default the reference policy to the student init checkpoint when omitted."""
    return reference_policy_checkpoint or checkpoint_path or ""
