"""Streamlit session state helpers.

Streamlit reruns your script on every interaction. Session state is how you keep:
  - the current ProjectMemory instance
  - chat messages
  - the current session_id
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


def new_session_id() -> str:
    return uuid.uuid4().hex[:12]


@dataclass
class ChatMessage:
    role: str  # "user" or "assistant"
    content: str


@dataclass
class UISession:
    session_id: str = field(default_factory=new_session_id)
    messages: List[ChatMessage] = field(default_factory=list)

    last_run_meta: Optional[Dict[str, Any]] = None  # debug info from last call
