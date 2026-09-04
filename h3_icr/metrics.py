from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ICRMetrics:
    events: list[dict[str, Any]] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)

    def event(self, name: str, **fields: Any) -> None:
        self.events.append({"event": name, "time": time.time(), **fields})

    def to_dict(self) -> dict[str, Any]:
        return {"api": 1, "started_at": self.started_at, "events": list(self.events)}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, indent=2)
