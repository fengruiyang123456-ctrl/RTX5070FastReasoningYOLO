import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


def compute_stats(latencies_ms: List[float]) -> Dict[str, float]:
    if not latencies_ms:
        return {"avg_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0, "fps": 0.0}
    data = np.array(latencies_ms, dtype=np.float32)
    avg_ms = float(np.mean(data))
    p95_ms = float(np.percentile(data, 95))
    p99_ms = float(np.percentile(data, 99))
    fps = 1000.0 / avg_ms if avg_ms > 0 else 0.0
    return {"avg_ms": avg_ms, "p95_ms": p95_ms, "p99_ms": p99_ms, "fps": fps}


def write_benchmark(
    out_dir: Path,
    name: str,
    stats: Dict[str, float],
    extra: Optional[Dict[str, float]] = None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"{name}_{ts}"
    json_path = out_dir / f"{base}.json"
    csv_path = out_dir / f"{base}.csv"

    payload = {"name": name, **stats}
    if extra:
        payload.update(extra)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(payload.keys()))
        writer.writeheader()
        writer.writerow(payload)

    return json_path
