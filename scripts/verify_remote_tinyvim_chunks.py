import json
import hashlib
from pathlib import Path

base = Path("/mnt/c/Users/sshuser/codex_runs/hybrid-mamba/weights/tinyvim/parts3")
expected = json.loads((base / "chunk_hashes_expected.json").read_text())

bad = []
for name, exp_hash in sorted(expected.items()):
    path = base / name
    if not path.exists():
        bad.append((name, "missing"))
        continue
    got = hashlib.sha256(path.read_bytes()).hexdigest()
    if got != exp_hash:
        bad.append((name, got))

print(f"bad_count={len(bad)}")
for name, got in bad:
    print(f"{name} {got}")
