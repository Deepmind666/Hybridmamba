from pathlib import Path

parts_dir = Path("/mnt/c/Users/sshuser/codex_runs/hybrid-mamba/weights/tinyvim/parts3")
out_path = Path("/mnt/c/Users/sshuser/codex_runs/hybrid-mamba/weights/tinyvim/tinyvim_b_300e.pth")

parts = sorted(parts_dir.glob("part_*.bin"))
if not parts:
    raise SystemExit(f"No parts found in {parts_dir}")

with out_path.open("wb") as w:
    for p in parts:
        data = p.read_bytes()
        w.write(data)

print(f"parts={len(parts)}")
print(f"bytes={out_path.stat().st_size}")
