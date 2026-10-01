import json
from pathlib import Path
import matplotlib.pyplot as plt

for f in sorted(Path("runs").glob("*.json")):
    r = json.load(open(f))
    plt.plot(r["acc"], marker="o", label=f"lr={r['lr']:.0e}, decay={r['decay']}")
plt.xlabel("epoch"); plt.ylabel("test top-1"); plt.ylim(0.8, 0.96)   # zoom in on the part that differs
plt.legend(); plt.grid(alpha=0.3)
plt.savefig("runs/curves.png", dpi=150, bbox_inches="tight")