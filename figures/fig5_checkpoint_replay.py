"""Regenerate Fig. 5 (appliance-level curtailment and shifting, 27 July 2018, CADR Seed 0)
directly from the archived Seed-0 checkpoint, and save the plotted values as CSV.

Usage (normally called by figures/make_fig5_device_schedule.py):
    python figures/fig5_checkpoint_replay.py <project_root> <output_dir>
"""
import sys
from pathlib import Path
import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(sys.argv[1]).resolve()
OUT = Path(sys.argv[2]).resolve()
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))
from env.env import Environment  # noqa: E402
from agent.agent_ddqn import DDQNAgent  # noqa: E402

DAY = 208
run = ROOT / "results" / "seed_0" / "training"
cfg = yaml.safe_load((run / "config.yaml").read_text(encoding="utf8"))
env = Environment(data_ids=cfg["environment"]["house_ids"], cfg_override=cfg)
state = env.reset(day=DAY, mode="test")
agent = DDQNAgent(len(state), env.num_actions, cfg_override=cfg,
                  action_costs=np.mean(env.all_actions, axis=1))
ckpt = agent.load(str(run / "ddqn_best.pth"), load_optimizer=False)
env.set_household_coefficients(np.asarray(ckpt["household_coefficients"], dtype=float))
agent.policy_net.eval()
state = env.reset(day=DAY, mode="test")
for _ in range(24):
    state, _, done, _ = env.step(int(agent.greedy_action(state)))

names = list(cfg["environment"]["DEVICES"])
label = {"air": "AC", "car": "EV", "clotheswasher": "WM", "dishwasher": "DW", "dry": "Dryer"}
houses = cfg["environment"]["house_ids"]
base = env.device_before
curt = env.device_curtailed
out_ = env.device_shifted_out
in_ = env.device_shifted_in
after = base - curt - out_ + in_

rows = []
panels = []
for i, hid in enumerate(houses):
    for d, nm in enumerate(names):
        if base[:, i, d].sum() < 1e-9 and in_[:, i, d].sum() < 1e-9:
            continue
        if curt[:, i, d].sum() < 1e-9 and out_[:, i, d].sum() < 1e-9 and nm != "air":
            continue  # appliance operated but was not adjusted
        panels.append((i, hid, d, nm))
        for h in range(24):
            rows.append([hid, label.get(nm, nm), h + 1, base[h, i, d], after[h, i, d],
                         curt[h, i, d], out_[h, i, d], in_[h, i, d]])

np.savetxt(OUT / "fig5_plotted_values.csv", np.array(rows, dtype=object), fmt="%s", delimiter=",",
           header="house_id,appliance,hour,baseline_kWh,after_DR_kWh,curtailed_kWh,shifted_out_kWh,shifted_in_kWh",
           comments="")

hours = np.arange(1, 25)
w = 0.4
fig, axes = plt.subplots(len(panels), 1, figsize=(7.5, 1.35 * len(panels) + 0.8), sharex=True)
eu_of = {h: f"EU {k + 1}" for k, h in enumerate(houses)}
prev = None
for ax, (i, hid, d, nm) in zip(axes, panels):
    b, a = base[:, i, d], after[:, i, d]
    c, o, s = curt[:, i, d], out_[:, i, d], in_[:, i, d]
    ax.bar(hours - w / 2, b, w, color="0.85", edgecolor="0.4", linewidth=0.5)
    ax.bar(hours - w / 2, c, w, bottom=b - c, color="white", edgecolor="tab:red", hatch="////", linewidth=0.5)
    ax.bar(hours - w / 2, o, w, bottom=b - c - o, color="white", edgecolor="tab:orange", hatch="xxx", linewidth=0.5)
    ax.bar(hours + w / 2, a, w, color="tab:blue", edgecolor="black", linewidth=0.5)
    ax.bar(hours + w / 2, s, w, bottom=a - s, color="white", edgecolor="tab:green", hatch="\\\\\\", linewidth=0.5)
    ax.text(0.01, 0.80, f"{eu_of[hid]} ({hid}) – {label.get(nm, nm)}", transform=ax.transAxes, fontsize=8, fontweight="bold")
    ax.set_ylim(0, max(b.max(), a.max(), 0.5) * 1.25)
    ax.tick_params(labelsize=7)
axes[-1].set_xticks(np.arange(1, 25, 2))
axes[-1].set_xlabel("Hour")
fig.text(0.01, 0.5, "Energy (kWh)", rotation=90, va="center")
fig.legend(handles=[Patch(facecolor="0.85", edgecolor="0.4", label="Baseline"),
                    Patch(facecolor="tab:blue", edgecolor="black", label="After DR"),
                    Patch(facecolor="white", edgecolor="tab:red", hatch="////", label="Curtailed"),
                    Patch(facecolor="white", edgecolor="tab:orange", hatch="xxx", label="Shifted out"),
                    Patch(facecolor="white", edgecolor="tab:green", hatch="\\\\\\", label="Shifted in")],
           loc="upper center", ncol=5, fontsize=7, frameon=False)
fig.tight_layout(rect=(0.03, 0, 1, 0.96))
fig.savefig(OUT / "device_schedule_day_208_regenerated.pdf")
fig.savefig(OUT / "device_schedule_day_208_regenerated.png", dpi=200)

tot = {label.get(nm, nm) + f"_{hid}": (curt[:, i, d].sum(), out_[:, i, d].sum(), in_[:, i, d].sum())
       for i, hid, d, nm in panels}
print("panel totals (curtailed, shifted out, shifted in):")
for k, v in tot.items():
    print(f"  {k}: {v[0]:.2f}, {v[1]:.2f}, {v[2]:.2f}")
