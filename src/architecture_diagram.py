"""Draws reports/figures/00_architecture.png. Run: python src/architecture_diagram.py"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
fig, ax = plt.subplots(figsize=(12, 6.2)); ax.set_xlim(0, 12); ax.set_ylim(0, 6.2); ax.axis("off")
def box(x, y, w, h, t, c):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.12", fc=c, ec="#374151", lw=1.3))
    ax.text(x + w/2, y + h/2, t, ha="center", va="center", fontsize=9.5, wrap=True)
def arrow(x1, y1, x2, y2, t="", c="#374151", dx=0, dy=0.12):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", color=c, lw=1.5))
    if t: ax.text((x1+x2)/2 + dx, (y1+y2)/2 + dy, t, ha="center", fontsize=8.5, color=c)
B, G, Y, R, P = "#dbeafe", "#dcfce7", "#fef9c3", "#fee2e2", "#ede9fe"
box(0.2, 4.6, 2.1, 1.1, "Weekly batch of\nloan applications\n(data_generator)", B)
box(3.0, 4.6, 2.3, 1.1, "Production model\nscores batch\n(registry.load)", G)
box(6.0, 4.6, 2.6, 1.1, "Drift detector\nPSI + KS per feature\nAUC vs baseline", Y)
box(9.4, 4.6, 2.4, 1.1, "Log + dashboard\nmonitoring_log.csv\ncharts", B)
box(6.0, 2.6, 2.6, 1.1, "ALERT\nretrain on last\n2 weeks (train.py)", R)
box(6.0, 0.5, 2.6, 1.1, "Validate candidate\nvs production on\nsame hold-out", Y)
box(9.4, 1.6, 2.4, 1.0, "Promote: new version\n= production,\nupdate reference", G)
box(9.4, 0.1, 2.4, 1.0, "Reject: keep\ncurrent model", R)
box(0.2, 1.2, 4.6, 1.9, "Model registry (registry.json)\nv1 archived · v2 archived · v3 production\n\nCI/CD (GitHub Actions): tests -> pipeline\n-> quality gate -> upload reports", P)
arrow(2.3, 5.15, 3.0, 5.15); arrow(5.3, 5.15, 6.0, 5.15)
arrow(8.6, 5.15, 9.4, 5.15, "no drift")
arrow(7.3, 4.6, 7.3, 3.7, "drift!", "#dc2626", dx=0.4, dy=0); arrow(7.3, 2.6, 7.3, 1.6)
arrow(8.6, 1.25, 9.4, 2.0, "better", "#16a34a", dx=-0.35, dy=0.15); arrow(8.6, 0.85, 9.4, 0.6, "worse", "#dc2626")
arrow(9.4, 2.4, 4.8, 2.4, "", "#7c3aed"); ax.text(6.9, 2.0, "", fontsize=8)
arrow(2.5, 3.1, 4.0, 4.6, "next week uses\nproduction model", "#7c3aed", dx=-1.25, dy=0)
ax.set_title("DriftGuard architecture: monitor -> detect -> retrain -> validate -> promote", fontsize=13, weight="bold")
fig.tight_layout(); fig.savefig("reports/figures/00_architecture.png", dpi=150)
