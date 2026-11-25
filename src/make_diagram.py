import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path
plt.rcParams["font.family"] = "DejaVu Sans"
fig, ax = plt.subplots(figsize=(11, 6.2)); ax.axis("off"); ax.set_xlim(0, 11); ax.set_ylim(0, 6.2)
def box(x, y, w, h, t, c="#dbe9f6", fs=9.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04", fc=c, ec="#333", lw=1.1))
    ax.text(x + w/2, y + h/2, t, ha="center", va="center", fontsize=fs)
def arr(x1, y1, x2, y2, t=None):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14, lw=1.2, color="#333"))
    if t: ax.text((x1+x2)/2, (y1+y2)/2 + .12, t, ha="center", fontsize=8)
box(0.1, 2.6, 1.5, 1.0, "Streamlit\nинтерфейс\n(app/ui.py)", "#fde9c9")
box(2.1, 2.6, 1.6, 1.0, "FastAPI\n/predict /explain\n/health /stats", "#fde9c9")
box(4.2, 4.6, 2.3, 1.0, "R1–R5 (VALID)\nДеректерді тексеру", "#f6d5d5")
box(4.2, 3.2, 2.3, 1.0, "R6–R8 (HARD)\nҚатаң шектеулер", "#f6d5d5")
box(4.2, 1.8, 2.3, 1.0, "ML моделі\nGradient Boosting\np = P(кету)", "#d4ecd4")
box(4.2, 0.4, 2.3, 1.0, "Шекті мән 0,41 ± 0,08\nсенімсіздік аймағы", "#d4ecd4")
box(7.2, 0.4, 2.0, 1.0, "R9–R15 (POST)\nҰсыныстар", "#f6d5d5")
box(7.2, 2.6, 3.5, 1.0, "Шешім + түсіндірме\n(іске қосылған ережелер,\nең әсерлі белгілер)", "#e5dbf4")
box(7.7, 4.6, 3.0, 1.0, "SQLite журналы\nlogs/decisions.db", "#e8e8e8")
arr(1.6, 3.1, 2.1, 3.1); arr(3.7, 3.1, 4.2, 5.1, "кіріс"); 
arr(5.35, 4.6, 5.35, 4.2); ax.text(5.45, 4.35, "дұрыс", fontsize=8); arr(5.35, 3.2, 5.35, 2.8); ax.text(5.45, 2.95, "ереже жоқ", fontsize=8)
arr(5.35, 1.8, 5.35, 1.4); arr(6.5, 0.9, 7.2, 0.9); arr(8.2, 1.4, 8.6, 2.6)
arr(6.5, 4.9, 7.6, 3.6); ax.text(6.55, 4.45, "қате → 422", fontsize=8)
arr(6.5, 3.7, 7.2, 3.3); ax.text(6.55, 3.85, "шешім", fontsize=8)
arr(9.2, 3.6, 9.2, 4.6, "жазу")
arr(7.2, 2.9, 3.7, 2.9); ax.text(5.5, 3.02, "", fontsize=7)
ax.set_title("Гибридті интеллектуалды жүйенің құрылымы", fontsize=11)
fig.tight_layout(); fig.savefig(Path("reports/figures/architecture.png"), dpi=150)
