"""Гибридті жүйені (ережелер + ML) тест жиынында бағалау және таза ML-мен салыстыру.
Іске қосу: python -m src.evaluate_hybrid"""
import json
from collections import Counter
import numpy as np
from sklearn.model_selection import train_test_split
from .train import load, FEATURES, SEED, COST_FN, COST_FP, ROOT
from . import decision as D


def main():
    df = load()
    X, y = df[FEATURES], df["y"]
    _, Xte, _, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    rows = []
    for (_, r), yt in zip(Xte.iterrows(), yte.values):
        c = {k: (None if (isinstance(v, float) and np.isnan(v)) else (v.item() if hasattr(v, "item") else v))
             for k, v in r.items()}
        rows.append((D.decide(c), yt))
    cnt = Counter(d["decision"] for d, _ in rows)
    churn_by = {k: round(float(np.mean([yt for d, yt in rows if d["decision"] == k])), 4) for k in cnt}
    hard_src = Counter(d["source"] for d, _ in rows)
    # Шығын: "шара қолданылады" = HIGH_RISK (REVIEW — маманға, шара қолданылмайды деп алынады)
    def cost(offer_set):
        c = 0.0
        for d, yt in rows:
            offer = d["decision"] in offer_set
            if offer and yt == 0: c += COST_FP
            if (not offer) and yt == 1: c += COST_FN
        return c
    out = {"counts": dict(cnt), "churn_rate_by_decision": churn_by, "source_counts": dict(hard_src),
           "cost_offer_HIGH_RISK_only": cost({"HIGH_RISK"}),
           "cost_offer_HIGH_RISK_and_REVIEW": cost({"HIGH_RISK", "REVIEW_REQUIRED"}),
           "n": len(rows)}
    (ROOT / "reports" / "hybrid_eval.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
