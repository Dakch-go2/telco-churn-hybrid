"""Қосымша талдаулар: теңгерімсіздік тәсілдері, топтар бойынша тексеру, ереже қамтуы, жылдамдық.
Іске қосу: python -m src.extra_eval"""
import json, time
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, average_precision_score
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from .train import load, build_pre, FEATURES, SEED, COST_FN, COST_FP, ROOT
from . import decision as D, rules as R


def cm_metrics(y, p, t):
    pr = (p >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pr, labels=[0, 1]).ravel()
    return {"threshold": float(t), "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
            "precision": round(float(precision_score(y, pr, zero_division=0)), 4),
            "recall": round(float(recall_score(y, pr, zero_division=0)), 4),
            "cost": float(fp * COST_FP + fn * COST_FN)}


def main():
    df = load()
    X, y = df[FEATURES], df["y"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    out = {}
    # 1) теңгерімсіздік: class_weight жоқ модель
    pipe = Pipeline([("pre", build_pre()), ("clf", HistGradientBoostingClassifier(
        random_state=SEED, learning_rate=0.03, max_depth=5, max_iter=100))])
    oof = cross_val_predict(pipe, Xtr, ytr, cv=cv, method="predict_proba")[:, 1]
    ths = np.round(np.arange(0.05, 0.96, 0.01), 2)
    costs = [cm_metrics(ytr, oof, t)["cost"] for t in ths]
    t_best = float(ths[int(np.argmin(costs))])
    pipe.fit(Xtr, ytr)
    pte = pipe.predict_proba(Xte)[:, 1]
    out["noweight"] = {"pr_auc": round(float(average_precision_score(yte, pte)), 4),
                       "at_0.5": cm_metrics(yte, pte, 0.5), "at_cost": cm_metrics(yte, pte, t_best)}
    # 2) топтар бойынша (негізгі модель, шекті мән)
    model, meta = D.load_artifacts()
    p_main = model.predict_proba(Xte)[:, 1]
    thr = meta["threshold"]
    groups = {}
    for col, mapping in [("gender", None), ("SeniorCitizen", {0: "SeniorCitizen=0", 1: "SeniorCitizen=1"})]:
        for val in sorted(Xte[col].unique()):
            idx = (Xte[col] == val).values
            g = cm_metrics(yte.values[idx], p_main[idx], thr)
            g["n"] = int(idx.sum()); g["base_rate"] = round(float(yte.values[idx].mean()), 4)
            g["flag_rate"] = round(float((p_main[idx] >= thr).mean()), 4)
            groups[f"{col}={val}"] = g
    out["groups"] = groups
    # 3) ереже қамтуы (барлық тест жиыны)
    fired = {r.id: [0, 0] for r in R.RULES}   # [іске қосылу саны, олардың ішінде кеткендер]
    t_dec = []
    for (_, r), yt in zip(Xte.iterrows(), yte.values):
        c = {k: (None if (isinstance(v, float) and np.isnan(v)) else (v.item() if hasattr(v, "item") else v)) for k, v in r.items()}
        s0 = time.perf_counter(); res = D.decide(c); t_dec.append(time.perf_counter() - s0)
        for f in res["fired_rules"]:
            fired[f["id"]][0] += 1; fired[f["id"]][1] += int(yt)
    out["rule_coverage"] = {k: {"fired": v[0], "churned": v[1]} for k, v in fired.items()}
    # 4) жылдамдық
    sample = [dict(zip(Xte.columns, [None if (isinstance(v, float) and np.isnan(v)) else (v.item() if hasattr(v, "item") else v) for v in row])) for row in Xte.head(40).itertuples(index=False)]
    t_exp = []
    for c in sample:
        s0 = time.perf_counter(); D.local_explanation(c); t_exp.append(time.perf_counter() - s0)
    out["latency_ms"] = {"decide_mean": round(1000 * float(np.mean(t_dec)), 2), "decide_p95": round(1000 * float(np.percentile(t_dec, 95)), 2),
                         "explain_mean": round(1000 * float(np.mean(t_exp)), 2), "n_decide": len(t_dec), "n_explain": len(t_exp)}
    (ROOT / "reports" / "extra_eval.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
