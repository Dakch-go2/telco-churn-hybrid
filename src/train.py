"""EDA + алдын ала өңдеу + модельдерді салыстыру + бағалау + сақтау.
Іске қосу:  python -m src.train
"""
import json, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, roc_auc_score, confusion_matrix, precision_score,
                             recall_score, f1_score, precision_recall_curve)
from sklearn.model_selection import (train_test_split, StratifiedKFold, GridSearchCV, cross_val_predict)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
(ROOT / "models").mkdir(exist_ok=True)
SEED = 42

# Бизнес-шығын моделі (болжам, README/есепте негізделген)
COST_FN = 200.0   # кетіп қалған абонент: ұстап қалуға болатын табыстан айырылу ($)
COST_FP = 50.0    # кетпейтін абонентке жұмсалған ұстап қалу шарасы ($)
UNCERTAIN_BAND = 0.08  # шектен ± осы аралықта "маманның тексеруі қажет"

NUM = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
CAT = ["gender", "Partner", "Dependents", "PhoneService", "MultipleLines", "InternetService",
       "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
       "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod"]
FEATURES = NUM + CAT

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.grid": True, "grid.alpha": .3})


def load():
    df = pd.read_csv(ROOT / "data" / "Telco-Customer-Churn.csv")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].replace(" ", np.nan), errors="coerce")
    df["y"] = (df["Churn"] == "Yes").astype(int)
    return df


def build_pre():
    num = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
    cat = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                    ("oh", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", num, NUM), ("cat", cat, CAT)])


def expected_cost(y, pred):
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return fp * COST_FP + fn * COST_FN


def eda(df):
    stats = {"shape": list(df.shape), "churn_rate": round(df["y"].mean(), 4),
             "missing": {k: int(v) for k, v in df[FEATURES].isna().sum().items() if v > 0},
             "dtypes": {k: str(v) for k, v in df[FEATURES].dtypes.items()}}
    # 1. класс таралуы
    fig, ax = plt.subplots(figsize=(5, 4))
    vc = df["Churn"].value_counts()
    ax.bar(["Қалды (No)", "Кетті (Yes)"], [vc["No"], vc["Yes"]], color=["#4c78a8", "#e45756"])
    for i, v in enumerate([vc["No"], vc["Yes"]]):
        ax.text(i, v + 50, f"{v} ({v/len(df):.1%})", ha="center")
    ax.set_title("Кластардың таралуы"); ax.set_ylabel("Абонент саны")
    fig.tight_layout(); fig.savefig(FIG / "eda1_class.png", dpi=150); plt.close(fig)
    # 2. келісімшарт бойынша кету үлесі
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for ax, col, t in zip(axs, ["Contract", "InternetService"], ["Келісімшарт", "Интернет қызметі"]):
        r = df.groupby(col)["y"].mean().sort_values(ascending=False) * 100
        ax.bar(r.index, r.values, color="#f58518")
        for i, v in enumerate(r.values): ax.text(i, v + .7, f"{v:.1f}%", ha="center")
        ax.set_title(f"{t} бойынша кету үлесі"); ax.set_ylabel("Кету үлесі, %")
    fig.tight_layout(); fig.savefig(FIG / "eda2_contract_internet.png", dpi=150); plt.close(fig)
    # 3. tenure таралуы
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df.loc[df.y == 0, "tenure"], bins=36, alpha=.6, label="Қалды", color="#4c78a8")
    ax.hist(df.loc[df.y == 1, "tenure"], bins=36, alpha=.6, label="Кетті", color="#e45756")
    ax.set_title("Абоненттік мерзім (tenure, ай) таралуы"); ax.set_xlabel("Ай"); ax.legend()
    fig.tight_layout(); fig.savefig(FIG / "eda3_tenure.png", dpi=150); plt.close(fig)
    # 4. MonthlyCharges boxplot
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot([df.loc[df.y == 0, "MonthlyCharges"], df.loc[df.y == 1, "MonthlyCharges"]],
               tick_labels=["Қалды", "Кетті"], patch_artist=True)
    ax.set_title("Айлық төлем және кету"); ax.set_ylabel("MonthlyCharges, $")
    fig.tight_layout(); fig.savefig(FIG / "eda4_monthly_box.png", dpi=150); plt.close(fig)
    # 5. корреляция
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    cm = df[["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen", "y"]].corr()
    im = ax.imshow(cm, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(5)); ax.set_xticklabels(cm.columns, rotation=45, ha="right")
    ax.set_yticks(range(5)); ax.set_yticklabels(cm.columns); ax.grid(False)
    for i in range(5):
        for j in range(5): ax.text(j, i, f"{cm.iloc[i, j]:.2f}", ha="center", va="center", fontsize=9)
    ax.set_title("Сандық белгілердің корреляциясы"); fig.colorbar(im)
    fig.tight_layout(); fig.savefig(FIG / "eda5_corr.png", dpi=150); plt.close(fig)
    # 6. төлем әдісі
    fig, ax = plt.subplots(figsize=(8, 4))
    r = df.groupby("PaymentMethod")["y"].mean().sort_values() * 100
    ax.barh(r.index, r.values, color="#54a24b")
    for i, v in enumerate(r.values): ax.text(v + .3, i, f"{v:.1f}%", va="center")
    ax.set_title("Төлем әдісі бойынша кету үлесі"); ax.set_xlabel("%")
    fig.tight_layout(); fig.savefig(FIG / "eda6_payment.png", dpi=150); plt.close(fig)
    # сандық корреляциялар
    stats["corr_with_churn"] = {k: round(float(v), 3) for k, v in cm["y"].drop("y").items()}
    stats["churn_by_contract"] = {k: round(float(v), 4) for k, v in df.groupby("Contract")["y"].mean().items()}
    stats["churn_by_internet"] = {k: round(float(v), 4) for k, v in df.groupby("InternetService")["y"].mean().items()}
    stats["churn_by_payment"] = {k: round(float(v), 4) for k, v in df.groupby("PaymentMethod")["y"].mean().items()}
    stats["tenure_median"] = {"stayed": float(df.loc[df.y == 0, "tenure"].median()),
                              "churned": float(df.loc[df.y == 1, "tenure"].median())}
    stats["monthly_median"] = {"stayed": float(df.loc[df.y == 0, "MonthlyCharges"].median()),
                               "churned": float(df.loc[df.y == 1, "MonthlyCharges"].median())}
    stats["tenure0"] = int((df.tenure == 0).sum())
    return stats


def main():
    df = load()
    stats = eda(df)
    X, y = df[FEATURES], df["y"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)

    candidates = {
        "Logistic Regression": (LogisticRegression(max_iter=2000, class_weight="balanced"),
                                {"clf__C": [0.01, 0.1, 1, 10]}),
        "Random Forest": (RandomForestClassifier(random_state=SEED, class_weight="balanced_subsample", n_jobs=-1),
                          {"clf__n_estimators": [200, 400], "clf__max_depth": [6, 10, None],
                           "clf__min_samples_leaf": [3, 10]}),
        "Gradient Boosting": (HistGradientBoostingClassifier(random_state=SEED, class_weight="balanced"),
                              {"clf__learning_rate": [0.03, 0.1], "clf__max_depth": [3, 5],
                               "clf__max_iter": [100, 250]}),
    }
    results, fitted = [], {}
    for name, (clf, grid) in candidates.items():
        pipe = Pipeline([("pre", build_pre()), ("clf", clf)])
        gs = GridSearchCV(pipe, grid, scoring="average_precision", cv=cv, n_jobs=-1)
        gs.fit(Xtr, ytr)
        fitted[name] = gs
        p = gs.predict_proba(Xte)[:, 1]
        results.append({"model": name, "best_params": {k.replace("clf__", ""): v for k, v in gs.best_params_.items()},
                        "cv_pr_auc": round(gs.best_score_, 4),
                        "test_pr_auc": round(average_precision_score(yte, p), 4),
                        "test_roc_auc": round(roc_auc_score(yte, p), 4)})
        print(results[-1])

    best_name = max(results, key=lambda r: r["cv_pr_auc"])["model"]
    best = fitted[best_name].best_estimator_
    # Шекті мәнді train бойынша out-of-fold болжамдарынан, шығынды минимумдау арқылы таңдау
    oof = cross_val_predict(best, Xtr, ytr, cv=cv, method="predict_proba")[:, 1]
    ths = np.round(np.arange(0.05, 0.96, 0.01), 2)
    costs = [expected_cost(ytr, (oof >= t).astype(int)) for t in ths]
    thr = float(ths[int(np.argmin(costs))])
    f1s = [f1_score(ytr, (oof >= t).astype(int)) for t in ths]
    thr_f1 = float(ths[int(np.argmax(f1s))])

    # Итоговая оценка на отложенной выборке
    best.fit(Xtr, ytr)
    pte = best.predict_proba(Xte)[:, 1]

    def metrics_at(t):
        pr = (pte >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(yte, pr).ravel()
        return {"threshold": t, "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
                "precision": round(precision_score(yte, pr), 4), "recall": round(recall_score(yte, pr), 4),
                "f1": round(f1_score(yte, pr), 4), "cost": float(fp * COST_FP + fn * COST_FN)}
    m_cost, m_05, m_f1 = metrics_at(thr), metrics_at(0.5), metrics_at(thr_f1)
    # Шығынсыз бизнес база: ешкімді ұстамау / барлығын ұстау
    base_none = float((yte == 1).sum() * COST_FN)
    base_all = float((yte == 0).sum() * COST_FP)

    # --- графиктер ---
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ths, costs, color="#e45756"); ax.axvline(thr, ls="--", color="k")
    ax.set_title(f"Күтілетін шығын мен шекті мән (оңтайлы = {thr:.2f})")
    ax.set_xlabel("Шекті мән (threshold)"); ax.set_ylabel("Шығын, $ (train, OOF)")
    fig.tight_layout(); fig.savefig(FIG / "model_threshold_cost.png", dpi=150); plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for ax, m, t in zip(axs, [m_05, m_cost], ["Threshold = 0.50", f"Threshold = {thr:.2f} (шығын бойынша)"]):
        cmx = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        ax.imshow(cmx, cmap="Blues"); ax.grid(False)
        for i in range(2):
            for j in range(2): ax.text(j, i, cmx[i, j], ha="center", va="center", fontsize=14,
                                       color="white" if cmx[i, j] > cmx.max() / 2 else "black")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Қалады", "Кетеді"]); ax.set_yticks([0, 1])
        ax.set_yticklabels(["Қалды", "Кетті"]); ax.set_xlabel("Болжам"); ax.set_ylabel("Шын мәні"); ax.set_title(t)
    fig.suptitle("Қателіктер матрицасы (тест жиыны)"); fig.tight_layout()
    fig.savefig(FIG / "model_confusion.png", dpi=150); plt.close(fig)

    pr_, rc_, _ = precision_recall_curve(yte, pte)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(rc_, pr_, color="#4c78a8"); ax.axhline(yte.mean(), ls=":", color="gray", label="Базалық (кездейсоқ)")
    ax.scatter([m_cost["recall"]], [m_cost["precision"]], color="red", zorder=5, label=f"thr={thr:.2f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.legend(); ax.set_title("Precision–Recall қисығы")
    fig.tight_layout(); fig.savefig(FIG / "model_pr_curve.png", dpi=150); plt.close(fig)

    # Модельдерді салыстыру графигі
    fig, ax = plt.subplots(figsize=(7, 4))
    xs = np.arange(len(results)); w = .35
    ax.bar(xs - w/2, [r["cv_pr_auc"] for r in results], w, label="CV PR-AUC", color="#4c78a8")
    ax.bar(xs + w/2, [r["test_pr_auc"] for r in results], w, label="Тест PR-AUC", color="#f58518")
    ax.set_xticks(xs); ax.set_xticklabels([r["model"] for r in results]); ax.set_ylim(0.4, 0.75); ax.legend()
    ax.set_title("Модельдерді салыстыру (PR-AUC)")
    fig.tight_layout(); fig.savefig(FIG / "model_compare.png", dpi=150); plt.close(fig)

    # --- интерпретация: permutation importance ---
    pi = permutation_importance(best, Xte, yte, scoring="average_precision", n_repeats=10,
                                random_state=SEED, n_jobs=-1)
    imp = pd.Series(pi.importances_mean, index=FEATURES).sort_values(ascending=False)
    imp_std = pd.Series(pi.importances_std, index=FEATURES)[imp.index]
    top = imp.head(10)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(top.index[::-1], top.values[::-1], xerr=imp_std[top.index].values[::-1], color="#54a24b")
    ax.set_title("Белгілердің маңыздылығы (permutation, PR-AUC)"); ax.set_xlabel("PR-AUC кемуі")
    fig.tight_layout(); fig.savefig(FIG / "model_importance.png", dpi=150); plt.close(fig)

    # --- қателерді талдау: ең сенімді қате 5 мысал ---
    err = Xte.copy(); err["y_true"] = yte.values; err["p"] = pte
    err["pred"] = (pte >= thr).astype(int)
    wrong = err[err.pred != err.y_true].copy()
    wrong["conf_err"] = np.abs(wrong["p"] - wrong["y_true"])
    worst = wrong.sort_values("conf_err", ascending=False).head(5)
    worst_list = worst.reset_index().rename(columns={"index": "row"}).to_dict(orient="records")
    err_summary = {
        "n_errors": int(len(wrong)),
        "fn_share": round(float((wrong.y_true == 1).mean()), 3),
        "fn_by_contract": {k: int(v) for k, v in wrong[wrong.y_true == 1]["Contract"].value_counts().items()},
        "fp_by_contract": {k: int(v) for k, v in wrong[wrong.y_true == 0]["Contract"].value_counts().items()},
    }

    # --- сақтау ---
    joblib.dump(best, ROOT / "models" / "model.joblib")
    meta = {"model": best_name, "threshold": thr, "uncertain_band": UNCERTAIN_BAND,
            "cost_fn": COST_FN, "cost_fp": COST_FP, "features": FEATURES, "trained_on": int(len(Xtr)),
            "reference": {c: (float(Xtr[c].median()) if c in NUM else str(Xtr[c].mode()[0])) for c in FEATURES},
            "version": "1.0"}
    (ROOT / "models" / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    out = {"eda": stats, "comparison": results, "best": best_name, "threshold_cost": thr, "threshold_f1": thr_f1,
           "test_at_0.5": m_05, "test_at_cost_thr": m_cost, "test_at_f1_thr": m_f1,
           "baseline_cost_no_action": base_none, "baseline_cost_offer_all": base_all,
           "n_test": int(len(yte)), "test_churn_rate": round(float(yte.mean()), 4),
           "importance": {k: [round(float(imp[k]), 4), round(float(imp_std[k]), 4)] for k in imp.index},
           "worst5": worst_list, "error_summary": err_summary}
    (ROOT / "reports" / "results.json").write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str))
    print(json.dumps({k: out[k] for k in ["best", "threshold_cost", "threshold_f1", "test_at_0.5", "test_at_cost_thr",
                                           "baseline_cost_no_action", "baseline_cost_offer_all"]}, indent=1))


if __name__ == "__main__":
    main()
