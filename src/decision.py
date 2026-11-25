"""Гибридті шешім модулі: алдын ала ережелер -> ML моделі -> кейінгі ережелер."""
import json
from pathlib import Path
from typing import Dict, Optional
import joblib
import pandas as pd

from . import rules as R

ROOT = Path(__file__).resolve().parent.parent
_MODEL = None
_META = None


def load_artifacts():
    global _MODEL, _META
    if _MODEL is None:
        _MODEL = joblib.load(ROOT / "models" / "model.joblib")
        _META = json.loads((ROOT / "models" / "meta.json").read_text())
    return _MODEL, _META


def _fired(rule: R.Rule) -> dict:
    return {"id": rule.id, "group": rule.group, "rule": rule.text, "action": rule.action}


def _row(client: dict, meta: dict) -> pd.DataFrame:
    return pd.DataFrame([{f: client.get(f) for f in meta["features"]}])


def predict_proba(client: dict) -> float:
    model, meta = load_artifacts()
    return float(model.predict_proba(_row(client, meta))[0, 1])


def local_explanation(client: dict, top: int = 5) -> list:
    """Белгіні 'әдеттегі' (референс) мәнге ауыстырғанда ықтималдық қалай өзгеретінін өлшейді."""
    model, meta = load_artifacts()
    base = _row(client, meta)
    p0 = float(model.predict_proba(base)[0, 1])
    out = []
    for f in meta["features"]:
        alt = base.copy()
        alt[f] = meta["reference"][f]
        if alt[f].iloc[0] == base[f].iloc[0]:
            continue
        delta = p0 - float(model.predict_proba(alt)[0, 1])
        out.append({"feature": f, "value": client.get(f), "effect": round(delta, 4),
                    "direction": "қауіпті арттырады" if delta > 0 else "қауіпті азайтады"})
    out.sort(key=lambda d: abs(d["effect"]), reverse=True)
    return out[:top]


def decide(client: dict) -> Dict:
    """Қайтарады: decision, source, probability, fired_rules, recommendations, message."""
    model, meta = load_artifacts()
    thr, band = meta["threshold"], meta["uncertain_band"]
    res = {"decision": None, "source": None, "probability": None, "threshold": thr,
           "fired_rules": [], "recommendations": [], "message": ""}

    # 1) деректерді тексеру (б)
    bad = R.apply_validation(client)
    if bad:
        res.update(decision="INVALID_INPUT", source="rules:VALID", fired_rules=[_fired(r) for r in bad],
                   message="Кіріс деректерінде қайшылық бар: " + "; ".join(r.purpose for r in bad))
        return res

    # 2) қатаң шектеулер (а)
    hard = R.apply_hard(client)
    if hard:
        mapping = {"NEW_CUSTOMER": ("NOT_ASSESSED", "Жаңа абонент: тарихы жоқ, бағалау сенімсіз. Алғашқы айдан кейін қайталаңыз."),
                   "LOCKED_IN": ("LOW_RISK", "Абонент 2 жылдық келісімшартпен байланған: кету қаупі төмен, шара қажет емес."),
                   "LOW_VALUE": ("NO_ACTION", "Табыс шамалы: ұстап қалу шарасы экономикалық тиімсіз.")}
        dec, msg = mapping[hard.action]
        res.update(decision=dec, source="rules:HARD", fired_rules=[_fired(hard)], message=msg)
        return res

    # 3) ML моделі
    p = predict_proba(client)
    res["probability"] = round(p, 4)
    high = p >= thr
    post = R.apply_post(client, high)
    res["fired_rules"] = [_fired(r) for r in post]
    res["recommendations"] = [r.action for r in post if not r.action.startswith("Ескерту")]
    warn = [r.action for r in post if r.action.startswith("Ескерту")]

    # 4) сенімсіздік аймағы
    if abs(p - thr) < band:
        res.update(decision="REVIEW_REQUIRED", source="model+rules",
                   message=f"Модель сенімсіз (p={p:.2f}, шек={thr:.2f}): маманның тексеруі қажет.")
        return res
    if high and warn:
        res.update(decision="HIGH_RISK_NO_ACTION", source="model+rules", recommendations=[],
                   message="Кету қаупі жоғары, бірақ " + warn[0].lower())
    elif high:
        res.update(decision="HIGH_RISK", source="model+rules",
                   message=f"Кету қаупі жоғары (p={p:.2f}). Ұстап қалу шарасы ұсынылады.")
    else:
        res.update(decision="LOW_RISK", source="model", recommendations=[],
                   message=f"Кету қаупі төмен (p={p:.2f}).")
    return res
