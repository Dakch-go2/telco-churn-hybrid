"""FastAPI қосымшасы: /predict, /explain, /health, /stats, /feedback."""
import sys
from pathlib import Path
from typing import Literal, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src import db, decision as D, rules as R

YN = Literal["Yes", "No"]

app = FastAPI(title="Telco Churn — гибридті интеллектуалды жүйе", version="1.0",
              description="Сараптамалық ережелер + ML + API. Абоненттің кету қаупін бағалайды.")


class Client(BaseModel):
    gender: Literal["Male", "Female"]
    SeniorCitizen: Literal[0, 1]
    Partner: YN
    Dependents: YN
    tenure: int = Field(..., description="Абонент болған ай саны")
    PhoneService: YN
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YN
    PaymentMethod: Literal["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    MonthlyCharges: float = Field(..., description="Айлық төлем, $")
    TotalCharges: Optional[float] = Field(None, description="Жалпы төлем, $ (бос болса — толтырылады)")

    model_config = {"json_schema_extra": {"example": {
        "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No", "tenure": 5,
        "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic",
        "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No",
        "StreamingTV": "Yes", "StreamingMovies": "Yes", "Contract": "Month-to-month",
        "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
        "MonthlyCharges": 95.5, "TotalCharges": 470.0}}}


class FeedbackIn(BaseModel):
    decision_id: int
    agrees: bool
    comment: Optional[str] = Field(None, max_length=500)


@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    errors = [{"field": ".".join(str(x) for x in e["loc"] if x != "body"), "problem": e["msg"], "type": e["type"]}
              for e in exc.errors()]
    return JSONResponse(status_code=422, content={
        "error": "INVALID_INPUT",
        "message": "Кіріс деректері қате: өрістердің түрін немесе мәнін тексеріңіз.",
        "details": errors})


def _run(endpoint: str, client: Client):
    data = client.model_dump()
    result = D.decide(data)
    rid = db.log_decision(endpoint, data, result)
    result["decision_id"] = rid
    if result["decision"] == "INVALID_INPUT":
        raise HTTPException(status_code=422, detail={"error": "INVALID_INPUT", "message": result["message"],
                                                     "rules": result["fired_rules"], "decision_id": rid})
    return data, result


@app.get("/health")
def health():
    try:
        _, meta = D.load_artifacts()
        return {"status": "ok", "model_loaded": True, "model": meta["model"], "version": meta["version"],
                "threshold": meta["threshold"], "rules_count": len(R.RULES)}
    except Exception as e:  # pragma: no cover
        return JSONResponse(status_code=503, content={"status": "error", "model_loaded": False, "message": str(e)})


@app.post("/predict")
def predict(client: Client):
    _, result = _run("/predict", client)
    return result


@app.post("/explain")
def explain(client: Client):
    data, result = _run("/explain", client)
    result["top_features"] = D.local_explanation(data) if result["probability"] is not None else []
    return result


@app.get("/stats")
def stats(n: int = 10):
    return db.stats(max(1, min(n, 1000)))


@app.post("/feedback")
def feedback(fb: FeedbackIn):
    fid = db.save_feedback(fb.decision_id, fb.agrees, fb.comment)
    if fid is None:
        raise HTTPException(status_code=404, detail={"error": "NOT_FOUND", "message": "Мұндай шешім табылмады"})
    return {"status": "saved", "feedback_id": fid}
