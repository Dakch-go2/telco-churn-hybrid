from src import decision as D


def test_probability_in_range(base):
    p = D.predict_proba(base)
    assert 0.0 <= p <= 1.0


def test_risky_client_gets_high_risk_and_recommendations(base):
    r = D.decide(base)
    assert r["decision"] in ("HIGH_RISK", "REVIEW_REQUIRED")
    assert r["source"] in ("model+rules",)


def test_loyal_two_year_client_low_risk(base):
    base.update(Contract="Two year", tenure=60, InternetService="DSL", MonthlyCharges=45.0, TotalCharges=2700.0,
                OnlineSecurity="Yes", TechSupport="Yes", OnlineBackup="Yes", DeviceProtection="Yes",
                StreamingTV="No", StreamingMovies="No", PaymentMethod="Credit card (automatic)")
    r = D.decide(base)
    assert r["decision"] == "LOW_RISK"


def test_uncertain_band_returns_review(monkeypatch, base):
    _, meta = D.load_artifacts()
    monkeypatch.setattr(D, "predict_proba", lambda c: meta["threshold"] + 0.01)
    assert D.decide(base)["decision"] == "REVIEW_REQUIRED"


def test_hard_rule_skips_model(monkeypatch, base):
    def boom(c): raise AssertionError("модель шақырылмауы керек")
    monkeypatch.setattr(D, "predict_proba", boom)
    base["tenure"] = 0
    base["TotalCharges"] = None
    r = D.decide(base)
    assert r["decision"] == "NOT_ASSESSED" and r["source"] == "rules:HARD"


def test_missing_total_charges_handled(base):
    base["TotalCharges"] = None
    assert D.decide(base)["decision"] in ("HIGH_RISK", "REVIEW_REQUIRED", "LOW_RISK", "HIGH_RISK_NO_ACTION")
