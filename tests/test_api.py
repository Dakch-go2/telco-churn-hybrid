def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["model_loaded"] is True


def test_predict_ok(client, base):
    r = client.post("/predict", json=base)
    assert r.status_code == 200
    j = r.json()
    assert {"decision", "probability", "fired_rules", "decision_id"} <= set(j)


def test_explain_has_top_features(client, base):
    r = client.post("/explain", json=base)
    assert r.status_code == 200
    j = r.json()
    assert len(j["top_features"]) > 0 and "fired_rules" in j


def test_negative_missing_field(client, base):
    del base["Contract"]
    r = client.post("/predict", json=base)
    assert r.status_code == 422
    assert r.json()["error"] == "INVALID_INPUT"


def test_negative_wrong_type(client, base):
    base["MonthlyCharges"] = "көп"
    r = client.post("/predict", json=base)
    assert r.status_code == 422


def test_negative_impossible_value(client, base):
    base["tenure"] = -5
    r = client.post("/predict", json=base)
    assert r.status_code == 422
    assert r.json()["detail"]["rules"][0]["id"] == "R1"


def test_negative_unknown_category(client, base):
    base["Contract"] = "Five year"
    assert client.post("/predict", json=base).status_code == 422


def test_negative_inconsistent_services(client, base):
    base["InternetService"] = "No"
    r = client.post("/predict", json=base)
    assert r.status_code == 422


def test_empty_total_charges_ok(client, base):
    base["TotalCharges"] = None
    assert client.post("/predict", json=base).status_code == 200


def test_stats_and_feedback(client, base):
    did = client.post("/predict", json=base).json()["decision_id"]
    assert client.get("/stats").json()["last_n"] >= 1
    assert client.post("/feedback", json={"decision_id": did, "agrees": False, "comment": "келіспеймін"}).status_code == 200
    assert client.post("/feedback", json={"decision_id": 99999, "agrees": True}).status_code == 404
