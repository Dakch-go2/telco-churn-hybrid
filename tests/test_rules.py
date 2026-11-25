from src import rules as R


def test_knowledge_base_has_15_rules_in_three_groups(base):
    assert len(R.RULES) >= 15
    assert {r.group for r in R.RULES} == {"HARD", "VALID", "POST"}
    assert len({r.id for r in R.RULES}) == len(R.RULES)


def test_valid_client_passes_validation(base):
    assert R.apply_validation(base) == []


def test_negative_tenure_caught(base):
    base["tenure"] = -3
    assert "R1" in [r.id for r in R.apply_validation(base)]


def test_service_inconsistency_caught(base):
    base["InternetService"] = "No"      # бірақ StreamingTV = Yes
    assert "R4" in [r.id for r in R.apply_validation(base)]


def test_new_customer_hard_rule(base):
    base["tenure"] = 0
    assert R.apply_hard(base).id == "R6"


def test_two_year_locked_in(base):
    base.update(Contract="Two year", tenure=10)
    assert R.apply_hard(base).id == "R7"


def test_post_rules_only_for_high_risk(base):
    assert R.apply_post(base, False) == []
    ids = [r.id for r in R.apply_post(base, True)]
    assert "R9" in ids and "R10" in ids and "R12" in ids
