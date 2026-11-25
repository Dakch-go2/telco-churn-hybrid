"""Білім базасы: 15 «ЕГЕР ... ОНДА ...» ережесі (Telco Churn, 5-нұсқа).

Үш топ:
  HARD  - қатаң шектеулер (модельге жібермей шешім қабылдайды)
  VALID - деректерді тексеру ережелері (мүмкін емес мәндерді ұстайды)
  POST  - кейінгі ережелер (модель нәтижесінен кейін ұсыныс береді)
"""
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

# Бизнес-параметрлер (кесте мен есепте негізделген)
AVG_RETENTION_COST = 50.0    # ұстап қалу шарасының орташа құны ($)
CUSTOMER_LIFETIME_MONTHS = 12  # абонент кетпесе, қосымша табыс алынатын мерзім

INTERNET_SERVICES = {"DSL", "Fiber optic", "No"}
CONTRACTS = {"Month-to-month", "One year", "Two year"}
YESNO = {"Yes", "No"}


@dataclass
class Rule:
    id: str
    group: str          # HARD | VALID | POST
    text: str           # ЕГЕР ... ОНДА ...
    source: str
    purpose: str
    check: Callable[[dict, Optional[float]], bool]
    action: str         # ереже іске қосылғанда қайтарылатын нәтиже/ұсыныс


def _g(c, k, d=None):
    return c.get(k, d)


RULES: List[Rule] = [
    # ---------------- (б) Деректерді тексеру ----------------
    Rule("R1", "VALID",
         "ЕГЕР tenure < 0 НЕМЕСЕ tenure > 120 ОНДА деректер қате",
         "Деректер сапасы: мерзім айлармен өлшенеді; 120 ай (10 жыл) жорамал жоғарғы шек, жиындағы максимум — 72 ай",
         "Мүмкін емес абоненттік мерзімді ұстау",
         lambda c, p: not (0 <= _g(c, "tenure", 0) <= 120),
         "INVALID"),
    Rule("R2", "VALID",
         "ЕГЕР MonthlyCharges <= 0 НЕМЕСЕ MonthlyCharges > 500 ОНДА деректер қате",
         "Деректер сапасы: айлық төлем оң болуы керек; жиындағы диапазон $18,25–$118,75, жоғарғы шек $500 — жорамал",
         "Мүмкін емес айлық төлемді ұстау",
         lambda c, p: not (0 < _g(c, "MonthlyCharges", 1) <= 500),
         "INVALID"),
    Rule("R3", "VALID",
         "ЕГЕР TotalCharges < 0 НЕМЕСЕ TotalCharges > MonthlyCharges * (tenure + 1) * 1.5 ОНДА деректер қате",
         "Бухгалтерлік логика: жалпы төлем шамамен айлық төлем × ай санынан аспауы керек (1,5 коэффициент — қор)",
         "Төлемдер арасындағы қайшылықты ұстау",
         lambda c, p: (_g(c, "TotalCharges") is not None and (
             _g(c, "TotalCharges") < 0 or
             _g(c, "TotalCharges") > _g(c, "MonthlyCharges", 0) * (_g(c, "tenure", 0) + 1) * 1.5)),
         "INVALID"),
    Rule("R4", "VALID",
         "ЕГЕР InternetService = 'No' БІРАҚ OnlineSecurity/TechSupport/StreamingTV = 'Yes' ОНДА деректер қате",
         "Қызмет логикасы (деректер құрылымынан): интернетсіз интернет-қосымша қызметтер болмайды",
         "Қызметтер арасындағы қайшылықты ұстау",
         lambda c, p: _g(c, "InternetService") == "No" and any(
             _g(c, k) == "Yes" for k in ("OnlineSecurity", "OnlineBackup", "DeviceProtection",
                                         "TechSupport", "StreamingTV", "StreamingMovies")),
         "INVALID"),
    Rule("R5", "VALID",
         "ЕГЕР PhoneService = 'No' БІРАҚ MultipleLines = 'Yes' ОНДА деректер қате",
         "Қызмет логикасы (деректер құрылымынан): телефонсыз бірнеше желі болмайды",
         "Қызметтер арасындағы қайшылықты ұстау",
         lambda c, p: _g(c, "PhoneService") == "No" and _g(c, "MultipleLines") == "Yes",
         "INVALID"),
    # ---------------- (а) Қатаң шектеулер ----------------
    Rule("R6", "HARD",
         "ЕГЕР абонент жаңа (tenure = 0) ОНДА модельсіз 'жаңа абонент, бағалау мүмкін емес' шешімі",
         "Сараптамалық ереже: tenure = 0 абонентте мінез-құлық тарихы жоқ (жиында 11 жазба, TotalCharges бос)",
         "Тарихсыз абонентке сенімсіз болжам бермеу",
         lambda c, p: _g(c, "tenure", 1) == 0,
         "NEW_CUSTOMER"),
    Rule("R7", "HARD",
         "ЕГЕР Contract = 'Two year' ЖӘНЕ tenure < 24 ОНДА абонент келісімшартпен байланған: кету қаупі төмен (қол жетімді емес)",
         "EDA: 2 жылдық келісімшартта кету үлесі 2,8%; сараптамалық болжам — мерзімінен бұрын бұзуға айыппұл бар",
         "Байланған абонентке ұстап қалу бюджетін жұмсамау",
         lambda c, p: _g(c, "Contract") == "Two year" and _g(c, "tenure", 99) < 24,
         "LOCKED_IN"),
    Rule("R8", "HARD",
         "ЕГЕР MonthlyCharges < 20 ЖӘНЕ InternetService = 'No' ОНДА шамалы пайда: ұстап қалу шарасына бюджет бөлінбейді",
         "Сараптамалық бизнес-болжам: $20-дан төмен төлемде $50 шара экономикалық тиімсіз",
         "Ұстап қалу бюджетін тиімді жұмсау",
         lambda c, p: _g(c, "MonthlyCharges", 99) < 20 and _g(c, "InternetService") == "No",
         "LOW_VALUE"),
    # ---------------- (в) Кейінгі ережелер ----------------
    Rule("R9", "POST",
         "ЕГЕР қауіп жоғары ЖӘНЕ Contract = 'Month-to-month' ОНДА 1-2 жылдық келісімшартқа өтуге жеңілдік ұсыну",
         "EDA: ай сайынғы келісімшартта кету үлесі 42,7% (1 жылдықта 11,3%, 2 жылдықта 2,8%)",
         "Келісімшарт арқылы абонентті ұстап қалу",
         lambda c, p: bool(p) and _g(c, "Contract") == "Month-to-month",
         "Ұсыныс: 1 немесе 2 жылдық келісімшартқа өткенде 15% жеңілдік"),
    Rule("R10", "POST",
         "ЕГЕР қауіп жоғары ЖӘНЕ InternetService = 'Fiber optic' ОНДА тарифті оңтайландыру немесе 2 ай жеңілдік ұсыну",
         "EDA: Fiber optic абоненттерінде кету үлесі 41,9% (DSL — 19,0%)",
         "Баға сезімтал абонентті ұстау",
         lambda c, p: bool(p) and _g(c, "InternetService") == "Fiber optic",
         "Ұсыныс: тиімдірек тариф ауыстыру немесе 2 ай жеңілдік"),
    Rule("R11", "POST",
         "ЕГЕР қауіп жоғары ЖӘНЕ TechSupport = 'No' (интернеті бар) ОНДА тегін техқолдау қызметін 3 айға ұсыну",
         "Модель: TechSupport — маңызды белгілердің бірі (permutation importance); ұсыныс сараптамалық",
         "Қызмет сапасымен қанағаттануды арттыру",
         lambda c, p: bool(p) and _g(c, "InternetService") != "No" and _g(c, "TechSupport") == "No",
         "Ұсыныс: тегін техқолдау 3 ай"),
    Rule("R12", "POST",
         "ЕГЕР қауіп жоғары ЖӘНЕ PaymentMethod = 'Electronic check' ОНДА автотөлемге өтуге бонус ұсыну",
         "EDA: Electronic check төлеушілерде кету үлесі 45,3% (басқа әдістерде 15–19%)",
         "Төлем достығын арттыру",
         lambda c, p: bool(p) and _g(c, "PaymentMethod") == "Electronic check",
         "Ұсыныс: автотөлемге өткенде $5 бонус"),
    Rule("R13", "POST",
         "ЕГЕР қауіп жоғары ЖӘНЕ tenure < 12 ОНДА жаңа абонентке 'қош келдіңіз' байланыс бағдарламасы",
         "EDA: кеткендердің медиана мерзімі 10 ай, қалғандардікі 38 ай",
         "Ерте кезеңде абонентті ұстау",
         lambda c, p: bool(p) and 0 < _g(c, "tenure", 99) < 12,
         "Ұсыныс: менеджердің жеке қоңырауы және алғашқы жылдың бонус пакеті"),
    Rule("R14", "POST",
         "ЕГЕР қауіп жоғары ЖӘНЕ tenure >= 48 ОНДА 'адалдық' (loyalty) бонусын ұсыну",
         "Сараптамалық болжам (CRM тәжірибесі): адал абонентті сақтау; деректермен тексерілмеген",
         "Адал абонентті құрметтеу",
         lambda c, p: bool(p) and _g(c, "tenure", 0) >= 48,
         "Ұсыныс: адалдық бонусы (1 ай тегін)"),
    Rule("R15", "POST",
         "ЕГЕР қауіп жоғары ЖӘНЕ MonthlyCharges * 12 < AVG_RETENTION_COST ОНДА ұстап қалу экономикалық тиімсіз",
         "Сараптамалық бизнес-болжам: жылдық табыс шара құнынан ($50) аз болса, шара залалды",
         "Залалды ұстап қалу шарасын болдырмау",
         lambda c, p: bool(p) and _g(c, "MonthlyCharges", 0) * 12 < AVG_RETENTION_COST,
         "Ескерту: шара экономикалық тиімсіз, ұсынылмайды"),
]

BY_ID: Dict[str, Rule] = {r.id: r for r in RULES}


def apply_validation(client: dict) -> List[Rule]:
    """(б) Деректерді тексеру ережелерін іске қосады."""
    return [r for r in RULES if r.group == "VALID" and r.check(client, None)]


def apply_hard(client: dict) -> Optional[Rule]:
    """(а) Қатаң шектеулер. Алғашқы іске қосылғанын қайтарады."""
    for r in RULES:
        if r.group == "HARD" and r.check(client, None):
            return r
    return None


def apply_post(client: dict, high_risk: bool) -> List[Rule]:
    """(в) Кейінгі ережелер. high_risk = модель ықтималдығы шекті мәннен жоғары."""
    return [r for r in RULES if r.group == "POST" and r.check(client, high_risk)]


def rules_table() -> List[dict]:
    return [{"id": r.id, "group": r.group, "rule": r.text, "source": r.source, "purpose": r.purpose}
            for r in RULES]
