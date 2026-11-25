# Telco Churn — гибридті интеллектуалды жүйе (ЕГЖ, 5-нұсқа)

Телеком абоненттерінің кетуін болжау: **сараптамалық ережелер (15) + ML (Gradient Boosting) + REST API + Streamlit**.

Деректер: Telco Customer Churn (7043 абонент), `data/Telco-Customer-Churn.csv`.
Қайта жүктеу: https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv
(Kaggle-дағы "Telco Customer Churn" жиынымен бірдей мазмұн.)

## 3 командамен іске қосу
```bash
pip install -r requirements.txt                 # 1) тәуелділіктер
uvicorn api.main:app --port 8000                # 2) API (Swagger: http://127.0.0.1:8000/docs)
streamlit run app/ui.py                         # 3) интерфейс (басқа терминалда)
```
Модель дайын (`models/model.joblib`). Қайта оқыту: `python -m src.train`. Тесттер: `python -m pytest -q`.

## Құрылым
| Жол | Мазмұны |
|---|---|
| `src/rules.py` | Білім базасы: R1–R15 (VALID, HARD, POST) |
| `src/decision.py` | Гибридті шешім: ережелер → ML → кейінгі ережелер, сенімсіздік аймағы |
| `src/train.py` | EDA, Pipeline, 3 модель + GridSearchCV (5-fold), шекті мәнді шығын бойынша таңдау |
| `src/db.py` | SQLite журналы (`logs/decisions.db`) |
| `api/main.py` | FastAPI: `/predict`, `/explain`, `/health`, `/stats`, `/feedback` |
| `app/ui.py` | Streamlit (API арқылы жұмыс істейді) |
| `tests/` | 23 pytest тест (оның ішінде 7 негатив) |
| `notebooks/eda_model.ipynb` | EDA және модельдерді бағалау |
| `src/evaluate_hybrid.py`, `src/extra_eval.py` | Гибридті жүйені және қосымша талдауларды бағалау |
| `report.docx` | Есеп |

## Шешім түрлері
`HIGH_RISK`, `HIGH_RISK_NO_ACTION`, `REVIEW_REQUIRED` (p шектен ±0,08 ішінде), `LOW_RISK`, `NOT_ASSESSED`, `NO_ACTION`; қате кіріс үшін HTTP 422 (`INVALID_INPUT`).

## Бизнес-шығын болжамы
FN (кеткен абонент) = $200, FP (бекер ұсыныс) = $50 — бұл **болжам**; `src/train.py` ішінде `COST_FN`, `COST_FP` арқылы өзгертіледі.
