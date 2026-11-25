"""Streamlit интерфейсі. Модельге тікелей емес, API арқылы жүгінеді."""
import os
import requests
import streamlit as st

API = os.environ.get("API_URL", "http://127.0.0.1:8000")
st.set_page_config(page_title="Telco Churn", page_icon="📡", layout="wide")
st.title("📡 Абоненттің кету қаупін бағалау")
st.caption("Гибридті жүйе: сараптамалық ережелер + ML + API")

LABELS = {"HIGH_RISK": ("🔴 Қауіп жоғары", "#d62728"), "HIGH_RISK_NO_ACTION": ("🟠 Қауіп жоғары, шара тиімсіз", "#ff7f0e"),
          "REVIEW_REQUIRED": ("🟡 Маманның тексеруі қажет", "#e6b800"), "LOW_RISK": ("🟢 Қауіп төмен", "#2ca02c"),
          "NOT_ASSESSED": ("⚪ Бағаланбады (жаңа абонент)", "#7f7f7f"), "NO_ACTION": ("⚪ Шара қажет емес", "#7f7f7f")}

with st.form("f"):
    c1, c2, c3 = st.columns(3)
    with c1:
        gender = st.selectbox("Жынысы", ["Female", "Male"])
        senior = st.selectbox("Зейнеткер (SeniorCitizen)", [0, 1])
        partner = st.selectbox("Жұбайы бар", ["Yes", "No"])
        dep = st.selectbox("Асырауындағылар", ["Yes", "No"])
        tenure = st.number_input("Абонент мерзімі, ай", value=5, step=1)
        phone = st.selectbox("Телефон қызметі", ["Yes", "No"])
    with c2:
        ml = st.selectbox("Бірнеше желі", ["No", "Yes", "No phone service"])
        inet = st.selectbox("Интернет", ["Fiber optic", "DSL", "No"])
        sec = st.selectbox("Онлайн қауіпсіздік", ["No", "Yes", "No internet service"])
        bk = st.selectbox("Онлайн резерв", ["No", "Yes", "No internet service"])
        dp = st.selectbox("Құрылғы қорғанысы", ["No", "Yes", "No internet service"])
        ts = st.selectbox("Техқолдау", ["No", "Yes", "No internet service"])
    with c3:
        tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        mv = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
        contract = st.selectbox("Келісімшарт", ["Month-to-month", "One year", "Two year"])
        paper = st.selectbox("Қағазсыз шот", ["Yes", "No"])
        pay = st.selectbox("Төлем әдісі", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        mc = st.number_input("Айлық төлем, $", value=95.5)
        tc = st.number_input("Жалпы төлем, $ (0 = белгісіз)", value=470.0)
    go = st.form_submit_button("Бағалау")

if go:
    body = {"gender": gender, "SeniorCitizen": senior, "Partner": partner, "Dependents": dep, "tenure": int(tenure),
            "PhoneService": phone, "MultipleLines": ml, "InternetService": inet, "OnlineSecurity": sec,
            "OnlineBackup": bk, "DeviceProtection": dp, "TechSupport": ts, "StreamingTV": tv, "StreamingMovies": mv,
            "Contract": contract, "PaperlessBilling": paper, "PaymentMethod": pay, "MonthlyCharges": mc,
            "TotalCharges": tc if tc > 0 else None}
    try:
        r = requests.post(f"{API}/explain", json=body, timeout=15)
    except requests.RequestException:
        st.error("API қолжетімсіз. Алдымен API-ды іске қосыңыз (uvicorn api.main:app).")
        st.stop()
    d = r.json()
    if r.status_code != 200:
        det = d.get("detail", d)
        st.error(det.get("message", "Қате"))
        for x in det.get("rules", det.get("details", [])):
            st.write("•", x.get("rule") or f'{x.get("field")}: {x.get("problem")}')
        st.stop()
    label, color = LABELS.get(d["decision"], (d["decision"], "#333"))
    st.markdown(f"<h2 style='color:{color}'>{label}</h2>", unsafe_allow_html=True)
    st.write(d["message"])
    if d["probability"] is not None:
        p = d["probability"]
        st.markdown(f"**Кету ықтималдығы: {p:.1%}** (шекті мән {d['threshold']:.0%})")
        st.progress(min(p, 1.0))
    if d["recommendations"]:
        st.subheader("Ұсыныстар")
        for x in d["recommendations"]:
            st.write("✅", x)
    with st.expander("Түсіндірме: іске қосылған ережелер"):
        for x in d["fired_rules"]:
            st.write(f"**{x['id']}** ({x['group']}): {x['rule']}")
        if not d["fired_rules"]:
            st.write("Ешбір ереже іске қосылмады — шешімді тек модель берді.")
    if d.get("top_features"):
        with st.expander("Түсіндірме: ең әсерлі белгілер", expanded=True):
            for x in d["top_features"]:
                st.write(f"{x['feature']} = {x['value']} → {x['direction']} ({x['effect']:+.3f})")
    st.caption(f"Шешім ID: {d['decision_id']}  |  Дереккөз: {d['source']}")
