from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="ANMELDELSE ANALYSE", layout="centered")
st.title("Analyser dine anmeldelser")
st.write("Lim inn inn din anmeldelse – én anmeldelse per linje:)")

#Her laster vi inn modellen som train.py har lagret og opptrent
model_path = Path(__file__).resolve().parent / "model.joblib"
if not model_path.exists():
    st.error("Modellen feiler.")
    st.stop()
model = joblib.load(model_path)

# Tekstfeltet for "reviews"
with st.form("tilbakemeldingskjema"):
    text = st.text_area("Kundeanmeldelser", height=180, max_chars=20000)
    submitted = st.form_submit_button("Analyser")

#Hvis sendt inn, ønsker vi at det er regler på hva som sendes inn
if submitted:
    # Ingen tomme lijer.
    reviews = [line.strip() for line in text.splitlines() if line.strip()]
#Man må ha noe i feltet
    if not reviews:
        st.warning("Lim inn minst en anmeldelse her.")
    else:
        # Normalisering / rensing.
        cleaned = [" ".join(review.lower().split()) for review in reviews]
        predictions = model.predict(cleaned)

        # Utskrift av modellens vurdering.
        results = pd.DataFrame({
            "Anmeldelse": reviews,
            "Modellens vurdering": [
                "Positiv" if value == 1 else "Negativ"
                for value in predictions
            ],
        })
        st.dataframe(results, hide_index=True)
        st.write(
            f"Positive: {sum(predictions == 1)} | "
            f"Negative: {sum(predictions == 0)}"
        )

st.caption(
    "Modellen kan ta feil. Den er trent på engelske produktanmeldelser innenfor ladere og mobiltilbehør anmeldelser "
    "og skiller bare mellom positiv og negativ."
)