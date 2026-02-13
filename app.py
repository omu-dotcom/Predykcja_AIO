import streamlit as st
import sys
import traceback

# Page config must be the first Streamlit command
st.set_page_config(
    page_title="Predykcja Google AI Overview",
    page_icon="🤖",
    layout="centered"
)

def main():
    import pandas as pd
    import pickle
    import os
    import sklearn

    # Load model
    @st.cache_resource
    def load_model():
        if not os.path.exists('aio_model.pkl'):
            st.error("⚠️ Brak pliku 'aio_model.pkl'.")
            return None
        
        try:
            with open('aio_model.pkl', 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            st.error(f"❌ Błąd wczytywania modelu: {e}")
            st.write(f"Python ver: {sys.version}")
            st.write(f"Sklearn ver: {sklearn.__version__}")
            return None

    model_data = load_model()

    # Header
    st.title("🤖 Predykcja wystąpienia Google AI Overview")
    st.markdown("""
    Aplikacja oblicza prawdopodobieństwo pojawienia się **AI Overview (SGE)** w wynikach Google dla podanej frazy.
    Model został wytrenowany na Twoich danych historycznych z Senuto.
    """)

    if model_data is None:
        st.warning("⚠️ Model nie został załadowany. Aplikacja działa w trybie demonstracyjnym.")
    else:
        model = model_data['model']
        feature_names = model_data['feature_names']

        # Input
        st.markdown("### 👇 Wpisz frazę do sprawdzenia")
        query = st.text_input("Fraza kluczowa", placeholder="np. jak zamontować panele podłogowe")

        if st.button("Oblicz prawdopodobieństwo", type="primary"):
            if not query:
                st.warning("Wpisz frazę!")
            else:
                # Feature extraction (SAME LOGIC AS TRAINING)
                features = {}
                
                # Linguistic stats
                features['word_count'] = len(query.split())
                features['char_count'] = len(query)
                
                # Question words
                question_words = ['jak', 'gdzie', 'kiedy', 'dlaczego', 'co', 'ile', 'kto', 'czy']
                for qw in question_words:
                    features[f'is_{qw}'] = 1 if query.lower().startswith(qw + ' ') or query.lower() == qw else 0
                    
                # Intent words
                intent_words = ['cena', 'opinia', 'ranking', 'najlepszy', 'tani', 'sklep', 'kup']
                for iw in intent_words:
                    features[f'has_{iw}'] = 1 if iw in query.lower() else 0
                    
                # Create DataFrame with correct column order
                input_df = pd.DataFrame([features])
                
                # Ensure all columns from training exist (fill missing with 0)
                for col in feature_names:
                    if col not in input_df.columns:
                        input_df[col] = 0
                
                # Reorder columns to match training
                input_df = input_df[feature_names]
                
                # Predict
                proba = model.predict_proba(input_df)[0][1] # Probability of class 1 (AIO present)
                prediction = int(proba * 100)
                
                # Display result
                st.divider()
                st.markdown(f"### Wynik dla frazy: _„{query}”_")
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.metric("Prawdopodobieństwo", f"{prediction}%")
                
                with col2:
                    # Progress bar custom color
                    if prediction < 30:
                        bar_color = "red"
                        msg = "Mała szansa na AIO."
                    elif prediction < 70:
                        bar_color = "yellow"
                        msg = "Umiarkowana szansa. Zależy od branży."
                    else:
                        bar_color = "green"
                        msg = "🔥 Wysokie ryzyko AI Overview! Warto optymalizować pod AIO."
                    
                    st.progress(prediction / 100)
                    st.caption(msg)
                    
                # Feature explanation (simplified)
                with st.expander("Dlaczego taki wynik? (Cechy frazy)"):
                    st.write(input_df.T.rename(columns={0: 'Wartość'}))

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error("Wystąpił krytyczny błąd podczas uruchamiania aplikacji:")
        st.code(traceback.format_exc())
