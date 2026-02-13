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
                # Feature extraction (MATCHING TRAIN_MODEL.PY)
                features = {}
                q_lower = query.lower()
                
                # 1. Basic Length Metrics
                words = q_lower.split()
                features['word_count'] = len(words)
                features['char_count'] = len(q_lower)
                # Calculate avg_word_len safely
                if len(words) > 0:
                    features['avg_word_len'] = sum(len(w) for w in words) / len(words)
                else:
                    features['avg_word_len'] = 0
                
                features['is_long_tail'] = 1 if len(words) > 4 else 0
                
                # 2. Question Types
                question_words = {
                    'jak': 'how', 
                    'gdzie': 'where', 
                    'kiedy': 'when', 
                    'dlaczego': 'why', 
                    'co': 'what', 
                    'ile': 'how_much', 
                    'kto': 'who', 
                    'czy': 'is_it',
                    'jaki': 'which'
                }
                is_any_question = 0
                for pl, en in question_words.items():
                    val = 1 if q_lower.startswith(pl + ' ') or q_lower == pl else 0
                    features[f'q_{en}'] = val
                    if val: is_any_question = 1
                features['is_question'] = is_any_question

                # 3. Informational Intent
                info_words = ['znaczenie', 'definicja', 'powody', 'objawy', 'zasady', 'historia', 'opis', 'przepis', 'poradnik', 'instrukcja']
                features['intent_info'] = sum(1 for w in info_words if w in q_lower)

                # 4. Transactional Intent
                comm_words = ['cena', 'koszt', 'tanio', 'sklep', 'gdzie kupić', 'opinie', 'ranking', 'najlepszy', 'promocja', 'wyprzedaż']
                features['intent_transactional'] = sum(1 for w in comm_words if w in q_lower)
                
                # 5. Entities
                import re
                features['has_number'] = 1 if re.search(r'\d', q_lower) else 0
                features['has_year'] = 1 if re.search(r'20\d{2}', q_lower) else 0
                features['has_step_words'] = 1 if re.search(r'krok po kroku|jak zrobić|jak naprawić', q_lower) else 0
                
                # Create DataFrame
                input_df = pd.DataFrame([features])
                
                # Ensure all columns from training exist (fill missing with 0)
                for col in feature_names:
                    if col not in input_df.columns:
                        input_df[col] = 0
                
                # Reorder columns to match training
                input_df = input_df[feature_names]
                
                # Predict
                try:
                    proba_array = model.predict_proba(input_df)
                    
                    # Handle cases where model only learned one class (e.g., all data was 0 or all was 1)
                    if proba_array.shape[1] == 2:
                        proba = proba_array[0][1]
                    else:
                        # Only one class present
                        learned_class = model.classes_[0]
                        if learned_class == 1:
                            proba = 1.0
                        else:
                            proba = 0.0
                        
                        st.warning(f"⚠️ Uwaga: Model został wytrenowany na danych zawierających tylko jedną klasę ({learned_class}). Wynik zawsze będzie taki sam.")

                    prediction = int(proba * 100)
                except Exception as e:
                    st.error(f"Błąd predykcji: {e}")
                    prediction = 0
                
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
