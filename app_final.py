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
        if not os.path.exists('aio_model_FIXED.pkl'):
            st.error("⚠️ Brak pliku 'aio_model_FIXED.pkl'.")
            return None
        
        try:
            with open('aio_model_FIXED.pkl', 'rb') as f:
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
    Aplikacja szacuje prawdopodobieństwo wystąpienia sekcji **Google AI Overview** dla wpisanej frazy kluczowej. 
    Analiza opiera się na zaawansowanym modelu predykcyjnym analizującym intencję i strukturę zapytania.
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
                # Feature extraction (EXACT MATCH WITH TRAIN_MODEL.PY)
                def extract_advanced_features(query):
                    import re
                    q = str(query).lower().strip()
                    words = q.split()
                    word_count = len(words)
                    char_count = len(q)
                    
                    features = {}
                    features['word_count'] = word_count
                    features['char_count'] = char_count
                    features['is_long_tail'] = 1 if word_count >= 3 else 0
                    features['is_very_long'] = 1 if word_count >= 6 else 0
                    
                    # 1. INTENT (User Synthesis #1)
                    # Know / Informational
                    info_patterns = r'\bjak\b|\bco to\b|\bdlaczego\b|\bile\b|\bkiedy\b|\bczy\b|znaczenie|definicja|powody|objawy|zasady|historia|opis|przepis|poradnik|instrukcja'
                    features['intent_info'] = 1 if re.search(info_patterns, q) else 0
                    
                    # Shopping / E-commerce (Anti-AIO)
                    shopping_patterns = r'zabawk|prezent|prezent|cena|koszt|tanio|sklep|kup|zamów|promocja|wyprzedaż|do koszyka|lodówk|pralk|laptop|telefon|ubrani'
                    features['intent_shopping'] = 1 if re.search(shopping_patterns, q) else 0
                    
                    # Transactional (Anti-AIO)
                    trans_patterns = r'cena|koszt|tanio|sklep|kup|zamów|promocja|wyprzedaż|do koszyka'
                    features['intent_trans'] = 1 if re.search(trans_patterns, q) else 0
                    
                    # Local (Anti-AIO)
                    local_cities = r'warszawa|kraków|wrocław|poznań|gdańsk|szczecin|lublin|katowice|restauracja|fryzjer|dentysta|pobliżu'
                    features['intent_local'] = 1 if re.search(local_cities, q) else 0

                    # Navigational (Anti-AIO)
                    brand_patterns = r'zara|allegro|facebook|instagram|youtube|gmail|logowanie|poczta'
                    features['intent_nav'] = 1 if re.search(brand_patterns, q) else 0

                    # 2. FUNNEL STAGE
                    features['stage_tofu'] = 1 if re.search(r'\bjak\b|\bco\b|\bkto\b|\bgdzie\b|\bdlaczego\b|\bczym\b|znaczenie', q) else 0
                    features['stage_mofu'] = 1 if re.search(r'vs|ranking|najlepszy|porównanie|opinie|test', q) else 0

                    # 3. TOPIC CATEGORIES
                    features['cat_medical'] = 1 if re.search(r'leczenie|lek|choroba|badanie|objawy|ból|doktor|szpital', q) else 0
                    features['cat_legal'] = 1 if re.search(r'prawo|przepis|procedura|zus|gov|wniosek|umowa|kara', q) else 0
                    features['cat_diy'] = 1 if re.search(r'zrobić|naprawić|montaż|budowa|remont|ogród|kuchnia', q) else 0
                    features['cat_dictionary'] = 1 if re.search(r'definicja|znaczenie|pojęcie|synonim|angielsku', q) else 0

                    # 4. STRUCTURE
                    question_words = ['jak', 'co', 'dlaczego', 'ile', 'kiedy', 'czy', 'jaki', 'kto']
                    features['starts_with_q'] = 1 if words and words[0] in question_words else 0
                    
                    # 5. SEMANTIC COMPLEXITY
                    features['has_number'] = 1 if re.search(r'\d', q) else 0
                    features['has_step_words'] = 1 if re.search(r'krok po kroku|jak zrobić|jak naprawić', q) else 0

                    return features

                features = extract_advanced_features(query)
                input_df = pd.DataFrame([features])
                for col in feature_names:
                    if col not in input_df.columns:
                        input_df[col] = 0
                input_df = input_df[feature_names]
                
                # Predict
                try:
                    proba_array = model.predict_proba(input_df)
                    proba = proba_array[0][1]
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
                    if prediction < 30:
                        msg = "Mała szansa na AIO. Spróbuj zmienić intencję na informacyjną."
                    elif prediction < 70:
                        msg = "Umiarkowana szansa. Google może, ale nie musi wyświetlić AIO."
                    else:
                        msg = "🔥 Wysokie prawdopodobieństwo! Fraza spełnia kryteria wysokiego pokrycia AI."
                    st.progress(prediction / 100)
                    st.caption(msg)
                    
                # Feature explanation (simplified)
                with st.expander("Analiza cech frazy (Scoring)"):
                    st.write("Model przeanalizował frazę pod kątem następujących kategorii:")
                    f = features
                    groups = {
                        "📁 Struktura": ["word_count", "char_count", "is_long_tail", "starts_with_q"],
                        "🧠 Intencja": ["intent_info", "intent_shopping", "intent_trans", "intent_local", "intent_nav"],
                        "📊 Etap Lejka": ["stage_tofu", "stage_mofu"],
                        "🏗️ Kategoria Tematyczna": ["cat_medical", "cat_legal", "cat_diy", "cat_dictionary"]
                    }
                    for name, cols in groups.items():
                        st.markdown(f"**{name}**")
                        mini_df = pd.DataFrame({c: [f.get(c, 0)] for c in cols}).T.rename(columns={0: 'Wartość'})
                        st.table(mini_df)



if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error("Wystąpił krytyczny błąd podczas uruchamiania aplikacji:")
        st.code(traceback.format_exc())
