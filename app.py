import os
import re
import pandas as pd
import pymorphy2
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import google.generativeai as genai
import inspect
import logging

# Flask тиркемеси үчүн логгерди конфигурациялоо
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# pymorphy2 анализаторун инициализациялоо (кыргыз тили үчүн)
try:
    morph_ky = pymorphy2.MorphAnalyzer(lang='ky')
    logger.info("Pymorphy2 кыргыз тили үчүн ийгиликтүү жүктөлдү.")
except Exception as e:
    logger.error(
        f"Pymorphy2 кыргыз тили үчүн анализаторду жүктөөдө ката: {e}. 'pymorphy2-dicts-ky' орнотулганын текшериңиз.")
    morph_ky = None

# inspect.getargspec эскиргендиктен, аны алмаштыруу (эгер кээ бир эски китепканалар үчүн керек болсо)
def _getargspec(func):
    full = inspect.getfullargspec(func)
    return full.args, full.varargs, full.varkw, full.defaults

inspect.getargspec = _getargspec

# Негизги жолдор
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
DATA_FILE = os.path.join(DATA_DIR, 'complaints_ky.csv')
MODEL_URGENCY_PATH = os.path.join(BASE_DIR, 'model_urgency.pkl')
MODEL_SPECIALIST_PATH = os.path.join(BASE_DIR, 'model_specialist.pkl')

# --- Gemini API Конфигурациясы ---
# ЭСКЕРТҮҮ: API ачкычын эч качан коддо сактабаңыз! Чөйрө өзгөрмөсүнөн окуңуз.
# Мисалы: os.environ.get('GOOGLE_API_KEY')
# Колдонуу үчүн 'your_actual_api_key_here' ордуна өзүңүздүн API ачкычыңызды коюңуз
# же аны чөйрө өзгөрмөсү катары орнотуңуз.
# Бул мисал үчүн убактылуу катуу коддолгон. Чыныгы долбоордо муну жасабаңыз!
# GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "your_actual_api_key_here") # Чөйрө өзгөрмөсүнөн алуу
GOOGLE_API_KEY = "AIzaSyCbdK1_JKiqgW_SHXzk01Ho4ldrHYIaidk" # Бул мисал үчүн.

if not GOOGLE_API_KEY or GOOGLE_API_KEY == "your_actual_api_key_here":
    logger.warning("GOOGLE_API_KEY белгиленген эмес же демейки. Gemini функциялары иштебейт.")
    gemini_model = None
else:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        gemini_model = genai.GenerativeModel('gemini-1.5-flash-latest')
        # Тесттик чакыруу:
        gemini_model.generate_content("Салам Gemini!")
        logger.info("Google Gemini API ийгиликтүү конфигурацияланды.")
    except Exception as e:
        logger.error(f"Google Gemini API конфигурациялоодо ката: {e}")
        gemini_model = None


# --- Gemini менен иштөө үчүн жардамчы функциялар ---
def generate_gemini_content(prompt_text, safety_settings=None):
    if not gemini_model:
        logger.warning("Gemini модели конфигурацияланган эмес. Генерация мүмкүн эмес.")
        return None
    try:
        if safety_settings is None:
            safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            ]

        response = gemini_model.generate_content(prompt_text, safety_settings=safety_settings)

        if not response.candidates:
            logger.warning(f"Gemini жооп кайтарган жок, себеби: {response.prompt_feedback}")
            if response.prompt_feedback and response.prompt_feedback.block_reason:
                return f"Gemini'ден жооп алуу мүмкүн болбоду. Себеби: Бөгөттөлдү ({response.prompt_feedback.block_reason_message or response.prompt_feedback.block_reason})."
            return "Gemini'ден жооп алуу мүмкүн болбоду. Белгисиз себеп."

        return response.text.strip()
    except Exception as e:
        logger.error(f"Gemini менен контент генерациялоодо ката: {e}", exc_info=True)
        if hasattr(e, 'response') and hasattr(e.response, 'prompt_feedback'):
            logger.error(f"Gemini катасынын чоо-жайы: {e.response.prompt_feedback}")
        elif hasattr(e, 'message'):
            logger.error(f"Gemini катасы: {e.message}")
        return None


def enhance_complaint_with_gemini(text_ky):
    """ Gemini аркылуу арыздын текстин жакшыртуу (орфография, грамматика). """
    if not gemini_model or not text_ky or not text_ky.strip():
        return text_ky

    prompt = f"""Төмөнкү медициналык арызды кыргыз тилинде жакшыртып, грамматикалык жана орфографиялык каталарын оңдоп бер.
Негизги маанисин өзгөртпө. Ашыкча сөз кошпо. Текстти кыска жана так кыл.
Арыз: "{text_ky}"
Оңдолгон арыз:"""
    enhanced_text = generate_gemini_content(prompt)
    if 'Gemini\'ден жооп алуу мүмкүн болбоду' in (enhanced_text or ''): # Проблема болсо, оригиналын кайтарабыз
        logger.warning(f"Gemini арызды жакшырта алган жок. Оригинал текст колдонулду: {text_ky}")
        return text_ky
    return enhanced_text if enhanced_text else text_ky


# --- Негизги функциялар ---
def preprocess_text(text, lemmatize=True):
    if not isinstance(text, str):
        return ""
    text_processed = text.lower()
    text_processed = re.sub(r'[^\w\s]', '', text_processed)
    text_processed = re.sub(r'\s+', ' ', text_processed).strip()

    if lemmatize and morph_ky:
        words = text_processed.split()
        lemmatized_words = [morph_ky.parse(word)[0].normal_form for word in words]
        text_processed = " ".join(lemmatized_words)
    return text_processed


def train_models():
    if not os.path.exists(DATA_FILE):
        logger.error(f"Маалымат файлы '{DATA_FILE}' табылган жок. Үйрөтүүнү өткөрүп жибериңиз же файлды түзүңүз.")
        return None, None

    logger.info("Маалыматтар жүктөлүүдө...")
    try:
        df = pd.read_csv(DATA_FILE, sep=';', encoding='utf-8')
    except Exception as e:
        logger.error(f"CSV файлын '{DATA_FILE}' окууда ката: {e}")
        return None, None

    required_columns = ['жалоба', 'категория_срочности', 'направление_врача']
    if not all(col in df.columns for col in required_columns):
        logger.error(f"Ката: CSV файлында керектүү тилкелер жок. Керектүүлөр: {', '.join(required_columns)}")
        return None, None

    df.dropna(subset=required_columns, inplace=True)
    df['processed_complaint'] = df['жалоба'].apply(preprocess_text)
    df = df[df['processed_complaint'].str.strip() != '']

    if df.empty:
        logger.warning("Алдын ала иштетүүдөн жана чыпкалоодон кийин үйрөтүү үчүн маалымат жок.")
        return None, None

    X = df['processed_complaint']
    y_urgency = df['категория_срочности']
    y_specialist = df['направление_врача']

    # --- Шашылыш категориясы моделин үйрөтүү ---
    logger.info("Шашылыш категориясын аныктоо үчүн модель үйрөтүлүүдө...")
    pipeline_urgency = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=2500, ngram_range=(1, 3), min_df=2)),
        ('clf', LogisticRegression(solver='liblinear', random_state=42, class_weight='balanced', C=1.0, max_iter=1000))
    ])

    test_split_possible = False
    if len(X) >= 10: # Minimum data for a 80/20 split (10*0.2=2)
        # Check if all urgency classes have at least 2 members for stratification
        urgency_value_counts = y_urgency.value_counts()
        if (urgency_value_counts >= 2).all() and len(urgency_value_counts) > 1:
            test_split_possible = True

    if test_split_possible:
        X_train_u, X_test_u, y_train_u, y_test_u = train_test_split(X, y_urgency, test_size=0.2, random_state=42, stratify=y_urgency)
        pipeline_urgency.fit(X_train_u, y_train_u)
        logger.info("Шашылыш категориясы боюнча классификация отчету (тесттик топтомдо):")
        logger.info(f"\n{classification_report(y_test_u, pipeline_urgency.predict(X_test_u), zero_division=0)}")
    else:
        pipeline_urgency.fit(X, y_urgency)
        logger.warning(
            "Маалымат аз болгондуктан же айрым шашылыш категория классында жетиштүү мүчө жок болгондуктан, модель бардык маалыматта үйрөтүлдү же стратификациялоосуз.")

    joblib.dump(pipeline_urgency, MODEL_URGENCY_PATH)
    logger.info(f"Шашылыш категориясы модели сакталды: {MODEL_URGENCY_PATH}")

    # --- Адисти аныктоо үчүн модель үйрөтүү ---
    logger.info("Адисти аныктоо үчүн модель үйрөтүлүүдө...")
    pipeline_specialist = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=2500, ngram_range=(1, 3), min_df=2)),
        ('clf', LogisticRegression(solver='liblinear', random_state=42, class_weight='balanced', C=1.0, max_iter=1000))
    ])

    test_split_possible = False
    if len(X) >= 10:
        # Check if all specialist classes have at least 2 members for stratification
        specialist_value_counts = y_specialist.value_counts()
        if (specialist_value_counts >= 2).all() and len(specialist_value_counts) > 1:
            test_split_possible = True

    if test_split_possible:
        X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X, y_specialist, test_size=0.2, random_state=42, stratify=y_specialist)
        pipeline_specialist.fit(X_train_s, y_train_s)
        logger.info("Адис боюнча классификация отчету (тесттик топтомдо):")
        logger.info(f"\n{classification_report(y_test_s, pipeline_specialist.predict(X_test_s), zero_division=0)}")
    else:
        pipeline_specialist.fit(X, y_specialist)
        logger.warning(
            "Маалымат аз болгондуктан же айрым адис классында жетиштүү мүчө жок болгондуктан, модель бардык маалыматта үйрөтүлдү же стратификациялоосуз.")

    joblib.dump(pipeline_specialist, MODEL_SPECIALIST_PATH)
    logger.info(f"Адис модели сакталды: {MODEL_SPECIALIST_PATH}")

    return pipeline_urgency, pipeline_specialist


def load_models_from_disk():
    loaded_model_urgency = None
    loaded_model_specialist = None
    if os.path.exists(MODEL_URGENCY_PATH):
        try:
            loaded_model_urgency = joblib.load(MODEL_URGENCY_PATH)
            logger.info(f"Шашылыш модели '{MODEL_URGENCY_PATH}' жүктөлдү.")
        except Exception as e:
            logger.error(f"Шашылыш моделин '{MODEL_URGENCY_PATH}' жүктөөдө ката: {e}")
    else:
        logger.warning(f"Шашылыш модели файлы '{MODEL_URGENCY_PATH}' табылган жок.")

    if os.path.exists(MODEL_SPECIALIST_PATH):
        try:
            loaded_model_specialist = joblib.load(MODEL_SPECIALIST_PATH)
            logger.info(f"Адис модели '{MODEL_SPECIALIST_PATH}' жүктөлдү.")
        except Exception as e:
            logger.error(f"Адис моделин '{MODEL_SPECIALIST_PATH}' жүктөөдө ката: {e}")
    else:
        logger.warning(f"Адис модели файлы '{MODEL_SPECIALIST_PATH}' табылган жок.")
    return loaded_model_urgency, loaded_model_specialist


# --- Flask Тиркемеси ---
app = Flask(__name__, static_folder=os.path.join(BASE_DIR, 'static'),
            template_folder=os.path.join(BASE_DIR, 'templates'))
CORS(app)

# Моделдерди жүктөө же үйрөтүү
active_model_urgency, active_model_specialist = None, None # Initially set to None

def ensure_data_file_exists():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(DATA_FILE):
        logger.info(
            f"Маалымат файлы '{DATA_FILE}' жок болгондуктан, бош файл түзүлүүдө. Үйрөтүү үчүн маалымат менен толтуруңуз.")
        with open(DATA_FILE, 'w', encoding='utf-8') as f:
            f.write("жалоба;категория_срочности;направление_врача\n")
            # Кыргызча мисалдарды көбөйтүү:
            f.write("башым ооруп жатат;Желтый;Терапевт\n")
            f.write("катуу жөтөл бар;Зеленый;Терапевт\n")
            f.write("колум сынды;Красный;Хирург/Травматолог\n")
            f.write("тамак ооруп жатат;Зеленый;ЛОР\n")
            f.write("жүрөгүм ооруп жатат;Оранжевый;Кардиолог\n")
            f.write("курсагым катуу ооруп жатат;Оранжевый;Хирург\n")
            f.write("башым айланып, кускум келип жатат;Желтый;Невролог\n")
            f.write("дене табым көтөрүлдү;Желтый;Терапевт\n")


# Initial model loading and training
ensure_data_file_exists() # Make sure the data file exists BEFORE trying to load/train
active_model_urgency, active_model_specialist = load_models_from_disk()
if active_model_urgency is None or active_model_specialist is None:
    logger.warning("Бир же эки модель тең жүктөлгөн жок. Моделдерди үйрөтүү аракети көрүлүүдө...")
    trained_urgency_model, trained_specialist_model = train_models()
    if trained_urgency_model and trained_specialist_model:
        active_model_urgency, active_model_specialist = trained_urgency_model, trained_specialist_model
        logger.info("Моделдер ийгиликтүү үйрөтүлүп, активдештирилди.")
    else:
        logger.error("Моделдерди үйрөтүү мүмкүн болбоду. API туура эмес иштеши мүмкүн. Логдорду текшериңиз.")


@app.route('/')
def index_page():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    global active_model_urgency, active_model_specialist
    if active_model_urgency is None or active_model_specialist is None:
        return jsonify({
            'error_ky': 'Моделдер үйрөтүлгөн эмес же жүктөлгөн эмес. Сураныч, /train эндпоинтин колдонуп көрүңүз же сервер логдорун текшериңиз.'
        }), 500

    try:
        data = request.get_json()
        if not data or 'complaint' not in data:
            return jsonify({'error_ky': 'JSON суроо-талабында "complaint" талаасы жок'}), 400

        complaint_text_original_ky = data['complaint']
        if not complaint_text_original_ky or not isinstance(complaint_text_original_ky,
                                                            str) or not complaint_text_original_ky.strip():
            return jsonify({'error_ky': '"complaint" талаасы бош болбошу керек жана сап түрүндө болушу керек'}), 400

        # 1. Gemini менен арыз текстин жакшыртуу
        complaint_text_enhanced_ky = enhance_complaint_with_gemini(complaint_text_original_ky)
        if complaint_text_enhanced_ky == complaint_text_original_ky:
            complaint_enhanced_status = "Жакшыртуу колдонулган жок же өзгөргөн жок."
        else:
            complaint_enhanced_status = complaint_text_enhanced_ky


        # 2. Текстти алдын ала иштетүү
        processed_complaint_ky = preprocess_text(complaint_text_enhanced_ky)

        if not processed_complaint_ky:
            return jsonify({
                'complaint_original_ky': complaint_text_original_ky,
                'complaint_enhanced_ky': complaint_enhanced_status,
                'predicted_urgency_ky': 'Аныкталган жок',
                'predicted_specialist_ky': 'Аныкталган жок',
                'error_message_ky': 'Арыздан маанилүү маалымат алуу мүмкүн болбоду. Сураныч, кененирээк сүрөттөп бериңиз.'
            }), 200

        # 3. Моделдер менен божомолдоо
        predicted_urgency_ky = active_model_urgency.predict([processed_complaint_ky])[0]
        proba_urgency_list = active_model_urgency.predict_proba([processed_complaint_ky])[0]

        predicted_specialist_ky = active_model_specialist.predict([processed_complaint_ky])[0]
        proba_specialist_list = active_model_specialist.predict_proba([processed_complaint_ky])[0]

        urgency_confidence_ky = {str(cls): float(prob) for cls, prob in
                                 zip(active_model_urgency.classes_, proba_urgency_list)}
        specialist_confidence_ky = {str(cls): float(prob) for cls, prob in
                                    zip(active_model_specialist.classes_, proba_specialist_list)}

        # 4. Gemini менен жоопту байытуу
        gemini_explanation_ky = "Түшүндүрмө түзүлгөн жок."
        gemini_follow_up_questions_ky = []
        gemini_general_advice_ky = "Дарыгерге кайрылыңыз."

        if gemini_model:
            prompt_for_enrichment = f"""Пациенттин арызы (жакшыртылган): "{complaint_text_enhanced_ky}"
Оригинал арыз: "{complaint_text_original_ky}"
Система төмөнкүдөй божомолдоду:
- Шашылыш категориясы: {predicted_urgency_ky} (Ишенимдүүлүк: {urgency_confidence_ky.get(predicted_urgency_ky, 0):.2f})
- Сунушталган адис: {predicted_specialist_ky} (Ишенимдүүлүк: {specialist_confidence_ky.get(predicted_specialist_ky, 0):.2f})

Кыргыз тилинде төмөнкүлөрдү даярда:
1.  **ТҮШҮНДҮРМӨ:** Эмне үчүн система ушундай божомол чыгарганын пациентке кыскача (1-2 сүйлөм) түшүндүрүп бер. Арыздагы негизги белгилерге таян.
2.  **ТАКТООЧУ СУРООЛОР:** Дарыгерге же системага кошумча маалымат берүү үчүн пациентке 1-2 тактоочу суроо бер. (Мисалы: "Бул белгилер качан башталды?", "Оорунун күчү кандай?")
3.  **ЖАЛПЫ КЕҢЕШ:** Пациентке кыскача жалпы медициналык эмес кеңеш бер (мисалы, "Өз алдынча дарыланбаңыз", "Абалыңыз начарласа, дароо дарыгерге кайрылыңыз").

Жоопту төмөнкүдөй форматта бер, ар бир бөлүктү так белгилеп:
ТҮШҮНДҮРМӨ: [бул жерге түшүндүрмө]
ТАКТООЧУ СУРООЛОР:
- [биринчи суроо]
- [экинчи суроо]
ЖАЛПЫ КЕҢЕШ: [бул жерге кеңеш]
"""
            enriched_response_text = generate_gemini_content(prompt_for_enrichment)
            if enriched_response_text:
                explanation_match = re.search(
                    r"ТҮШҮНДҮРМӨ:\s*(.+?)(?=(\n\s*ТАКТООЧУ СУРООЛОР:|\n\s*ЖАЛПЫ КЕҢЕШ:|$))",
                    enriched_response_text,
                    re.DOTALL | re.IGNORECASE
                )
                questions_match = re.search(
                    r"ТАКТООЧУ СУРООЛОР:\s*(.+?)(?=(\n\s*ЖАЛПЫ КЕҢЕШ:|$))",
                    enriched_response_text,
                    re.DOTALL | re.IGNORECASE
                )
                advice_match = re.search(
                    r"ЖАЛПЫ КЕҢЕШ:\s*(.+)",
                    enriched_response_text,
                    re.DOTALL | re.IGNORECASE
                )

                if explanation_match:
                    gemini_explanation_ky = explanation_match.group(1).strip()
                else:
                    logger.warning(
                        f"Gemini'ден 'ТҮШҮНДҮРМӨ' бөлүгүн алуу мүмкүн болбоду. Жооп: {enriched_response_text[:200]}...")

                if questions_match:
                    questions_block = questions_match.group(1).strip()
                    gemini_follow_up_questions_ky = [q.strip().lstrip('-').strip() for q in questions_block.split('\n')
                                                     if q.strip()]
                else:
                    logger.warning(
                        f"Gemini'ден 'ТАКТООЧУ СУРООЛОР' бөлүгүн алуу мүмкүн болбоду. Жооп: {enriched_response_text[:200]}...")

                if advice_match:
                    gemini_general_advice_ky = advice_match.group(1).strip()
                else:
                    logger.warning(
                        f"Gemini'ден 'ЖАЛПЫ КЕҢЕШ' бөлүгүн алуу мүмкүн болбоду. Жооп: {enriched_response_text[:200]}...")

        print(f"Original Complaint: {complaint_text_original_ky}")
        print(f"Enhanced Complaint: {complaint_enhanced_status}")
        print(f"Processed Complaint: {processed_complaint_ky}")
        print(f"Predicted Urgency: {predicted_urgency_ky}")
        print(f"Urgency Confidence: {urgency_confidence_ky}")
        print(f"Predicted Specialist: {predicted_specialist_ky}")
        print(f"Specialist Confidence: {specialist_confidence_ky}")
        print(f"Gemini Explanation: {gemini_explanation_ky}")
        print(f"Gemini Follow-up Questions: {gemini_follow_up_questions_ky}")
        print(f"Gemini General Advice: {gemini_general_advice_ky}")

        return jsonify({
            'complaint_original_ky': complaint_text_original_ky,
            'complaint_enhanced_ky': complaint_enhanced_status,
            'complaint_processed_for_model_ky': processed_complaint_ky,
            'predicted_urgency_ky': predicted_urgency_ky,
            'urgency_confidence_ky': urgency_confidence_ky,
            'predicted_specialist_ky': predicted_specialist_ky,
            'specialist_confidence_ky': specialist_confidence_ky,
            'gemini_explanation_ky': gemini_explanation_ky,
            'gemini_follow_up_questions_ky': gemini_follow_up_questions_ky,
            'gemini_general_advice_ky': gemini_general_advice_ky
        })

    except Exception as e:
        logger.error(f"/predict эндпоинтинде ката: {e}", exc_info=True)
        return jsonify({'error_ky': 'Суроо-талапты иштетүүдө сервердин ички катасы.'}), 500


@app.route('/train', methods=['GET', 'POST'])
def retrain_models_endpoint():
    global active_model_urgency, active_model_specialist
    logger.info("Моделдерди кайра үйрөтүү суроо-талабы...")
    ensure_data_file_exists()

    trained_urgency, trained_specialist = train_models()
    if trained_urgency and trained_specialist:
        active_model_urgency, active_model_specialist = trained_urgency, trained_specialist
        return jsonify({'message_ky': 'Моделдер ийгиликтүү кайра үйрөтүлүп, сакталды.'})
    else:
        return jsonify({
                           'error_ky': 'Моделдерди кайра үйрөтүү учурунда ката кетти. Маалымат файлын жана сервер логдорун текшериңиз.'}), 500


@app.route('/augment_data', methods=['POST'])
def augment_data_endpoint():
    if not gemini_model:
        return jsonify({'error_ky': 'Gemini модели конфигурацияланган эмес. Маалыматты толуктоо мүмкүн эмес.'}), 503

    try:
        data = request.get_json()
        num_examples = data.get('num_examples', 5)
        target_urgency = data.get('target_urgency')
        target_specialist = data.get('target_specialist')

        if not isinstance(num_examples, int) or not (1 <= num_examples <= 50):
            return jsonify({'error_ky': 'num_examples 1ден 50гө чейинки сан болушу керек.'}), 400

        logger.info(f"{num_examples} жаңы арыз генерацияланууда. Максат: Урматтуулук: {target_urgency or 'кайсыл болсо'} Адис: {target_specialist or 'кайсыл болсо'}")

        prompt_lines = [
            f"Кыргыз тилинде {num_examples} жаңы медициналык арыздын мисалын CSV форматында түзүп бер.",
            "Ар бир сап 'жалоба;категория_срочности;направление_врача' форматында болсун.",
            "Жообуңда CSV баштарын (header) кошпо. Жана эч кандай кошумча текст, киргизүү же түшүндүрмө кошпо, бир гана CSV саптарын бер.",
            "Арыздар ар түрдүү жана реалдуу симптомдорду камтысын.",
            "Эгер максаттуу категория же адис көрсөтүлсө, арыздын мазмуну аларга дал келиши керек."
        ]
        if target_urgency:
            prompt_lines.append(f"Бардык мисалдар үчүн шашылыш категориясы '{target_urgency}' болсун.")
        if target_specialist:
            prompt_lines.append(f"Бардык мисалдар үчүн дарыгерге багыт '{target_specialist}' болсун.")

        # Мурунку маалыматтардан алынган уникалдуу категориялар жана дарыгерлердин тизмесин кошуу (Gemini'ге контекст үчүн)
        if os.path.exists(DATA_FILE):
            try:
                df_existing = pd.read_csv(DATA_FILE, sep=';', encoding='utf-8')
                if 'категория_срочности' in df_existing.columns and not target_urgency:
                    unique_urgencies = df_existing['категория_срочности'].dropna().unique().tolist()
                    if unique_urgencies:
                        prompt_lines.append(f"Мүмкүн болгон шашылыш категориялары: {', '.join(unique_urgencies)}.")
                if 'направление_врача' in df_existing.columns and not target_specialist:
                    unique_specialists = df_existing['направление_врача'].dropna().unique().tolist()
                    if unique_specialists:
                        prompt_lines.append(f"Мүмкүн болгон дарыгерге багыттар: {', '.join(unique_specialists)}.")
            except Exception as e:
                logger.warning(f"Маалыматтарды толуктоо үчүн учурдагы категорияларды/адистерди окууда ката: {e}")

        generation_prompt = "\n".join(prompt_lines)
        logger.debug(f"Gemini'ге генерация үчүн промпт: {generation_prompt}")

        generated_csv_text = generate_gemini_content(generation_prompt)

        if not generated_csv_text or "Gemini'ден жооп алуу мүмкүн болбоду" in generated_csv_text:
            logger.error(f"Gemini маалымат генерациялай алган жок. Жооп: {generated_csv_text}")
            return jsonify(
                {'error_ky': f'Gemini жаңы маалыматтарды генерациялай алган жок. Жооп: {generated_csv_text}'}), 500

        new_entries = []
        for line in generated_csv_text.strip().split('\n'):
            parts = line.strip().split(';')
            if len(parts) == 3:
                complaint, urgency, specialist = parts[0].strip(), parts[1].strip(), parts[2].strip()
                if complaint and urgency and specialist:
                    new_entries.append(f"{complaint};{urgency};{specialist}\n")
            else:
                logger.warning(f"Gemini генерациялаган туура эмес форматтагы сап: '{line}'")

        if not new_entries:
            return jsonify({'message_ky': 'Жаңы маалыматтар генерацияланган жок же формат туура эмес.'}), 200

        ensure_data_file_exists()
        with open(DATA_FILE, 'a', encoding='utf-8') as f:
            for entry in new_entries:
                f.write(entry)

        return jsonify({
            'message_ky': f'{len(new_entries)} жаңы жазуу "{DATA_FILE}" файлына ийгиликтүү кошулду.',
            'added_entries_count': len(new_entries),
            'first_few_entries_added': new_entries[:min(3, len(new_entries))]
        }), 201

    except Exception as e:
        logger.error(f"/augment_data эндпоинтинде ката: {e}", exc_info=True)
        return jsonify({'error_ky': 'Маалыматты толуктоо учурунда сервердин ички катасы.'}), 500


if __name__ == '__main__':
    # Initial setup for model loading/training is now handled globaly
    # This block ensures all global setup runs when the script is executed
    app.run(debug=True, host='0.0.0.0', port=5000)
