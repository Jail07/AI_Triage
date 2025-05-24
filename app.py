import os
import re
import pandas as pd
import nltk
# import pymorphy2
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

import inspect

def _getargspec(func):
    full = inspect.getfullargspec(func)
    return full.args, full.varargs, full.varkw, full.defaults
inspect.getargspec = _getargspec

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
DATA_FILE = os.path.join(DATA_DIR, 'complaints_ky.csv')
MODEL_URGENCY_PATH = os.path.join(BASE_DIR, 'model_urgency.pkl')
MODEL_SPECIALIST_PATH = os.path.join(BASE_DIR, 'model_specialist.pkl')

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def train_models():
    if not os.path.exists(DATA_FILE):
        print(f"Файл данных {DATA_FILE} не найден. Пропустите обучение или создайте файл.")
        return None, None

    print("Загрузка данных...")
    try:
        df = pd.read_csv(DATA_FILE, sep=';', encoding='utf-8')
    except Exception as e:
        print(f"Ошибка при чтении CSV файла: {e}")
        return None, None

    required_columns = ['жалоба', 'категория_срочности', 'направление_врача']
    if not all(col in df.columns for col in required_columns):
        print(f"Ошибка: в CSV файле отсутствуют необходимые колонки. Требуются: {', '.join(required_columns)}")
        return None, None

    df.dropna(subset=required_columns, inplace=True)
    df['processed_complaint'] = df['жалоба'].apply(preprocess_text)

    df = df[df['processed_complaint'].str.strip() != '']

    if df.empty:
        print("Нет данных для обучения после предобработки и фильтрации.")
        return None, None

    X = df['processed_complaint']
    y_urgency = df['категория_срочности']
    y_specialist = df['направление_врача']

    stratify_urgency = y_urgency if len(y_urgency.unique()) > 1 else None
    stratify_specialist = y_specialist if len(y_specialist.unique()) > 1 else None

    print("Обучение модели для определения срочности...")
    pipeline_urgency = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=1500, ngram_range=(1, 2))),  # ngram_range может улучшить
        ('clf', LogisticRegression(solver='liblinear', random_state=42, class_weight='balanced', C=1.0))
    ])

    if len(X) > 50:
        X_train_u, X_test_u, y_train_u, y_test_u = train_test_split(X, y_urgency, test_size=0.2, random_state=42, stratify=stratify_urgency)
        pipeline_urgency.fit(X_train_u, y_train_u)
        print("Отчет по классификации срочности (на тестовой выборке):")
        print(classification_report(y_test_u, pipeline_urgency.predict(X_test_u), zero_division=0))
    else:
        pipeline_urgency.fit(X, y_urgency)

    pipeline_urgency.fit(X, y_urgency)
    joblib.dump(pipeline_urgency, MODEL_URGENCY_PATH)
    print(f"Модель для срочности сохранена в {MODEL_URGENCY_PATH}")

    print("Обучение модели для определения специалиста...")
    pipeline_specialist = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=1500, ngram_range=(1, 2))),
        ('clf', LogisticRegression(solver='liblinear', random_state=42, class_weight='balanced', C=1.0))
    ])

    if len(X) > 50:
        X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X, y_specialist, test_size=0.2, random_state=42, stratify=stratify_specialist)
        pipeline_specialist.fit(X_train_s, y_train_s)
        print("Отчет по классификации специалиста (на тестовой выборке):")
        print(classification_report(y_test_s, pipeline_specialist.predict(X_test_s), zero_division=0))
    else:
        pipeline_specialist.fit(X, y_specialist)

    pipeline_specialist.fit(X, y_specialist)
    joblib.dump(pipeline_specialist, MODEL_SPECIALIST_PATH)
    print(f"Модель для специалиста сохранена в {MODEL_SPECIALIST_PATH}")

    return pipeline_urgency, pipeline_specialist


def load_models():
    model_urgency = None
    model_specialist = None
    if os.path.exists(MODEL_URGENCY_PATH):
        try:
            model_urgency = joblib.load(MODEL_URGENCY_PATH)
            print(f"Модель срочности загружена из {MODEL_URGENCY_PATH}")
        except Exception as e:
            print(f"Ошибка загрузки модели срочности: {e}")
    else:
        print(f"Файл модели срочности {MODEL_URGENCY_PATH} не найден.")

    if os.path.exists(MODEL_SPECIALIST_PATH):
        try:
            model_specialist = joblib.load(MODEL_SPECIALIST_PATH)
            print(f"Модель специалиста загружена из {MODEL_SPECIALIST_PATH}")
        except Exception as e:
            print(f"Ошибка загрузки модели специалиста: {e}")
    else:
        print(f"Файл модели специалиста {MODEL_SPECIALIST_PATH} не найден.")
    return model_urgency, model_specialist


app = Flask(__name__, static_folder=os.path.join(BASE_DIR, 'static'),
            template_folder=os.path.join(BASE_DIR, 'templates'))
CORS(app)

model_urgency, model_specialist = load_models()


def ensure_data_file_exists():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(DATA_FILE):
        print(f"Создаю пустой файл {DATA_FILE}, так как он отсутствует. Заполните его данными для обучения.")
        with open(DATA_FILE, 'w', encoding='utf-8') as f:
            f.write("жалоба;категория_срочности;направление_врача\n")
            f.write("тестовая жалоба для запуска;Зеленый;Терапевт\n")


if model_urgency is None or model_specialist is None:
    print("Одна или обе модели не загружены. Попытка обучить модели...")
    ensure_data_file_exists()
    trained_urgency, trained_specialist = train_models()
    if trained_urgency and trained_specialist:
        model_urgency, model_specialist = trained_urgency, trained_specialist
    else:
        print("Не удалось обучить модели. API может не работать корректно.")


@app.route('/')
def index_page():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    global model_urgency, model_specialist
    if model_urgency is None or model_specialist is None:
        return jsonify(
            {'error': 'Модели не обучены или не загружены. Попробуйте /train или проверьте логи сервера.'}), 500

    try:
        data = request.get_json()
        if not data or 'complaint' not in data:
            return jsonify({'error': 'Отсутствует поле "complaint" в JSON запросе'}), 400

        complaint_text = data['complaint']
        if not complaint_text or not isinstance(complaint_text, str) or not complaint_text.strip():
            return jsonify({'error': 'Поле "complaint" не должно быть пустым и должно быть непустой строкой'}), 400

        processed_complaint = preprocess_text(complaint_text)
        if not processed_complaint:
            return jsonify({
                'complaint': complaint_text,
                'processed_complaint': processed_complaint,
                'predicted_urgency': 'Не определено',
                'predicted_specialist': 'Не определено',
                'error_message': 'Не удалось извлечь значимую информацию из жалобы. Пожалуйста, опишите подробнее.'
            }), 200

        predicted_urgency = model_urgency.predict([processed_complaint])[0]
        proba_urgency_list = model_urgency.predict_proba([processed_complaint])[0]

        predicted_specialist = model_specialist.predict([processed_complaint])[0]
        proba_specialist_list = model_specialist.predict_proba([processed_complaint])[0]

        urgency_confidence = {cls: prob for cls, prob in zip(model_urgency.classes_, proba_urgency_list)}
        specialist_confidence = {cls: prob for cls, prob in zip(model_specialist.classes_, proba_specialist_list)}

        return jsonify({
            'complaint_original': complaint_text,
            'complaint_processed': processed_complaint,
            'predicted_urgency': predicted_urgency,
            'predicted_specialist': predicted_specialist,
            'urgency_confidence': {k: float(v) for k, v in urgency_confidence.items()},
            'specialist_confidence': {k: float(v) for k, v in specialist_confidence.items()}
        })

    except Exception as e:
        app.logger.error(f"Ошибка при обработке запроса /predict: {e}", exc_info=True)
        return jsonify({'error': 'Внутренняя ошибка сервера при обработке запроса.'}), 500


@app.route('/train', methods=['GET', 'POST'])
def retrain_models_endpoint():
    global model_urgency, model_specialist
    print("Запрос на переобучение моделей...")
    ensure_data_file_exists()
    trained_urgency, trained_specialist = train_models()
    if trained_urgency and trained_specialist:
        model_urgency, model_specialist = trained_urgency, trained_specialist
        return jsonify({'message': 'Модели успешно переобучены и сохранены.'})
    else:
        return jsonify({'error': 'Ошибка во время переобучения моделей. Проверьте файл данных и логи.'}), 500


if __name__ == '__main__':
    ensure_data_file_exists()
    if model_urgency is None or model_specialist is None:
        print("Модели не были загружены или обучены при старте. Попробуйте запустить /train или проверьте файл данных.")

    app.run(debug=True, host='0.0.0.0', port=5000)