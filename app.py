import os
import re
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
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

# --- Setup for NLTK (for English Lemmatization) ---
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/wordnet')
except LookupError:
    print("NLTK data not found. Downloading 'punkt' and 'wordnet'...")
    nltk.download('punkt')
    nltk.download('wordnet')
    print("Download complete.")

# Configure logger for the Flask application
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize the lemmatizer for English
lemmatizer = WordNetLemmatizer()
logger.info("NLTK WordNetLemmatizer for English loaded successfully.")


# Monkey patch for inspect.getargspec if needed by older libraries
def _getargspec(func):
    full = inspect.getfullargspec(func)
    return full.args, full.varargs, full.varkw, full.defaults


inspect.getargspec = _getargspec

# --- Core Paths ---
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
# Using a separate file for English data
DATA_FILE = os.path.join(DATA_DIR, 'complaints_en.csv')
MODEL_URGENCY_PATH = os.path.join(BASE_DIR, 'model_urgency_en.pkl')
MODEL_SPECIALIST_PATH = os.path.join(BASE_DIR, 'model_specialist_en.pkl')

# --- Gemini API Configuration ---
# NOTE: Never hardcode API keys in production! Read from environment variables.
# For example: os.environ.get('GOOGLE_API_KEY')
# For this example, replace 'your_actual_api_key_here' with your key
# or set it as an environment variable.
try:
    genai.configure(api_key="AIzaSyCbdK1_JKiqgW_SHXzk01Ho4ldrHYIaidk")
    gemini_model = genai.GenerativeModel('gemini-1.5-flash-latest')
    # Test call to ensure connectivity
    gemini_model.generate_content("Hello Gemini!")
    logger.info("Google Gemini API configured successfully.")
except Exception as e:
    logger.error(f"Error configuring Google Gemini API: {e}")
    gemini_model = None


# --- Helper functions for Gemini ---
def generate_gemini_content(prompt_text, safety_settings=None):
    """Generates content using the Gemini model with error handling."""
    if not gemini_model:
        logger.warning("Gemini model is not configured. Cannot generate content.")
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
            logger.warning(f"Gemini returned no response, reason: {response.prompt_feedback}")
            if response.prompt_feedback and response.prompt_feedback.block_reason:
                return f"Could not get a response from Gemini. Reason: Blocked ({response.prompt_feedback.block_reason_message or response.prompt_feedback.block_reason})."
            return "Could not get a response from Gemini. Unknown reason."

        return response.text.strip()
    except Exception as e:
        logger.error(f"Error during Gemini content generation: {e}", exc_info=True)
        # Attempt to get more detailed error info if available
        if hasattr(e, 'response') and hasattr(e.response, 'prompt_feedback'):
            logger.error(f"Gemini error details: {e.response.prompt_feedback}")
        elif hasattr(e, 'message'):
            logger.error(f"Gemini error: {e.message}")
        return None


def enhance_complaint_with_gemini(text_en):
    """Enhances the complaint text (spelling, grammar) using Gemini."""
    if not gemini_model or not text_en or not text_en.strip():
        return text_en

    prompt = f"""Please correct and rephrase the following medical complaint in English.
Fix any grammatical or spelling errors. Do not change the core meaning. Keep it concise.
Complaint: "{text_en}"
Corrected Complaint:"""
    enhanced_text = generate_gemini_content(prompt)
    if not enhanced_text or 'Could not get a response from Gemini' in enhanced_text:
        logger.warning(f"Gemini could not enhance the complaint. Using original text: {text_en}")
        return text_en
    return enhanced_text


# --- Core Functions ---
def preprocess_text(text, lemmatize=True):
    """Preprocesses English text: lowercase, remove punctuation, lemmatize."""
    if not isinstance(text, str):
        return ""
    text_processed = text.lower()
    text_processed = re.sub(r'[^\w\s]', '', text_processed)
    text_processed = re.sub(r'\s+', ' ', text_processed).strip()

    if lemmatize:
        words = word_tokenize(text_processed)
        lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
        text_processed = " ".join(lemmatized_words)
    return text_processed


def train_models():
    """Trains and saves the classification models."""
    if not os.path.exists(DATA_FILE):
        logger.error(f"Data file '{DATA_FILE}' not found. Skipping training. Please create the file.")
        return None, None

    logger.info("Loading data...")
    try:
        df = pd.read_csv(DATA_FILE, sep=';', encoding='utf-8')
    except Exception as e:
        logger.error(f"Error reading CSV file '{DATA_FILE}': {e}")
        return None, None

    # Using English column names
    required_columns = ['complaint', 'urgency_category', 'doctor_referral']
    if not all(col in df.columns for col in required_columns):
        logger.error(f"Error: CSV file is missing required columns. Expected: {', '.join(required_columns)}")
        return None, None

    df.dropna(subset=required_columns, inplace=True)
    df['processed_complaint'] = df['complaint'].apply(preprocess_text)
    df = df[df['processed_complaint'].str.strip() != '']

    if df.empty:
        logger.warning("No data available for training after preprocessing and filtering.")
        return None, None

    X = df['processed_complaint']
    y_urgency = df['urgency_category']
    y_specialist = df['doctor_referral']

    # --- Train Urgency Model ---
    logger.info("Training urgency category model...")
    pipeline_urgency = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=2500, ngram_range=(1, 3), min_df=2)),
        ('clf', LogisticRegression(solver='liblinear', random_state=42, class_weight='balanced', C=1.0, max_iter=1000))
    ])

    # Check if a stratified test split is possible
    test_split_possible_u = False
    if len(X) >= 10:
        urgency_counts = y_urgency.value_counts()
        if (urgency_counts >= 2).all() and len(urgency_counts) > 1:
            test_split_possible_u = True

    if test_split_possible_u:
        X_train_u, X_test_u, y_train_u, y_test_u = train_test_split(X, y_urgency, test_size=0.2, random_state=42,
                                                                    stratify=y_urgency)
        pipeline_urgency.fit(X_train_u, y_train_u)
        logger.info("Urgency Classification Report (on test set):")
        logger.info(f"\n{classification_report(y_test_u, pipeline_urgency.predict(X_test_u), zero_division=0)}")
    else:
        pipeline_urgency.fit(X, y_urgency)
        logger.warning(
            "Training on full dataset for urgency model due to small data size or insufficient members in some classes for stratification.")

    joblib.dump(pipeline_urgency, MODEL_URGENCY_PATH)
    logger.info(f"Urgency model saved to: {MODEL_URGENCY_PATH}")

    # --- Train Specialist Model ---
    logger.info("Training specialist referral model...")
    pipeline_specialist = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=2500, ngram_range=(1, 3), min_df=2)),
        ('clf', LogisticRegression(solver='liblinear', random_state=42, class_weight='balanced', C=1.0, max_iter=1000))
    ])

    test_split_possible_s = False
    if len(X) >= 10:
        specialist_counts = y_specialist.value_counts()
        if (specialist_counts >= 2).all() and len(specialist_counts) > 1:
            test_split_possible_s = True

    if test_split_possible_s:
        X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X, y_specialist, test_size=0.2, random_state=42,
                                                                    stratify=y_specialist)
        pipeline_specialist.fit(X_train_s, y_train_s)
        logger.info("Specialist Classification Report (on test set):")
        logger.info(f"\n{classification_report(y_test_s, pipeline_specialist.predict(X_test_s), zero_division=0)}")
    else:
        pipeline_specialist.fit(X, y_specialist)
        logger.warning(
            "Training on full dataset for specialist model due to small data size or insufficient members in some classes for stratification.")

    joblib.dump(pipeline_specialist, MODEL_SPECIALIST_PATH)
    logger.info(f"Specialist model saved to: {MODEL_SPECIALIST_PATH}")

    return pipeline_urgency, pipeline_specialist


def load_models_from_disk():
    """Loads models from disk if they exist."""
    loaded_model_urgency, loaded_model_specialist = None, None
    if os.path.exists(MODEL_URGENCY_PATH):
        try:
            loaded_model_urgency = joblib.load(MODEL_URGENCY_PATH)
            logger.info(f"Urgency model '{MODEL_URGENCY_PATH}' loaded.")
        except Exception as e:
            logger.error(f"Error loading urgency model '{MODEL_URGENCY_PATH}': {e}")
    else:
        logger.warning(f"Urgency model file '{MODEL_URGENCY_PATH}' not found.")

    if os.path.exists(MODEL_SPECIALIST_PATH):
        try:
            loaded_model_specialist = joblib.load(MODEL_SPECIALIST_PATH)
            logger.info(f"Specialist model '{MODEL_SPECIALIST_PATH}' loaded.")
        except Exception as e:
            logger.error(f"Error loading specialist model '{MODEL_SPECIALIST_PATH}': {e}")
    else:
        logger.warning(f"Specialist model file '{MODEL_SPECIALIST_PATH}' not found.")
    return loaded_model_urgency, loaded_model_specialist


# --- Flask Application ---
app = Flask(__name__, static_folder=os.path.join(BASE_DIR, 'static'),
            template_folder=os.path.join(BASE_DIR, 'templates'))
CORS(app)

# Global variables for active models
active_model_urgency, active_model_specialist = None, None


def ensure_data_file_exists():
    """Creates an empty data file with headers if it doesn't exist."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(DATA_FILE):
        logger.info(f"Data file '{DATA_FILE}' does not exist. Creating a new file with sample data.")
        with open(DATA_FILE, 'w', encoding='utf-8') as f:
            f.write("complaint;urgency_category;doctor_referral\n")
            f.write("i have a bad headache;Yellow;General Practitioner\n")
            f.write("severe cough;Green;General Practitioner\n")
            f.write("i broke my arm;Red;Surgeon/Traumatologist\n")
            f.write("sore throat;Green;ENT\n")
            f.write("my heart hurts;Orange;Cardiologist\n")
            f.write("severe stomach ache;Orange;Surgeon\n")
            f.write("feeling dizzy and nauseous;Yellow;Neurologist\n")
            f.write("i have a fever;Yellow;General Practitioner\n")


# --- Initial Setup on App Start ---
ensure_data_file_exists()
active_model_urgency, active_model_specialist = load_models_from_disk()
if active_model_urgency is None or active_model_specialist is None:
    logger.warning("One or both models failed to load. Attempting to train new models...")
    trained_urgency_model, trained_specialist_model = train_models()
    if trained_urgency_model and trained_specialist_model:
        active_model_urgency, active_model_specialist = trained_urgency_model, trained_specialist_model
        logger.info("Models trained and activated successfully.")
    else:
        logger.error("Failed to train models. The API might not function correctly. Check logs.")


@app.route('/')
def index_page():
    # Renders the main HTML page. Assumes you have an index.html in the templates folder.
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handles prediction requests."""
    global active_model_urgency, active_model_specialist
    if active_model_urgency is None or active_model_specialist is None:
        return jsonify({
            'error': 'Models are not trained or loaded. Try using the /train endpoint or check server logs.'
        }), 500

    try:
        data = request.get_json()
        if not data or 'complaint' not in data:
            return jsonify({'error': 'Missing "complaint" field in JSON request'}), 400

        complaint_text_original = data['complaint']
        if not complaint_text_original or not isinstance(complaint_text_original,
                                                         str) or not complaint_text_original.strip():
            return jsonify({'error': '"complaint" field must be a non-empty string'}), 400

        # 1. Enhance complaint text with Gemini
        complaint_text_enhanced = enhance_complaint_with_gemini(complaint_text_original)
        if complaint_text_enhanced == complaint_text_original:
            complaint_enhanced_status = "Enhancement not applied or no changes made."
        else:
            complaint_enhanced_status = complaint_text_enhanced

        # 2. Preprocess the text for the model
        processed_complaint = preprocess_text(complaint_text_enhanced)

        if not processed_complaint:
            return jsonify({
                'complaint_original': complaint_text_original,
                'complaint_enhanced': complaint_enhanced_status,
                'predicted_urgency': 'Undetermined',
                'predicted_specialist': 'Undetermined',
                'error_message': 'Could not extract meaningful information from the complaint. Please describe it in more detail.'
            }), 200

        # 3. Predict with the models
        predicted_urgency = active_model_urgency.predict([processed_complaint])[0]
        proba_urgency_list = active_model_urgency.predict_proba([processed_complaint])[0]

        predicted_specialist = active_model_specialist.predict([processed_complaint])[0]
        proba_specialist_list = active_model_specialist.predict_proba([processed_complaint])[0]

        urgency_confidence = {str(cls): float(prob) for cls, prob in
                              zip(active_model_urgency.classes_, proba_urgency_list)}
        specialist_confidence = {str(cls): float(prob) for cls, prob in
                                 zip(active_model_specialist.classes_, proba_specialist_list)}

        # 4. Enrich the response with Gemini
        gemini_explanation = "Explanation could not be generated."
        gemini_follow_up_questions = []
        gemini_general_advice = "Please consult a doctor."

        if gemini_model:
            prompt_for_enrichment = f"""A patient's complaint (enhanced): "{complaint_text_enhanced}"
Original complaint: "{complaint_text_original}"
The system made the following prediction:
- Urgency Category: {predicted_urgency} (Confidence: {urgency_confidence.get(predicted_urgency, 0):.2f})
- Recommended Specialist: {predicted_specialist} (Confidence: {specialist_confidence.get(predicted_specialist, 0):.2f})

Please prepare the following in English:
1.  **EXPLANATION:** Briefly explain (1-2 sentences) to the patient why the system made this prediction, based on the key symptoms in the complaint.
2.  **FOLLOW-UP QUESTIONS:** Provide 1-2 clarifying questions for the patient to provide more information to a doctor or the system. (e.g., "When did these symptoms start?", "How severe is the pain on a scale of 1 to 10?")
3.  **GENERAL ADVICE:** Give a short, non-medical piece of advice (e.g., "Do not self-medicate," "If your condition worsens, seek immediate medical attention").

Provide the response in the following format, clearly labeling each section:
EXPLANATION: [your explanation here]
FOLLOW-UP QUESTIONS:
- [first question]
- [second question]
GENERAL ADVICE: [your advice here]
"""
            enriched_response_text = generate_gemini_content(prompt_for_enrichment)
            if enriched_response_text:
                explanation_match = re.search(
                    r"EXPLANATION:\s*(.+?)(?=(\n\s*FOLLOW-UP QUESTIONS:|\n\s*GENERAL ADVICE:|$))",
                    enriched_response_text, re.DOTALL | re.IGNORECASE
                )
                questions_match = re.search(
                    r"FOLLOW-UP QUESTIONS:\s*(.+?)(?=(\n\s*GENERAL ADVICE:|$))",
                    enriched_response_text, re.DOTALL | re.IGNORECASE
                )
                advice_match = re.search(
                    r"GENERAL ADVICE:\s*(.+)",
                    enriched_response_text, re.DOTALL | re.IGNORECASE
                )

                if explanation_match:
                    gemini_explanation = explanation_match.group(1).strip()
                if questions_match:
                    questions_block = questions_match.group(1).strip()
                    gemini_follow_up_questions = [q.strip().lstrip('-').strip() for q in questions_block.split('\n') if
                                                  q.strip()]
                if advice_match:
                    gemini_general_advice = advice_match.group(1).strip()

                if not all([explanation_match, questions_match, advice_match]):
                    logger.warning(
                        f"Could not parse all parts of the Gemini response. Response: {enriched_response_text[:300]}...")

        # Log the full prediction details to the console for debugging
        logger.info(f"Prediction Details:\n"
                    f"  - Original Complaint: {complaint_text_original}\n"
                    f"  - Enhanced Complaint: {complaint_enhanced_status}\n"
                    f"  - Predicted Urgency: {predicted_urgency}\n"
                    f"  - Predicted Specialist: {predicted_specialist}\n"
                    f"  - Gemini Explanation: {gemini_explanation}")

        return jsonify({
            'complaint_original': complaint_text_original,
            'complaint_enhanced': complaint_enhanced_status,
            'complaint_processed_for_model': processed_complaint,
            'predicted_urgency': predicted_urgency,
            'urgency_confidence': urgency_confidence,
            'predicted_specialist': predicted_specialist,
            'specialist_confidence': specialist_confidence,
            'gemini_explanation': gemini_explanation,
            'gemini_follow_up_questions': gemini_follow_up_questions,
            'gemini_general_advice': gemini_general_advice
        })

    except Exception as e:
        logger.error(f"Error in /predict endpoint: {e}", exc_info=True)
        return jsonify({'error': 'An internal server error occurred while processing the request.'}), 500


@app.route('/train', methods=['GET', 'POST'])
def retrain_models_endpoint():
    """Endpoint to trigger model retraining."""
    global active_model_urgency, active_model_specialist
    logger.info("Retraining request received...")
    ensure_data_file_exists()

    trained_urgency, trained_specialist = train_models()
    if trained_urgency and trained_specialist:
        active_model_urgency, active_model_specialist = trained_urgency, trained_specialist
        return jsonify({'message': 'Models have been successfully retrained and saved.'})
    else:
        return jsonify(
            {'error': 'An error occurred during model retraining. Check the data file and server logs.'}), 500


@app.route('/augment_data', methods=['POST'])
def augment_data_endpoint():
    """Endpoint to generate and add new data using Gemini."""
    if not gemini_model:
        return jsonify({'error': 'Gemini model is not configured. Data augmentation is not possible.'}), 503

    try:
        data = request.get_json()
        num_examples = data.get('num_examples', 5)
        target_urgency = data.get('target_urgency')
        target_specialist = data.get('target_specialist')

        if not isinstance(num_examples, int) or not (1 <= num_examples <= 50):
            return jsonify({'error': 'num_examples must be an integer between 1 and 50.'}), 400

        logger.info(
            f"Generating {num_examples} new complaints. Target Urgency: {target_urgency or 'any'}, Specialist: {target_specialist or 'any'}")

        prompt_lines = [
            f"Generate {num_examples} new examples of medical complaints in English, formatted as CSV.",
            "Each line must be in the format: 'complaint;urgency_category;doctor_referral'.",
            "Do not include a CSV header or any other text, introduction, or explanation in your response—only the CSV lines.",
            "The complaints should be diverse and reflect realistic symptoms.",
            "If a target category or specialist is provided, the complaint's content must match it."
        ]
        if target_urgency:
            prompt_lines.append(f"The urgency_category for all examples must be '{target_urgency}'.")
        if target_specialist:
            prompt_lines.append(f"The doctor_referral for all examples must be '{target_specialist}'.")

        # Add existing unique categories/specialists to the prompt for context
        if os.path.exists(DATA_FILE):
            try:
                df_existing = pd.read_csv(DATA_FILE, sep=';', encoding='utf-8')
                if 'urgency_category' in df_existing.columns and not target_urgency:
                    unique_urgencies = df_existing['urgency_category'].dropna().unique().tolist()
                    if unique_urgencies:
                        prompt_lines.append(f"Possible urgency categories include: {', '.join(unique_urgencies)}.")
                if 'doctor_referral' in df_existing.columns and not target_specialist:
                    unique_specialists = df_existing['doctor_referral'].dropna().unique().tolist()
                    if unique_specialists:
                        prompt_lines.append(f"Possible doctor referrals include: {', '.join(unique_specialists)}.")
            except Exception as e:
                logger.warning(f"Could not read existing categories/specialists for data augmentation context: {e}")

        generation_prompt = "\n".join(prompt_lines)
        logger.debug(f"Prompt for Gemini data generation: {generation_prompt}")

        generated_csv_text = generate_gemini_content(generation_prompt)

        if not generated_csv_text or "Could not get a response from Gemini" in generated_csv_text:
            logger.error(f"Gemini failed to generate data. Response: {generated_csv_text}")
            return jsonify({'error': f'Gemini failed to generate new data. Response: {generated_csv_text}'}), 500

        new_entries = []
        for line in generated_csv_text.strip().split('\n'):
            parts = line.strip().split(';')
            if len(parts) == 3:
                complaint, urgency, specialist = parts[0].strip(), parts[1].strip(), parts[2].strip()
                if complaint and urgency and specialist:
                    new_entries.append(f"{complaint};{urgency};{specialist}\n")
            else:
                logger.warning(f"Skipping improperly formatted line from Gemini: '{line}'")

        if not new_entries:
            return jsonify({'message': 'No new data was generated or the format was incorrect.'}), 200

        ensure_data_file_exists()
        with open(DATA_FILE, 'a', encoding='utf-8') as f:
            for entry in new_entries:
                f.write(entry)

        return jsonify({
            'message': f'{len(new_entries)} new entries successfully added to "{DATA_FILE}".',
            'added_entries_count': len(new_entries),
            'first_few_entries_added': new_entries[:min(3, len(new_entries))]
        }), 201

    except Exception as e:
        logger.error(f"Error in /augment_data endpoint: {e}", exc_info=True)
        return jsonify({'error': 'An internal server error occurred during data augmentation.'}), 500


if __name__ == '__main__':
    # Initial model setup is now handled globally when the script starts.
    app.run(debug=True, host='0.0.0.0', port=5000)