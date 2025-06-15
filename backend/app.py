import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
import numpy as np

app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app, origins=[
    "http://localhost:5000",
    "http://127.0.0.1:5000",
    "http://localhost:5173",
    "http://127.0.0.1:5173"
])
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB limit

# Absolute path handling for Docker
UPLOAD_DIR = os.path.join(os.getcwd(), 'uploads')
os.makedirs(UPLOAD_DIR, exist_ok=True)

nltk.download('punkt')
nltk.download('stopwords')

# Model loading
vectorizer = joblib.load('model/tfidf_vectorizer.pkl')
clf_text = joblib.load('model/knn_model.pkl')
image_model = load_model('model/skin_lesion_model.h5')
image_class_names = joblib.load('model/image_class_names.pkl')
IMAGE_SIZE = (224, 224)

stop_words = set(stopwords.words('english'))
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(uploaded_image):
    img = keras_image.load_img(uploaded_image, target_size=IMAGE_SIZE)
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

def preprocess_text(text):
    words = word_tokenize(text.lower())
    words = [word for word in words if word.isalpha() and word not in stop_words]
    return ' '.join(words)

@app.route('/predict', methods=['POST'])
def predict():
    text_pred = None
    image_pred = None
    filename = None
    img_path = None  # Track image path for cleanup

    try:
        # Text prediction
        if 'symptoms' in request.form:
            text = request.form['symptoms'].strip()
            if not text:
                raise ValueError("Symptoms description cannot be empty")
            processed_text = preprocess_text(text)
            text_features = vectorizer.transform([processed_text])
            text_pred = clf_text.predict(text_features)[0]

        # Image prediction
        if 'image' in request.files:
            image_file = request.files['image']
            if image_file and allowed_file(image_file.filename):
                filename = secure_filename(image_file.filename)
                img_path = os.path.join(UPLOAD_DIR, filename)
                image_file.save(img_path)

                img_array = preprocess_image(img_path)
                pred = image_model.predict(img_array)
                class_idx = np.argmax(pred, axis=1)[0]
                image_pred = image_class_names[class_idx]

        return jsonify({
            'disease_from_text': text_pred,
            'disease_from_image': image_pred,
            'image_filename': filename
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400
    finally:
        # Clean up uploaded image
        if img_path and os.path.exists(img_path):
            os.remove(img_path)

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path != '' and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0',port = 5000)
