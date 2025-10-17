from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import base64
import os
import pickle
from transformers import T5ForConditionalGeneration, T5Tokenizer

app = Flask(__name__)

# Load face cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Store face data
face_data = {}
DATA_FILE = 'face_data.pkl'

# Load AI model (lightweight)
print("Loading AI model...")
model_name = "google/flan-t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)
print("AI model loaded!")

def load_face_data():
    global face_data
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'rb') as f:
            face_data = pickle.load(f)

def save_face_data():
    with open(DATA_FILE, 'wb') as f:
        pickle.dump(face_data, f)

def extract_face_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) > 0:
        x, y, w, h = faces[0]
        face_roi = gray[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, (100, 100))
        return face_roi.flatten()
    return None

def compare_faces(features1, features2, threshold=0.8):
    correlation = np.corrcoef(features1, features2)[0, 1]
    return correlation > threshold

def generate_response(prompt):
    input_text = f"Answer this question: {prompt}"
    input_ids = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).input_ids
    outputs = model.generate(input_ids, max_length=100, num_beams=2, early_stopping=True)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['POST'])
def register_face():
    data = request.json
    name = data['name']
    image_data = data['image'].split(',')[1]
    
    image_bytes = base64.b64decode(image_data)
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    features = extract_face_features(image)
    if features is not None:
        face_data[name] = features
        save_face_data()
        cv2.imwrite(f"faces/{name}.jpg", image)
        return jsonify({"success": True, "message": "Face registered successfully"})
    
    return jsonify({"success": False, "message": "No face detected"})

@app.route('/login', methods=['POST'])
def login():
    data = request.json
    image_data = data['image'].split(',')[1]
    
    image_bytes = base64.b64decode(image_data)
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    features = extract_face_features(image)
    if features is not None:
        for name, stored_features in face_data.items():
            if compare_faces(features, stored_features):
                return jsonify({"success": True, "message": f"Welcome, {name}!", "user": name})
    
    return jsonify({"success": False, "message": "Face not recognized"})

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    message = data['message']
    response = generate_response(message)
    return jsonify({"response": response})

if __name__ == '__main__':
    load_face_data()
    app.run(debug=True)
