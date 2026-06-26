from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import base64
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
import os
import json

app = Flask(__name__)
CORS(app)

# Load pre-trained model or create one if not exists
MODEL_PATH = 'facial_expression_model.h5'
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def create_model():
    """Create a CNN model for facial expression recognition"""
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(7, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def load_or_create_model():
    """Load existing model or create a new one with sample data"""
    if os.path.exists(MODEL_PATH):
        print("Loading existing model...")
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    else:
        print("Creating new model with sample data...")
        model = create_model()
        # Train with sample data (in production, you'd use real dataset)
        X_sample = np.random.rand(100, 48, 48, 1).astype(np.float32)
        y_sample = tf.keras.utils.to_categorical(np.random.randint(0, 7, 100), num_classes=7)
        
        model.fit(X_sample, y_sample, epochs=5, batch_size=16, verbose=0)
        model.save(MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}")
        return model

# Load model globally
model = load_or_create_model()

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Predict facial expression from image data"""
    try:
        data = request.json
        image_data = data['image']
        
        # Decode base64 image
        image_data = image_data.split(',')[1] if ',' in image_data else image_data
        image_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            return jsonify({'error': 'Invalid image data'}), 400
        
        # Preprocess
        img = cv2.resize(img, (48, 48))
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=0)
        img = np.expand_dims(img, axis=-1)
        
        # Predict
        predictions = model.predict(img, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))
        emotion = EMOTIONS[predicted_class]
        
        # Get top 3 predictions
        top_indices = np.argsort(predictions[0])[-3:][::-1]
        top_emotions = [
            {'emotion': EMOTIONS[i], 'confidence': float(predictions[0][i])}
            for i in top_indices
        ]
        
        return jsonify({
            'emotion': emotion,
            'confidence': confidence,
            'top_predictions': top_emotions,
            'all_probabilities': {
                EMOTIONS[i]: float(predictions[0][i]) 
                for i in range(len(EMOTIONS))
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    print("🚀 Starting Facial Expression Detector...")
    print(f"📊 Model loaded with {len(EMOTIONS)} emotion classes")
    print(f"🌐 Server running at http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
