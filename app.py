from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import os

# Initialize API Server
app = Flask(__name__)
# CORS is fully required now because your HTML and Python are running separately!
CORS(app) 

# Safely load the AI Model (if it exists)
try:
    from tensorflow.keras.models import load_model
    MODEL_PATH = 'action_model.h5'
    if os.path.exists(MODEL_PATH):
        model = load_model(MODEL_PATH)
        MODEL_EXISTS = True
        print("🧠 Neural Network loaded successfully!")
    else:
        MODEL_EXISTS = False
        print("⚠️ Model not found. Running in UI Simulation Mode.")
except ImportError:
    MODEL_EXISTS = False
    print("⚠️ TensorFlow not fully loaded. Running in Simulation Mode.")

ACTIONS = ['Hello', 'Thanks', 'I Love You', 'Yes', 'No', 'Help']

# ==========================================
# ONLY ONE ROUTE: THE API DATA BRIDGE
# ==========================================
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        sequence = data.get('sequence')
        
        if not sequence or len(sequence) != 30:
            return jsonify({'error': 'Invalid sequence length.'}), 400

        if MODEL_EXISTS:
            input_data = np.expand_dims(sequence, axis=0)
            predictions = model.predict(input_data, verbose=0)[0]
            predicted_index = np.argmax(predictions)
            confidence = float(predictions[predicted_index])

            if confidence > 0.60:
                return jsonify({'action': ACTIONS[predicted_index], 'confidence': round(confidence * 100, 1)})
            else:
                return jsonify({'action': '...', 'confidence': round(confidence * 100, 1)})
        else:
            import random
            is_confident = random.random() > 0.4
            return jsonify({
                'action': random.choice(ACTIONS) if is_confident else '...',
                'confidence': round(random.uniform(65.0, 99.9), 1) if is_confident else 0
            })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🚀 NEUROSIGN API ONLINE!")
    print("👉 Keep this terminal open, then run your index.html using Live Server!")
    print("="*50 + "\n")
    app.run(host='127.0.0.1', port=5000, debug=True)