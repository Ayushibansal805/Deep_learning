from flask import Flask, render_template, request
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Load model
# Load model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "brain_tumor_model1.h5")

print("Looking for model at:", MODEL_PATH)
print("File exists:", os.path.exists(MODEL_PATH))

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

model = load_model(MODEL_PATH, compile=False)  # Avoid potential issues with custom objects

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")



# ⚠️ IMPORTANT: Set this SAME as training
IMG_SIZE = (256, 256, 3)   # <-- change if your model used different size

# ⚠️ MUST match training class order
class_names = ["glioma", "meningioma", "pituitary", "notumor"]

def preprocess(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)

    # Ensure 3 channels
    if img_array.shape[-1] == 1:
        img_array = np.repeat(img_array, 3, axis=-1)

    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    return img_array

@app.route("/")
def index():
    return render_template("index.html")

from flask import jsonify

@app.route("/predict", methods=["POST"])
def predict():
    try:
        file = request.files["file"]

        os.makedirs("uploads", exist_ok=True)
        filepath = os.path.join("uploads", file.filename)
        file.save(filepath)

        # preprocess image
        img = preprocess(filepath)

        # 🔥 model prediction
        prediction = model.predict(img)

        # get predicted class
        predicted_index = np.argmax(prediction)
        result = class_names[predicted_index]

        confidence = float(np.max(prediction))  # keep this (0–1)

        # ✅ ADD YOUR CODE HERE
        probs = prediction[0]

        probabilities = [
            {"label": class_names[i], "value": float(probs[i] * 100)}
            for i in range(len(class_names))
        ]

        # ✅ RETURN JSON (IMPORTANT)
        return jsonify({
            "prediction": result,
            "confidence": round(confidence * 100, 2),
            "probabilities": probabilities
        })

    except Exception as e:
        return jsonify({"error": str(e)})
# Optional: remove favicon error
@app.route('/favicon.ico')
def favicon():
    return '', 204

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)