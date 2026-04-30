# # ----------------------------
# # IMPORTS
# # ----------------------------
# import torch
# import torch.nn as nn
# from torchvision.models import resnet18
# import torchvision.transforms as transforms
# from PIL import Image
# import numpy as np
# import cv2
# import os
# import logging
# import time

# from flask import Flask, request, jsonify
# from flask_cors import CORS

# # ----------------------------
# # APP CONFIG
# # ----------------------------
# app = Flask(__name__)
# CORS(app)

# logging.basicConfig(level=logging.INFO)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # ----------------------------
# # LOAD MODEL
# # ----------------------------
# MODEL_PATH = "lung_disease_model.pth"

# if not os.path.exists(MODEL_PATH):
#     raise FileNotFoundError("Model file not found!")

# model = resnet18()
# model.fc = nn.Linear(model.fc.in_features, 3)
# model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
# model.to(device)
# model.eval()

# classes = ['Normal', 'Lung_Opacity', 'Viral Pneumonia']

# # ----------------------------
# # IMAGE ENHANCEMENT (CLAHE)
# # ----------------------------
# def enhance_image(image):
#     img = np.array(image)
#     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

#     lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
#     l, a, b = cv2.split(lab)

#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#     l = clahe.apply(l)

#     enhanced = cv2.merge((l, a, b))
#     enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
#     enhanced = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)

#     return Image.fromarray(enhanced)

# # ----------------------------
# # TRANSFORM PIPELINE
# # ----------------------------
# transform = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406],
#                          [0.229, 0.224, 0.225])
# ])

# # ----------------------------
# # HEALTH CHECK ROUTE
# # ----------------------------
# @app.route('/')
# def home():
#     return jsonify({"message": "Lung Disease Detection API is running 🚀"})

# # ----------------------------
# # PREDICTION ROUTE
# # ----------------------------
# @app.route('/predict', methods=['POST'])
# def predict():
#     start_time = time.time()

#     # ---------- VALIDATION ----------
#     if 'file' not in request.files:
#         return jsonify({"status": "error", "message": "No file uploaded"}), 400

#     file = request.files['file']

#     if file.filename == '':
#         return jsonify({"status": "error", "message": "Empty file"}), 400

#     try:
#         image = Image.open(file).convert("RGB")
#     except:
#         return jsonify({"status": "error", "message": "Invalid image format"}), 400

#     # ---------- PREPROCESS ----------
#     image = enhance_image(image)
#     image = transform(image).unsqueeze(0).to(device)

#     # ---------- PREDICTION ----------
#     with torch.no_grad():
#         output = model(image)

#         prob = torch.softmax(output, dim=1).cpu().numpy()[0]

#         # Top predictions
#         pred = np.argmax(prob)
#         confidence = float(prob[pred])

#         # Confidence gap (important)
#         sorted_probs = np.sort(prob)
#         top1 = sorted_probs[-1]
#         top2 = sorted_probs[-2]
#         confidence_gap = float(top1 - top2)

#     # ---------- REJECTION LOGIC ----------
#     if confidence < 0.6 or confidence_gap < 0.15:
#         return jsonify({
#             "status": "error",
#             "message": "Invalid image or not a chest X-ray"
#         }), 400

#     # ---------- RESPONSE ----------
#     result = {
#         "prediction": classes[pred],
#         "confidence": round(confidence, 4),
#         "status": "success",
#         "processing_time_sec": round(time.time() - start_time, 3)
#     }

#     logging.info(f"Prediction: {result}")

#     return jsonify(result)

# # ----------------------------
# # RUN SERVER
# # ----------------------------
# if __name__ == '__main__':
#     app.run(debug=True)

# ----------------------------
# IMPORTS
# ----------------------------
import torch
import torch.nn as nn
from torchvision.models import resnet18
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import os
import logging
import time

from flask import Flask, request, jsonify
from flask_cors import CORS

# ----------------------------
# APP CONFIG
# ----------------------------
app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# LOAD MODEL
# ----------------------------
MODEL_PATH = "lung_disease_model.pth"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model file not found!")

model = resnet18()
model.fc = nn.Linear(model.fc.in_features, 3)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

classes = ['Normal', 'Lung_Opacity', 'Viral Pneumonia']

# ----------------------------
# ROBUST X-RAY VALIDATION
# ----------------------------
def is_xray_like(image):
    try:
        img = np.array(image)

        # Safety: remove NaN/Inf
        img = np.nan_to_num(img).astype(np.uint8)

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # 1. Check intensity variation
        std_dev = np.std(gray)
        if std_dev < 10:
            return False

        # 2. Edge detection (lungs/ribs structure)
        edges = cv2.Canny(gray, 50, 150)
        edge_ratio = np.sum(edges > 0) / edges.size

        if edge_ratio < 0.01:
            return False

        return True

    except Exception as e:
        logging.error(f"X-ray validation error: {e}")
        return False

# ----------------------------
# IMAGE ENHANCEMENT (CLAHE)
# ----------------------------
def enhance_image(image):
    img = np.array(image)
    img = np.nan_to_num(img).astype(np.uint8)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    enhanced = cv2.merge((l, a, b))
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)

    return Image.fromarray(enhanced)

# ----------------------------
# TRANSFORM PIPELINE
# ----------------------------
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ----------------------------
# HEALTH CHECK ROUTE
# ----------------------------
@app.route('/')
def home():
    return jsonify({"message": "Lung Disease Detection API is running 🚀"})

# ----------------------------
# PREDICTION ROUTE
# ----------------------------
@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()

    # ---------- VALIDATION ----------
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file uploaded"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"status": "error", "message": "Empty file"}), 400

    try:
        image = Image.open(file).convert("RGB")
    except:
        return jsonify({"status": "error", "message": "Invalid image format"}), 400

    # ---------- X-RAY CHECK ----------
    if not is_xray_like(image):
        return jsonify({
            "status": "error",
            "message": "Not a valid chest X-ray"
        }), 400

    # ---------- PREPROCESS ----------
    image = enhance_image(image)
    image = transform(image).unsqueeze(0).to(device)

    # ---------- PREDICTION ----------
    try:
        with torch.no_grad():
            output = model(image)

            prob = torch.softmax(output, dim=1).cpu().numpy()[0]
            prob = np.nan_to_num(prob)

            pred = np.argmax(prob)
            confidence = float(prob[pred])

            sorted_probs = np.sort(prob)
            top1 = sorted_probs[-1]
            top2 = sorted_probs[-2]
            confidence_gap = float(top1 - top2)

    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return jsonify({
            "status": "error",
            "message": "Model prediction failed"
        }), 500

    # ---------- REJECTION LOGIC ----------
    if confidence < 0.30 or confidence_gap < 0.35:
        return jsonify({
            "status": "error",
            "message": "Unclear or low-confidence X-ray"
        }), 400

    # ---------- RESPONSE ----------
    result = {
        "prediction": classes[pred],
        "confidence": round(confidence, 4),
        "status": "success",
        "processing_time_sec": round(time.time() - start_time, 3)
    }

    logging.info(f"Prediction: {result}")

    return jsonify(result)

# ----------------------------
# RUN SERVER
# ----------------------------
if __name__ == '__main__':
    app.run(debug=True)
