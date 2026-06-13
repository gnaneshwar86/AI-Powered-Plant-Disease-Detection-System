import os
import time
import io
import base64
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename

from src.app.monitoring import PREDICTION_COUNT, PREDICTION_LATENCY, ERROR_COUNT
from src.model.gradcam import save_gradcam_visualization

api_bp = Blueprint('api', __name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@api_bp.route('/health', methods=['GET'])
def health_check():
    """Returns application health and model load status."""
    model_name = current_app.config.get('MODEL_NAME', 'unknown')
    model_status = "loaded" if current_app.config.get('MODEL') is not None else "failed"
    return jsonify({
        "status": "healthy",
        "model": {
            "name": model_name,
            "status": model_status
        },
        "timestamp": time.time()
    }), 200

@api_bp.route('/diseases', methods=['GET'])
def get_diseases():
    """Retrieves list of all crop diseases and metadata."""
    disease_info = current_app.config.get('DISEASE_INFO')
    if disease_info is None:
        return jsonify({"error": "Disease metadata not loaded."}), 500
        
    diseases = []
    for idx, row in disease_info.iterrows():
        diseases.append({
            "class_id": int(idx),
            "disease_name": row['disease_name'],
            "description": row['description'],
            "possible_steps": row['Possible Steps'],
            "image_url": row.get('image_url', '')
        })
    return jsonify({"diseases": diseases}), 200

@api_bp.route('/diseases/<int:class_id>', methods=['GET'])
def get_disease(class_id):
    """Retrieves description and treatment steps for a specific disease class ID."""
    disease_info = current_app.config.get('DISEASE_INFO')
    if disease_info is None:
        return jsonify({"error": "Disease metadata not loaded."}), 500
        
    if class_id < 0 or class_id >= len(disease_info):
        return jsonify({"error": f"Disease class ID {class_id} not found."}), 404
        
    row = disease_info.iloc[class_id]
    return jsonify({
        "class_id": class_id,
        "disease_name": row['disease_name'],
        "description": row['description'],
        "possible_steps": row['Possible Steps'],
        "image_url": row.get('image_url', '')
    }), 200

@api_bp.route('/predict', methods=['POST'])
def predict():
    """
    Accepts multipart/form-data upload of leaf images.
    Returns primary prediction, Top-3 class breakdown, and Grad-CAM localization.
    """
    start_time = time.time()
    
    if 'image' not in request.files:
        ERROR_COUNT.labels(error_type="missing_image_file").inc()
        return jsonify({"error": "No image file provided in key 'image'"}), 400
        
    file = request.files['image']
    if file.filename == '':
        ERROR_COUNT.labels(error_type="empty_filename").inc()
        return jsonify({"error": "Empty filename provided"}), 400
        
    if not allowed_file(file.filename):
        ERROR_COUNT.labels(error_type="invalid_file_extension").inc()
        return jsonify({"error": f"Invalid file extension. Allowed extensions: {ALLOWED_EXTENSIONS}"}), 400
        
    try:
        model = current_app.config.get('MODEL')
        disease_info = current_app.config.get('DISEASE_INFO')
        supplement_info = current_app.config.get('SUPPLEMENT_INFO')
        
        if model is None or disease_info is None or supplement_info is None:
            raise RuntimeError("Application context models or CSV metadata are not fully initialized.")
        
        # Read image bytes
        img_bytes = file.read()
        image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        
        # Preprocess
        input_image = image.resize((224, 224))
        input_tensor = TF.to_tensor(input_image)
        input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension
        
        # Measure latency
        with PREDICTION_LATENCY.time():
            with torch.no_grad():
                outputs = model(input_tensor)
                # Compute probabilities
                probabilities = F.softmax(outputs, dim=1).squeeze(0).cpu().numpy()
                
        # Top 3 classes
        top3_indices = np.argsort(probabilities)[::-1][:3]
        
        top3_predictions = []
        for rank, idx in enumerate(top3_indices):
            idx = int(idx)
            top3_predictions.append({
                "rank": rank + 1,
                "class_id": idx,
                "disease_name": disease_info['disease_name'][idx],
                "confidence": float(probabilities[idx])
            })
            
        primary_idx = int(top3_indices[0])
        primary_disease = disease_info['disease_name'][primary_idx]
        primary_desc = disease_info['description'][primary_idx]
        primary_prevent = disease_info['Possible Steps'][primary_idx]
        primary_image = disease_info['image_url'][primary_idx]
        
        # Supplement recommendations
        supp_name = supplement_info['supplement name'][primary_idx]
        supp_image = supplement_info['supplement image'][primary_idx]
        supp_buy = supplement_info['buy link'][primary_idx]
        
        # Save original file temporarily to overlay Grad-CAM
        uploads_dir = os.path.join(current_app.root_path, 'static', 'uploads')
        os.makedirs(uploads_dir, exist_ok=True)
        
        temp_filename = f"temp_{int(time.time())}_{secure_filename(file.filename)}"
        temp_path = os.path.join(uploads_dir, temp_filename)
        image.save(temp_path)
        
        # Run Grad-CAM visualization
        gradcam_filename = f"gradcam_{temp_filename}"
        gradcam_path = os.path.join(uploads_dir, gradcam_filename)
        
        success, _ = save_gradcam_visualization(model, temp_path, input_tensor, gradcam_path, primary_idx)
        
        # Read back as base64 bytes for integration ease
        gradcam_base64 = ""
        if success and os.path.exists(gradcam_path):
            with open(gradcam_path, "rb") as gc_file:
                gradcam_base64 = base64.b64encode(gc_file.read()).decode('utf-8')
                
        # Clean up temporary original image to save disk space
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as e:
                print(f"Warning: Failed to clean up temp file {temp_path}: {e}")
                
        # Update metrics
        PREDICTION_COUNT.labels(disease_name=primary_disease, status="success").inc()
        latency = time.time() - start_time
        
        return jsonify({
            "prediction": {
                "class_id": primary_idx,
                "disease_name": primary_disease,
                "confidence": float(probabilities[primary_idx]),
                "description": primary_desc,
                "prevention": primary_prevent,
                "image_url": primary_image
            },
            "top_predictions": top3_predictions,
            "supplement": {
                "name": supp_name,
                "image_url": supp_image,
                "buy_link": supp_buy
            },
            "gradcam_image_base64": f"data:image/jpeg;base64,{gradcam_base64}" if gradcam_base64 else "",
            "gradcam_url": f"/static/uploads/{gradcam_filename}" if success else "",
            "latency_seconds": latency
        }), 200
        
    except Exception as e:
        ERROR_COUNT.labels(error_type=type(e).__name__).inc()
        return jsonify({"error": f"Prediction server failure: {str(e)}"}), 500
