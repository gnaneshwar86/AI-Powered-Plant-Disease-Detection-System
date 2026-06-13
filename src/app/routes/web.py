import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from flask import Blueprint, render_template, request, current_app, redirect, url_for
from werkzeug.utils import secure_filename

from src.app.monitoring import PREDICTION_COUNT, PREDICTION_LATENCY, ERROR_COUNT
from src.model.gradcam import save_gradcam_visualization

web_bp = Blueprint('web', __name__)

@web_bp.route('/')
def home_page():
    return render_template('home.html')

@web_bp.route('/contact')
def contact():
    return render_template('contact-us.html')

@web_bp.route('/index')
def ai_engine_page():
    return render_template('index.html')

@web_bp.route('/market')
def market():
    disease_info = current_app.config.get('DISEASE_INFO')
    supplement_info = current_app.config.get('SUPPLEMENT_INFO')
    
    if disease_info is None or supplement_info is None:
        return "Metadata not loaded. Make sure CSV files are present in the root folder.", 500
        
    return render_template(
        'market.html',
        supplement_image=list(supplement_info['supplement image']),
        supplement_name=list(supplement_info['supplement name']),
        disease=list(disease_info['disease_name']),
        buy=list(supplement_info['buy link'])
    )

@web_bp.route('/submit', methods=['GET', 'POST'])
def submit():
    """Handles classic form POST uploads and renders submission results page."""
    if request.method != 'POST':
        return redirect(url_for('web.ai_engine_page'))
        
    if 'image' not in request.files:
        return redirect(url_for('web.ai_engine_page'))
        
    file = request.files['image']
    if file.filename == '':
        return redirect(url_for('web.ai_engine_page'))
        
    try:
        model = current_app.config.get('MODEL')
        disease_info = current_app.config.get('DISEASE_INFO')
        supplement_info = current_app.config.get('SUPPLEMENT_INFO')
        
        if model is None or disease_info is None or supplement_info is None:
            raise RuntimeError("Flask app state is uninitialized. Model or metadata are missing.")
            
        uploads_dir = os.path.join(current_app.static_folder, 'uploads')
        os.makedirs(uploads_dir, exist_ok=True)
        
        filename = secure_filename(file.filename)
        # Use timestamped filename to prevent caching/overwrites
        filename_ts = f"{int(time.time())}_{filename}"
        file_path = os.path.join(uploads_dir, filename_ts)
        file.save(file_path)
        
        # Read and preprocess image
        image = Image.open(file_path).convert('RGB')
        input_image = image.resize((224, 224))
        input_tensor = TF.to_tensor(input_image).unsqueeze(0)
        
        # Inference with latency tracking
        start_time = time.time()
        with PREDICTION_LATENCY.time():
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = F.softmax(outputs, dim=1).squeeze(0).cpu().numpy()
                
        # Top 3 predicted classes
        top3_indices = np.argsort(probabilities)[::-1][:3]
        pred = int(top3_indices[0])
        
        title = disease_info['disease_name'][pred]
        description = disease_info['description'][pred]
        prevent = disease_info['Possible Steps'][pred]
        image_url = disease_info['image_url'][pred]
        
        supplement_name = supplement_info['supplement name'][pred]
        supplement_image_url = supplement_info['supplement image'][pred]
        supplement_buy_link = supplement_info['buy link'][pred]
        
        # Grad-CAM visualization output
        gradcam_filename = f"gradcam_{filename_ts}"
        gradcam_path = os.path.join(uploads_dir, gradcam_filename)
        
        success, _ = save_gradcam_visualization(model, file_path, input_tensor, gradcam_path, pred)
        
        gradcam_url = f"/static/uploads/{gradcam_filename}" if success else None
        
        # Prepare list of Top-3 predictions for front-end charts
        top_predictions = []
        for idx in top3_indices:
            idx = int(idx)
            top_predictions.append({
                "disease_name": disease_info['disease_name'][idx],
                "confidence": float(probabilities[idx]) * 100
            })
            
        # Log success metrics
        PREDICTION_COUNT.labels(disease_name=title, status="success").inc()
        
        return render_template(
            'submit.html',
            title=title,
            desc=description,
            prevent=prevent,
            image_url=image_url,
            pred=pred,
            sname=supplement_name,
            simage=supplement_image_url,
            buy_link=supplement_buy_link,
            gradcam_url=gradcam_url,
            original_image_url=f"/static/uploads/{filename_ts}",
            top_predictions=top_predictions,
            confidence=float(probabilities[pred]) * 100
        )
        
    except Exception as e:
        ERROR_COUNT.labels(error_type=type(e).__name__).inc()
        print(f"Error handling page upload: {e}")
        return render_template('index.html', error=f"An error occurred during prediction: {str(e)}")
