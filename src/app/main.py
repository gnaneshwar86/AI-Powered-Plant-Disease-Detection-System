import os
import pandas as pd
import torch
from flask import Flask

from src.model.transfer import load_model
from src.app.routes.api import api_bp
from src.app.routes.web import web_bp
from src.app.monitoring import metrics_endpoint

def create_app():
    # Find absolute paths relative to repository root
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
    
    # Initialize Flask with paths pointing to root templates/static directories
    app = Flask(
        __name__,
        template_folder=os.path.join(root_dir, 'templates'),
        static_folder=os.path.join(root_dir, 'static')
    )
    
    # Load Environment Configurations
    model_name = os.environ.get("MODEL_NAME", "efficientnet_b0")
    # First search root path for plant_disease_model_latest.pt, then fallback to baseline.pt
    default_weights_filename = "plant_disease_model_1_latest.pt" if model_name == "cnn" else "plant_disease_model_latest.pt"
    weights_path = os.environ.get("MODEL_WEIGHTS_PATH", os.path.join(root_dir, default_weights_filename))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load CSV Metadata
    disease_info_path = os.path.join(root_dir, 'disease_info.csv')
    supplement_info_path = os.path.join(root_dir, 'supplement_info.csv')
    
    try:
        disease_info = pd.read_csv(disease_info_path, encoding='cp1252')
        supplement_info = pd.read_csv(supplement_info_path, encoding='cp1252')
        app.config['DISEASE_INFO'] = disease_info
        app.config['SUPPLEMENT_INFO'] = supplement_info
        print(f"Loaded metadata from {disease_info_path} and {supplement_info_path}")
    except Exception as e:
        print(f"Critical Error: Failed to load CSV metadata. Ensure they exist at {root_dir}. Error: {e}")
        raise e
        
    # Load Model using Factory
    print(f"Initializing {model_name} on device '{device}'...")
    try:
        model = load_model(
            model_name=model_name,
            num_classes=39,
            weights_path=weights_path,
            device=device
        )
        app.config['MODEL'] = model
        app.config['MODEL_NAME'] = model_name
    except Exception as e:
        print(f"Critical Error: Model initialization failed. Error: {e}")
        # In a production context, initialize model with empty weights to prevent app crash
        app.config['MODEL'] = None
        app.config['MODEL_NAME'] = model_name
        
    # Register Blueprints
    app.register_blueprint(web_bp)
    app.register_blueprint(api_bp, url_prefix="/api/v1")
    
    # Register Prometheus Scrape Route
    app.add_url_rule("/metrics", "metrics", metrics_endpoint)
    
    # Ensure static/uploads folder exists
    os.makedirs(os.path.join(root_dir, 'static', 'uploads'), exist_ok=True)
    
    return app

if __name__ == '__main__':
    app = create_app()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
