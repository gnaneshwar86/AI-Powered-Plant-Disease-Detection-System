import numpy as np
import torch
import cv2
import os

class GradCAM:
    def __init__(self, model, target_layer=None):
        """
        Grad-CAM tool to generate heatmaps showing which parts of an image the model focused on.
        If target_layer is None, it dynamically searches for the last Conv2d layer in the model.
        """
        self.model = model
        self.gradients = None
        self.activations = None
        
        # Detect target conv layer if not specified
        if target_layer is None:
            self.target_layer = self._detect_target_layer()
        else:
            self.target_layer = target_layer
            
        # Register hooks for forward and backward passes
        self.target_layer.register_forward_hook(self._save_activation)
        
        # register_full_backward_hook is standard in newer PyTorch versions
        if hasattr(self.target_layer, 'register_full_backward_hook'):
            self.target_layer.register_full_backward_hook(self._save_gradient)
        else:
            self.target_layer.register_backward_hook(self._save_gradient)

    def _detect_target_layer(self):
        """Finds the last convolutional layer in the model."""
        conv_layers = []
        for module in self.model.modules():
            if isinstance(module, torch.nn.Conv2d):
                conv_layers.append(module)
        
        if not conv_layers:
            raise ValueError("No Conv2d layers found in the provided model architecture.")
        
        # Use the final conv layer
        return conv_layers[-1]

    def _save_activation(self, module, input, output):
        self.activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        # grad_output is a tuple; gradients are the first element
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_tensor, class_idx=None):
        """
        Generates a 2D Grad-CAM heatmap normalized between [0, 1].
        """
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()
            
        # Zero out gradients
        self.model.zero_grad()
        
        # Backward pass for the specific target class
        target_score = output[0, class_idx]
        target_score.backward(retain_graph=True)
        
        # Extract gradients and activations
        if self.gradients is None or self.activations is None:
            # Fallback if hook was not triggered (e.g. backward hook issue with some modules)
            # We return a dummy empty heatmap of the size of the input tensor (last 2 dims)
            h, w = input_tensor.shape[2], input_tensor.shape[3]
            return np.zeros((h, w), dtype=np.float32), class_idx
            
        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]
        
        # Mean gradients across height and width to get channel weights
        weights = np.mean(gradients, axis=(1, 2))
        
        # Compute weighted sum of activations
        heatmap = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            heatmap += w * activations[i]
            
        # Apply ReLU to keep positive features
        heatmap = np.maximum(heatmap, 0)
        
        # Normalize heatmap to [0, 1] range
        max_val = np.max(heatmap)
        if max_val > 0:
            heatmap = heatmap / max_val
            
        return heatmap, class_idx


def overlay_heatmap(image_path, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """
    Applies the heatmap as a colored overlay over the original image.
    Saves the output or returns the BGR image data.
    """
    # Read original image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load original image from: {image_path}")
        
    # Resize heatmap to match original image dimensions
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    
    # Scale heatmap to [0, 255] and convert to 8-bit integer
    heatmap_color = np.uint8(255 * heatmap_resized)
    
    # Apply colormap to get RGB representation
    heatmap_colored = cv2.applyColorMap(heatmap_color, colormap)
    
    # Blend the original image and colormapped heatmap
    overlay = cv2.addWeighted(img, 1 - alpha, heatmap_colored, alpha, 0)
    
    return overlay


def save_gradcam_visualization(model, image_path, input_tensor, output_path, class_idx=None):
    """
    Convenience method to generate, overlay, and save a Grad-CAM visualization.
    """
    try:
        gradcam = GradCAM(model)
        heatmap, pred_class = gradcam.generate_heatmap(input_tensor, class_idx)
        overlay = overlay_heatmap(image_path, heatmap)
        
        # Ensure directories exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, overlay)
        return True, pred_class
    except Exception as e:
        print(f"Error generating Grad-CAM: {e}")
        return False, None
