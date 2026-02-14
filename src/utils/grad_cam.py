import torch
import torch.nn.functional as F
import cv2
import numpy as np

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_layers()

    def hook_layers(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_heatmap(self, input_tensor):
        """
        Generates heatmap for the single output logit.
        """
        self.model.eval()
        output = self.model(input_tensor)
        
        # For binary classification with 1 output, we just backward on the output itself
        self.model.zero_grad()
        output.backward()

        gradients = self.gradients.data.cpu().numpy()
        activations = self.activations.data.cpu().numpy()

        # Global Average Pooling of gradients
        weights = np.mean(gradients, axis=(2, 3))[0]
        
        # Weighted sum of activations
        heatmap = np.zeros(activations.shape[2:], dtype=np.float32)
        for i, w in enumerate(weights):
            heatmap += w * activations[0, i, :, :]

        # ReLU on heatmap
        heatmap = np.maximum(heatmap, 0)
        
        # Normalize
        heatmap_max = np.max(heatmap)
        if heatmap_max > 0:
            heatmap /= heatmap_max
            
        return heatmap

def overlay_heatmap(img, heatmap, alpha=0.5, colormap=cv2.COLORMAP_JET):
    """
    Overlays a heatmap on an image.
    img: RGB image [H, W, 3] in range [0, 255]
    heatmap: 2D array [H', W'] in range [0, 1]
    """
    # Resize heatmap to match image size
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    
    # Convert heatmap to RGB
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), colormap)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    
    # Overlay
    overlayed_img = cv2.addWeighted(img, 1 - alpha, heatmap_color, alpha, 0)
    return overlayed_img
