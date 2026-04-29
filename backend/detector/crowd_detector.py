"""
CrowdDetector — High-Level Interface for CSRNet.

This module acts as the bridge between raw OpenCV video frames and the raw PyTorch neural network.
It handles all the complex data transformations required before and after AI inference.
"""
import os
import cv2
import torch
import numpy as np
from torchvision import transforms
from .csrnet_model import CSRNet


class CrowdDetector:
    """
    Crowd density estimator using CSRNet.
    
    This class handles:
    1. GPU Memory Management (resizing massive frames).
    2. ImageNet Normalisation (math required to make colors match the training data).
    3. PyTorch Automatic Mixed Precision (AMP) to double inference speed on RTX GPUs.
    4. Post-processing the raw float-based heatmap into visual color maps for the UI.
    """

    # ImageNet normalisation constants.
    # The CSRNet frontend (VGG-16) was originally trained on millions of generic photos from ImageNet.
    # For the model to work, our CCTV frames must have their Red, Green, and Blue channels 
    # normalized using the exact same mean and standard deviation used during that original training.
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    def __init__(self, model_path: str):
        """
        Loads the CSRNet model and pre-trained weights into Video RAM (VRAM) if a GPU is present.
        """
        # Automatically detect if an NVIDIA GPU with CUDA drivers is available.
        # Fallback to the system CPU if not (though this will be exponentially slower).
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Add a clear warning if CUDA isn't available, as user experience will degrade severely.
        if self.device.type == 'cpu':
            print("======================================================")
            print("WARNING: CUDA GPU not found! Running on slow CPU.")
            print("Please install PyTorch with CUDA support to use GPU.")
            print("======================================================")
        else:
            print(f"Using GPU Device: {torch.cuda.get_device_name(self.device)}")

        # Instantiate the model architecture
        self.model = CSRNet(load_weights=True) 
        
        # Load the massive matrix of trained "weights" from disk into memory
        if os.path.isfile(model_path):
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            # Accommodate different saving formats (raw dictionary vs PyTorch checkpoint object)
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
                
            self.model.load_state_dict(state_dict)
            print(f"Loaded CSRNet weights from {model_path}")
        else:
            raise FileNotFoundError(f"CSRNet weights not found: {model_path}")

        # Push the model to the GPU and set it to evaluation mode (disables training features like Dropout)
        self.model.to(self.device)
        self.model.eval()

        # Define the pipeline that will convert an OpenCV image matrix into a PyTorch Tensor
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.MEAN, std=self.STD),
        ])

    def detect(self, frame: np.ndarray) -> dict:
        """
        Runs the neural network on a single frame and returns the detection results.
        """
        # --- Performance Optimization ---
        # If the input frame is very high res (e.g. 1920x1080 or 4K), it creates millions of tensor 
        # parameters which instantly causes GPU Out-of-Memory (OOM) crashes.
        # We proportionally downscale any frame larger than 1280px to protect system stability.
        h, w = frame.shape[:2]
        max_dim = max(h, w)
        if max_dim > 1280:
            scale = 1280.0 / max_dim
            proc_frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        else:
            proc_frame = frame

        # OpenCV reads images in BGR format by default. Neural Networks expect RGB format.
        rgb = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2RGB)

        # Apply normalisation and convert to a 4D Tensor (BatchSize=1, Channels=3, Height, Width)
        input_tensor = self.transform(rgb).unsqueeze(0).to(self.device)

        # We don't need to track gradients (backpropagation) during inference, 
        # so we disable it to save massive amounts of RAM and CPU cycles.
        with torch.no_grad():
            if self.device.type == 'cuda':
                # Automatic Mixed Precision (AMP): Automatically casts certain math operations from 
                # 32-bit floats down to 16-bit floats. This nearly doubles inference speed on 
                # modern NVIDIA Tensor cores with no loss in accuracy.
                with torch.autocast(device_type='cuda'):
                    density_map = self.model(input_tensor)
            else:
                density_map = self.model(input_tensor)

        # The model returns a 2D density map. We must cast it back to standard float32,
        # because the OpenCV drawing functions in `annotate()` will crash if given float16 AMP arrays.
        density_np = density_map.squeeze().cpu().numpy().astype(np.float32)

        # The core logic of CSRNet: The total number of people is simply the sum of all pixels 
        # in the predicted density map.
        raw_count = float(density_np.sum())
        count = max(0, int(round(raw_count)))

        return {
            'count': count,
            'density_map': density_np,
            'smoothed_count': raw_count,
        }

    def annotate(self, frame: np.ndarray, detection_result: dict) -> np.ndarray:
        """
        Visually blends the invisible AI density map onto the original video frame.
        """
        annotated = frame.copy()
        density_map = detection_result['density_map']
        count = detection_result['count']

        # Normalise the raw mathematical density values to a 0.0 - 1.0 range so it can be visualised
        if density_map.max() > 0:
            norm_map = density_map / density_map.max()
        else:
            norm_map = density_map

        # The AI outputs a smaller heatmap than the original frame. Resize it back up.
        h, w = annotated.shape[:2]
        heatmap = cv2.resize(norm_map, (w, h))
        
        # Scale to 0-255 for standard 8-bit image processing
        heatmap = np.clip(heatmap * 255, 0, 255).astype(np.uint8)
        
        # Apply the 'Jet' colormap. This turns low values blue and high density "hotspots" red.
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # Alpha-blend the heatmap over the original frame. 
        # (60% opacity for the original camera feed, 40% opacity for the heatmap overlay).
        annotated = cv2.addWeighted(annotated, 0.6, heatmap_color, 0.4, 0)

        # Overlay the final count text
        info_text = f"Estimated Count: {count}"
        cv2.putText(annotated, info_text, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

        return annotated
