"""
CrowdDetector — High-Level Interface for CSRNet.

This module acts as the bridge between raw OpenCV video frames and the raw PyTorch neural network.
It handles all the complex data transformations required before and after AI inference.
"""
import os  # PURPOSE: File system operations - checks if model weights file exists, handles file paths
import cv2  # PURPOSE: OpenCV - resizes frames to fit GPU memory, applies colormaps for heatmap visualization, draws annotations
import torch  # PURPOSE: PyTorch deep learning framework - loads CSRNet model, runs inference on GPU/CPU, manages tensors
import numpy as np  # PURPOSE: NumPy arrays - stores frame data, heatmap calculations, array operations for image processing
from torchvision import transforms  # PURPOSE: Image preprocessing pipeline - converts numpy arrays to tensors, applies ImageNet normalization
from .csrnet_model import CSRNet  # PURPOSE: CSRNet architecture - the core neural network for crowd density estimation


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
        # torch.device: Automatically detects NVIDIA GPU with CUDA drivers or falls back to CPU
        # GPU inference is 10-50x faster than CPU, critical for real-time processing
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
        
        # torch.load: Loads the massive pre-trained weight matrix from disk into memory
        # map_location ensures weights are moved to correct device (GPU or CPU)
        # weights_only=False allows loading custom model architectures (not just weights)
        if os.path.isfile(model_path):  # os.path.isfile: Checks if model weights file exists
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

        # model.to(device): Moves all model parameters to GPU VRAM for fast inference
        # model.eval(): Disables training-only features (Dropout, BatchNorm) for inference
        self.model.to(self.device)
        self.model.eval()

        # transforms.Compose: Creates a preprocessing pipeline for converting frames
        # Each step transforms the data sequentially (frame → tensor → normalized tensor)
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # Converts numpy BGR array to PyTorch tensor (0-1 range)
            transforms.Normalize(mean=self.MEAN, std=self.STD),  # Applies ImageNet normalization
        ])

    def detect(self, frame: np.ndarray) -> dict:
        """
        Runs the neural network on a single frame and returns the detection results.
        """
        # --- Performance Optimization ---
        # cv2.resize(): Proportionally downscales large frames to prevent GPU Out-of-Memory (OOM) crashes
        # If the input frame is very high res (e.g. 1920x1080 or 4K), it creates millions of tensor 
        # parameters which instantly causes GPU memory exhaustion
        # We downscale any frame larger than 1280px to protect system stability
        h, w = frame.shape[:2]
        max_dim = max(h, w)
        if max_dim > 1280:
            scale = 1280.0 / max_dim
            proc_frame = cv2.resize(frame, (int(w * scale), int(h * scale)))  # Maintains aspect ratio
        else:
            proc_frame = frame

        # cv2.cvtColor(frame, cv2.COLOR_BGR2RGB): Converts OpenCV's BGR format to PyTorch's RGB format
        # CRITICAL: CSRNet was trained on ImageNet RGB images; BGR channels would corrupt predictions
        rgb = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2RGB)

        # transforms.Compose pipeline: Converts numpy RGB array to normalized PyTorch tensor
        # .unsqueeze(0): Adds batch dimension (1, 3, H, W) so model can process it
        # .to(device): Moves tensor to GPU VRAM for fast computation
        input_tensor = self.transform(rgb).unsqueeze(0).to(self.device)

        # torch.no_grad(): Disables automatic differentiation tracking during inference
        # Saves massive amounts of GPU VRAM and CPU cycles since we don't need gradients for backpropagation
        with torch.no_grad():
            if self.device.type == 'cuda':
                # torch.autocast('cuda'): Automatic Mixed Precision - uses float16 for matrix operations
                # Speeds up inference ~2x on modern NVIDIA GPUs (Tensor cores) with negligible accuracy loss
                with torch.autocast(device_type='cuda'):
                    density_map = self.model(input_tensor)  # Runs CSRNet forward pass
            else:
                density_map = self.model(input_tensor)

        # Convert PyTorch tensor back to NumPy array for post-processing
        # .squeeze(): Removes batch dimension
        # .cpu(): Moves from GPU VRAM back to RAM for NumPy operations
        # .numpy(): Converts PyTorch tensor to NumPy array
        # .astype(np.float32): Ensures OpenCV functions can process the array (float16 AMP would crash cv2.resize)
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
        Uses OpenCV and NumPy to transform raw AI predictions into intuitive visual heatmaps.
        """
        annotated = frame.copy()
        density_map = detection_result['density_map']
        count = detection_result['count']

        # NumPy normalization: Scales raw density values to 0.0-1.0 range for visualization
        # density_map.max(): Finds maximum density value (e.g., 150.5)
        # Dividing by max normalizes all values to 0-1 range for standard visualization
        if density_map.max() > 0:
            norm_map = density_map / density_map.max()
        else:
            norm_map = density_map

        # cv2.resize(): Upscales the smaller heatmap to match original frame dimensions
        # CSRNet outputs smaller heatmaps for efficiency; must resize to overlay on original frame
        h, w = annotated.shape[:2]
        heatmap = cv2.resize(norm_map, (w, h))
        
        # NumPy clipping and type conversion: Scales 0-1 range to 0-255 for 8-bit images
        # np.clip(): Ensures all values stay within [0, 255] range
        # .astype(np.uint8): Converts float32 to unsigned 8-bit integer (standard for images)
        heatmap = np.clip(heatmap * 255, 0, 255).astype(np.uint8)
        
        # cv2.applyColorMap(): Applies color mapping (Jet: blue→green→yellow→red)
        # Blue = low density (safe), Red = high density (crowded)
        # Makes heatmap visually intuitive for operators without training
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # Alpha-blend the heatmap over the original frame. 
        # (60% opacity for the original camera feed, 40% opacity for the heatmap overlay).
        annotated = cv2.addWeighted(annotated, 0.6, heatmap_color, 0.4, 0)

        # Overlay the final count text
        info_text = f"Estimated Count: {count}"
        cv2.putText(annotated, info_text, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

        return annotated
