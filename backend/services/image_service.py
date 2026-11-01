import base64
import cv2
import numpy as np

class ImageService:
    @staticmethod
    def decode_base64_image(base64_string: str):
        """
        Decodes a base64 string into an OpenCV image (BGR).
        """
        try:
            # Remove header if present (e.g., "data:image/jpeg;base64,")
            if "," in base64_string:
                base64_string = base64_string.split(",")[1]
            
            img_data = base64.b64decode(base64_string)
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return img
        except Exception as e:
            print(f"Error decoding image: {e}")
            return None

    @staticmethod
    def preprocess_for_local_model(img, img_size=96):
        """
        Preprocessing logic based on the training notebook:
        1. Gaussian Blur (3x3)
        2. Median Blur (3)
        3. Resize to target size
        4. Normalize (0-1)
        5. Convert to RGB
        """
        if img is None:
            return None
        
        try:
            # Convert to RGB (OpenCV uses BGR by default)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Gaussian smoothing
            img_smooth = cv2.GaussianBlur(img_rgb, (3, 3), 0)
            
            # Median filtering
            img_filtered = cv2.medianBlur(img_smooth, 3)
            
            # Resize
            img_resized = cv2.resize(img_filtered, (img_size, img_size))
            
            # Normalize
            img_normalized = img_resized.astype("float32") / 255.0
            
            return img_normalized
        except Exception as e:
            print(f"Error preprocessing image: {e}")
            return None
