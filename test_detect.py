import cv2
import numpy as np
from deepface import DeepFace
img = np.zeros((480, 640, 3), dtype=np.uint8)
res = DeepFace.analyze(img_path=img, actions=['emotion'], enforce_detection=False, detector_backend='opencv', silent=True)
data = res[0] if isinstance(res, list) else res
print('face_conf:', data.get('face_confidence'), 'region:', data.get('region'))
