import numpy as np
from deepface import DeepFace
img = np.zeros((480, 640, 3), dtype=np.uint8)
try:
    res = DeepFace.analyze(img_path=img, actions=['emotion'], enforce_detection=False, detector_backend='ssd', silent=True)
    print('SSD WORKS:', getattr(res[0], 'get', lambda k: None)('face_confidence'))
except Exception as e:
    print('SSD FAIL:', e)
