import cv2
import urllib.request
import numpy as np
from deepface import DeepFace
url = 'https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg'
req = urllib.request.urlopen(url)
arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
for backend in ['opencv', 'mtcnn', 'retinaface', 'yunet', 'ssd']:
    try:
        res = DeepFace.analyze(img_path=img_bgr, actions=['emotion'], enforce_detection=True, detector_backend=backend, silent=True)
        data = res[0] if isinstance(res, list) else res
        print(f'[{backend}] face_conf: {data.get("face_confidence")}, region: {data.get("region")}')
    except Exception as e:
        print(f'[{backend}] failed: {e}')
