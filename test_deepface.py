import cv2
import urllib.request
import numpy as np
from deepface import DeepFace
url = 'https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg'
req = urllib.request.urlopen(url)
arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
print('Testing BGR:')
try:
    res_bgr = DeepFace.analyze(img_path=img_bgr, actions=['emotion'], enforce_detection=True, detector_backend='opencv', silent=True)
    print(' BGR Face Detected! Emotion:', res_bgr[0]['dominant_emotion'])
except Exception as e:
    print(' BGR Failed:', e)
print('Testing RGB:')
try:
    res_rgb = DeepFace.analyze(img_path=img_rgb, actions=['emotion'], enforce_detection=True, detector_backend='opencv', silent=True)
    print(' RGB Face Detected! Emotion:', res_rgb[0]['dominant_emotion'])
except Exception as e:
    print(' RGB Failed:', e)
