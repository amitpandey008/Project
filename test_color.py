import cv2
import urllib.request
import numpy as np
from deepface import DeepFace
url = 'https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg'
req = urllib.request.urlopen(url)
arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
for bk in ['mtcnn', 'retinaface']:
 for name, img in [('BGR', bgr), ('RGB', rgb)]:
  res = DeepFace.analyze(img_path=img, actions=['emotion'], enforce_detection=False, detector_backend=bk, silent=True)
  conf = res[0]['face_confidence'] if isinstance(res, list) else res['face_confidence']
  print(f'[{bk}] {name} conf: {conf}')
