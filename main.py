import cv2
from deepface import DeepFace
import matplotlib.pyplot as plt

img = cv2.imread('imagens/12164629.jpg')
img = cv2.resize(img,(640,480))
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

resultado = DeepFace.analyze(img,actions=['emotion'])

print(resultado[0]['dominant_emotion'])

plt.imshow(img)