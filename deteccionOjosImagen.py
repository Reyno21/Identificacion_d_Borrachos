#id Eyes Face 
#Detecta solo de frente
import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('C:/Users/HP/OneDrive/Escritorio/8vo Semestre/TMPI/2do Parcial/haarcascade_frontalface_default.xml') #Cargar el archivo de cascada para reconocer la cara
eye_casade = cv2.CascadeClassifier('C:/Users/HP/OneDrive/Escritorio/8vo Semestre/TMPI/2do Parcial/haarcascade_eye.xml') #Cargar el archivo de cascada para reconocer los ojos
#face_cascade = cv2.CascadeClassifier('C://Users//Reyno21//OneDrive//Escritorio//Identificacion_d_Borrachos//haarcascade_frontalface_default.xml') #Cargar el archivo de cascada para reconocer la cara
#eye_casade = cv2.CascadeClassifier("C://Users//Reyno21//OneDrive//Escritorio//Identificacion_d_Borrachos//haarcascade_eye.xml")

img = cv2.imread('Arturo.jpg')
res = cv2.resize(img, dsize=(610, 620), interpolation=cv2.INTER_CUBIC)
gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY) #Convertir la imagen a escala de grises

face = face_cascade.detectMultiScale(gray, 1.3, 5) #Detectar la cara

for (x,y,w,h) in face: #Recorrer el arreglo de caras
    cv2.rectangle(res, (x,y), (x+w, y+h), (255,0,0), 2) #Dibujar un rectangulo en la cara
    roi_gray = gray[y:y+h, x:x+w] #Recortar la cara
    roi_color = res[y:y+h, x:x+w] #Recortar la cara
    eyes = eye_casade.detectMultiScale(roi_gray) #Detectar los ojos
    for (ex,ey,ew,eh) in eyes: #Recorrer el arreglo de ojos
        cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (0,255,0), 2) #Dibujar un rectangulo en los ojos

cv2.imshow('img', res) #Mostrar la imagen
cv2.waitKey(0) #Esperar una tecla
cv2.destroyAllWindows() #Cerrar todas las ventanas