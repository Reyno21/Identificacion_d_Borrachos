import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()
    
    face_cascade = cv2.CascadeClassifier('C:/Users/HP/OneDrive/Escritorio/8vo Semestre/TMPI/2do Parcial/haarcascade_frontalface_default.xml') #Cargar el archivo de cascada para reconocer la cara
    eye_casade = cv2.CascadeClassifier('C:/Users/HP/OneDrive/Escritorio/8vo Semestre/TMPI/2do Parcial/haarcascade_eye.xml') #Cargar el archivo de cascada para reconocer los ojos

    img = frame #Cargar la imagen

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #Convertir la imagen a escala de grises

    face = face_cascade.detectMultiScale(gray, 1.3, 5) #Detectar la cara

    for (x,y,w,h) in face: #Recorrer el arreglo de caras
        cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2) #Dibujar un rectangulo en la cara
        roi_gray = gray[y:y+h, x:x+w] #Recortar la cara
        roi_color = img[y:y+h, x:x+w] #Recortar la cara
        eyes = eye_casade.detectMultiScale(roi_gray) #Detectar los ojos
        for (ex,ey,ew,eh) in eyes: #Recorrer el arreglo de ojos
            cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (0,255,0), 2) #Dibujar un rectangulo en los ojos

    cv2.imshow('img', img) #Mostrar la imagen
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()