import numpy as np
import cv2

#Obtenemos la imagen de video
cap = cv2.VideoCapture(0)

#Deteccion de color 
#Rango bajo de color rojo
redBajo = np.array([0,200,9], np.uint8)
redBajo1 = np.array([175,100,20], np.uint8)

#Rango alto de color rojo
redAlto = np.array([8, 255, 255], np.uint8)
redAlto1 = np.array([179,255,255], np.uint8)

while(True):
    ret, frame = cap.read()
    
    if ret == True:
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
                #transformar de bgr a hsv
                frameSHV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                #Mascara permite la identificacion de colores
                maskRed = cv2.inRange(frameSHV, redBajo, redAlto)
                #Combinacion de ellos para poder visualizar la imagen normal y la imagen del color
                maskRedvis = cv2.bitwise_and(frame, frame, mask = maskRed)
                cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (0,255,0), 2) #Dibujar un rectangulo en los ojos
            #Muestra la infor generada
            cv2.imshow('maskRedvis', maskRedvis) #Tiempo real
            cv2.imshow('maskRed', maskRed) #El twice
            cv2.imshow('Video', frame) #Visualizacion de ambas
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else: break

cap.release()
cv2.destroyAllWindows()