#Codigo para el reconocimiento de rostros en tiempo real

#Importamos las librerias que vamos a usar en este caso open cv2 y os
import cv2
import os

#La ruta donde estan las carpetas con las capturas de los rostros
dataPath = './Imagenes'
#Busca la lista de carpetas creadas en captura
peopleList = os.listdir(dataPath)
#Imprime en consola la lista de las personas
print('Lista de personas' , peopleList)

#Metodo de entrenamiento
face_recognizer = cv2.face.FisherFaceRecognizer_create()

#lectura del modelo
face_recognizer.read('modeloFisherFace.xml')

#Abrimos la captura de video
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

#Obtenemos la ruta y guardamos en una variable el archivo haarcascade que nos ayudara con el reconocimiento de rostros
d='C://Users//HP//OneDrive//Escritorio//8vo Semestre//TMPI//2do Parcial//'
faceClassif = cv2.CascadeClassifier(d +'haarcascade_frontalface_default.xml')

#Mientras la captura de video este abierta
while True:
    ret,frame = cap.read()
    if ret == False: break
    #Pasamos el frame a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #Creamos una copia del frame con la escala de grises
    auxFrame = gray.copy()
    #Detectar marcos en diferentes tamaños 
    faces = faceClassif.detectMultiScale(gray,1.3,5)
    #Ciclo para cada rostro
    for (x,y,w,h) in faces:
        rostro = auxFrame[y:y+h,x:x+w]
        #Damos una medida
        rostro = cv2.resize(rostro,(150,150),interpolation= cv2.INTER_CUBIC)
        #Reconocemos
        result = face_recognizer.predict(rostro)
        #Colocamos el texto que sera el nombre de la persona reconocida
        cv2.putText(frame,'{}'.format(result),(x,y-5),1,1.3,(255,255,0),1,cv2.LINE_AA)
        
        #Modelo que usas ademas de si no se reconoce la persona se coloca "Desconocido"
        if result[1] < 70: 
            cv2.putText(frame,'{}'.format(peopleList[result[0]]), (x, y-25), 2,1.1,(0,0,255), 1, cv2.LINE_AA)
            cv2.rectangle(frame, (x,y) ,(x+w, y+h), (0,0,255), 2)
        else:
            cv2.putText(frame,'Desconocido', (x, y-20), 2,0.8,(0,0,255), 1, cv2.LINE_AA)
            cv2.rectangle(frame, (x,y) ,(x+w, y+h), (0,0,255), 2)

    #Mostramos el frame
    cv2.imshow('frame', frame)
    #Cerramos el programa al precionar la tecla k o ESC
    k = cv2.waitKey(1)
    if k==27:
        break

cap.release()
cv2.destroyAllWindows()