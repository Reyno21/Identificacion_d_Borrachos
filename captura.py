#Codigo que nos ayudara a obtener las imagenes (fotografias necesarias) 
#de una persona mediante un video para su reconocimiento y guardarlo en una carpeta. 

#Importamos las librerias necesarias, open cv2, os e imutils
import cv2
import os
import imutils

#Asignamos el nombre de la carpeta que se va a crear 
pnombre ='Naarai'
#La ruta donde se va a crear la carpeta
dataPath = './Imagenes'
#Guardamos la ruta y el nombre de nuestra carpeta concatenados
pPath = dataPath + '/' + pnombre

#Validamos si existe el nombre de la carpeta, sino existe crearemos una nueva carpeta
if not os.path.exists(pPath):
    print('Carpeta creada: ' ,pPath)
    os.makedirs(pPath)

#Comenzamos a capturar el video en tiempo real.
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

#Obtenemos la ruta y guardamos en una variable el archivo haarcascade que nos ayudara con el reconocimiento de rostros
d='C://Users//HP//OneDrive//Escritorio//8vo Semestre//TMPI//2do Parcial//'
faceClassif = cv2.CascadeClassifier(d +'haarcascade_frontalface_default.xml')

#Creamos un contador
count = 0

#Mientras la camara este abierta 
while True:
    ret, frame = cap.read()
    if ret == False: break
    #Asiganmos un nuevo tamaño al frame 
    frame = imutils.resize(frame, width = 640)
    #Pasamos a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #Creamos una copia del frame
    auxFrame = frame.copy()

    #Detectar marcos en diferentes tamaños 
    faces = faceClassif.detectMultiScale(gray, 1.3,5)

    #Ciclo para cada rostro
    for (x,y,w,h) in faces:
        #Encerramos el rostro en un rectangulo en el frame
        cv2.rectangle(frame, (x,y), (x+w,y+h),(0,255,0),2)
        rostro = auxFrame[y:y+h, x: x +w]
        rostro = cv2.resize(rostro,(150,150),interpolation=cv2.INTER_CUBIC)
        #Guardamos la captura del rostro en la carpeta
        cv2.imwrite(pPath + '/rostro_{}.jpg'.format(count),rostro)
        #Incrementamos la imagen en 1
        count = count +1

    #Mostramos la pantalla
    cv2.imshow('frame', frame)

    #Para salir presionamos una tecla en este caso ESC o esperamos a que tome 300 fotos
    k= cv2.waitKey(1)
    if k==27 or count >= 300:
        break

cap.release()
cv2.destroyAllWindows()