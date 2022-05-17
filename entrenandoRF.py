#Codigo que entrenara el reconocimiento de rostros mediante las fotografias tomadas en captura

#Importamos las librerias necesarias, open cv2, os y numpy abreviada como np
import cv2
import os
import numpy as np 

#La ruta donde estan las carpetas con las capturas de los rostros
dataPath = './Imagenes'
#Busca la lista de carpetas creadas en captura
peopleList = os.listdir(dataPath)
#Imprime en consola la lista de las personas
print('Lista de personas', peopleList)

labels = []
faceData = []
label = 0

#Metodo que recorre las carpetas y las imagenes para su futuro entrenamiento
for nameDir in peopleList:
    #Creamos nuestra ruta de las imagenes
    personPath = dataPath + '/' + nameDir
    print('Leyendo Imagenes')

    #Metodo que recorre todas las imagenes de cada carpeta
    for fileName in os.listdir(personPath):
        print('Rostros', nameDir + '/' + fileName)
        labels.append(label)
        faceData.append(cv2.imread(personPath + '/' + fileName))
    label = label + 1

#Metodos de entrenamiento que podemos usar 
#face_recognizer = cv2.face.EigenFaceRecognizer_create()
face_recognizer = cv2.face.FisherFaceRecognizer_create()
#face_recognizer = cv2.face.LBPHFaceRecognizer_create()

#Entrenar en el reconocimiento
face_recognizer.train(faceData, np.array(labels))

#Almacena el modelo
#face_recognizer.write('modeloEigenFace.xml')
face_recognizer.write('modeloFisherFace.xml')
#face_recognizer.write('modeloLBPHFace.xml')