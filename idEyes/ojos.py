import cv2

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

while True:
    #SE REALIZA LA LECTURA DE LA CAMARA
    ret, frame = cap.read()
    if ret == False:
        break

    #se extrae el ancho y alto
    al, an, c = frame.shape

    #tomar el centro de la imagen
    x1 = int(an / 3)
    x2 = int(x1 * 2)

    y1 = int(al / 3)
    y2 = int(y1 * 2)

    cv2.putText(frame, 'ponga el ojaso en el rectangulo prro', (x1 - 50, y1 - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255 , 0),2)

    cv2.rectangle(frame, (x1, y1) , (x2, y2), (0, 255, 0), 2)
    
    recorte = frame[y1:y2, x1:x2]

    gris = cv2.cvtColor(recorte, cv2.COLOR_BGR2GRAY)

    gris = cv2.GaussianBlur(gris, (3, 3), 1)

    _, umbral = cv2.threshold(gris, 7, 255, cv2.THRESH_BINARY_INV)

    contornos, _ = cv2.findContours(umbral, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contornos = sorted(contornos, key=lambda x: cv2.contourArea(x), reverse=True)

    for contorno in contornos:
        (x, y, ancho, alto) = cv2.boundingRect(contorno)

        cv2.rectangle(frame, (x + x1, y + y1), (x+ ancho + x1, y + alto + y1), (0, 255, 0), 1)

        cv2.line(frame , (x1 + x + int(ancho/2), 0), (x1 + x + int(ancho/2), al), (0,255,0), 1)

        cv2.line(frame, (0, y1 + y + int(ancho/2)), (an, y1 + y ++ int(ancho/2)), (0,255,0), 1)
        break

    #se muestra el recorte en gris
    cv2.imshow("ojasos", frame)

    #se muestra el recorte
    cv2.imshow("recorte", recorte)

    #mostramos el umbral
    cv2.imshow("umbral", umbral)

    t = cv2.waitKey(1)

    if t == 27:
        break

cap.release()
cv2.destroyAllWindows()