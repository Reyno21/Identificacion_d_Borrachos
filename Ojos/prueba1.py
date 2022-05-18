import cv2
import numpy as np

#read image
img = cv2.imread("Arturo.jpg") # pass the name of image to be read
res = cv2.resize(img, dsize=(610, 620), interpolation=cv2.INTER_CUBIC)
outImage = res.copy()

# Load HAAR cascade
eyesCascade = cv2.CascadeClassifier("C://Users//Reyno21//OneDrive//Escritorio//Identificacion_d_Borrachos//haarcascade_eye.xml")

#------------ Detect Eyes ------------#
# eyeRects contain bounding rectangle of all detected eyes

eyeRects = eyesCascade.detectMultiScale(res , 1.1, 5 )

#Iterate over all eyes to remove red eye defect

for x,y,w,h in eyeRects:

    #Crop the eye region
    eyeImage = res [y:y+h , x:x+w]

    #split the images into 3 channels
    b, g ,r = cv2.split(eyeImage)

    # Add blue and green channels
    bg = cv2.add(b,g)

    #threshold the mask based on red color and combination ogf blue and gree color
    mask = ( (r>(bg-20)) & (r>80) ).astype(np.uint8)*255
    
#find all contours

contours, _ = cv2. findContours(mask.copy() ,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE )  # It return contours and Hierarchy

    #find contour with max Area
maxArea = 0
maxCont = None
for cont in contours:
    area = cv2.contourArea(cont)
    if area > maxArea:
        maxArea = area
        maxCont = cont
mask = mask * 0  # Reset the mask image to complete black image
    # draw the biggest contour on mask
cv2.drawContours(mask , [maxCont],0 ,(255),-1 )
    #Close the holes to make a smooth region
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_DILATE,(5,5)) )
mask = cv2.dilate(mask , (3,3) ,iterations=3)

# Stack both input and output image horizontally
result = np.hstack((res,outImage))
#Display the Result
cv2.imshow("RedEyeCorrection" , result )
cv2.waitKey() # Wait for a keyPress
cv2.destroyAllWindows()