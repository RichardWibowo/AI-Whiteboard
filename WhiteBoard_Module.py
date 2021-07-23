import cv2
from mediapipe.framework.formats.landmark_pb2 import LandmarkList
import numpy as np
import os
import HandTrack_Module as HT
#######################
brushThickness = 15
eraserThickness = 100
########################


folderPath = "header"
myList = os.listdir(folderPath)
print(myList)


headerList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    headerList.append(image)
print(len(headerList))
header = headerList[0]

drawColor = (255, 0, 255)

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = HT.handDetect(detectConf=0.75,maxHands=1)
xp, yp = 0, 0
imgCanvas = np.zeros((720, 1280, 3), np.uint8)

while True:

    # 1. Import image
    success, img = cap.read()
    img = cv2.flip(img, 1)

    # 2. Find Hand Landmarks
    img = detector.lookForHands(img)
    lmList = detector.lookForPosition(img, draw=False)

    if len(lmList) != 0:

        # print(lmList)

        # tip of index and middle fingers
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        # 3. Check which fingers are up
        fingers = detector.lookForFinger()
        # print(fingers)

        # 4. If Selection Mode - Two finger are up
        if fingers[1] and fingers[2]:
            # xp, yp = 0, 0
            #print("Selection Mode")
            #cracking my head open yes thankyou
            if(y1 < 125):
                if(x1 >= 200 and x1 <= (200+100)):
                    #print("select color : red")
                    Header = headerList[0]
                    drawColor = (52, 64, 235)
                elif(x1 >= 350 and x1 <= (350+100)):
                   # print("Select Colour : yellow")
                    Header = headerList[1]
                    drawColor = (11,195,255)
                elif(x1 >= 500 and x1 <= (500 + 100)):
                    #print("select color : blue")
                    Header = headerList[2]
                    drawColor=(235, 64, 52)
                elif(x1 >= 1000 and x1 <=(1000+100)):
                    #print("erase")
                    activeHeader = headerList[3]
                    drawColor = (0,0,0)
            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)

        # 5. If Drawing Mode - Index finger is up
        if fingers[1] and fingers[2] == False:
            cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
            #print("Drawing Mode")
            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
            cv2.line(imgCanvas,(xp,yp), (x1,y1), drawColor,brushThickness)

            if drawColor == (0, 0, 0):
                cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
            #
            else:
                cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)

            xp, yp = x1, y1


        # # Clear Canvas when all fingers are up
        #if all (x >= 1 for x in fingers):
                #imgCanvas = np.zeros((720, 1280, 3), np.uint8)

    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv,cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img,imgInv)
    img = cv2.bitwise_or(img,imgCanvas)


    # Setting the header image
    img[0:125, 0:1280] = header
    # img = cv2.addWeighted(img,0.5,imgCanvas,0.5,0)
    cv2.imshow("Image", img)
    #cv2.imshow("Canvas", imgCanvas)
    #cv2.imshow("Inv", imgInv)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()