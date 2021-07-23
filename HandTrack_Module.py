import cv2
import mediapipe as mp
import time
from mediapipe.framework.formats.landmark_pb2 import LandmarkList

from mediapipe.python.solutions.drawing_utils import draw_landmarks

class handDetect():
    def __init__(self,mode = False,maxHands = 2, detectConf = 0.6534, trackConf = 0.69541) :
        self.mode = mode
        self.maxHands = maxHands
        self.minDetectConf = detectConf
        self.minTrackConf = trackConf

        #prepare hand model 
        self.mpHands =  mp.solutions.hands
        self.hand =  self.mpHands.Hands(self.mode,self.maxHands,
                                        self.minDetectConf,self.minTrackConf)
        #prepare to draw
        self.mpDraw =  mp.solutions.drawing_utils

        self.tipID = [4,8,12,16,20]

    def lookForHands(self,cam,draw = True):
        self.imgRGB =  cv2.cvtColor(cam,cv2.COLOR_BGR2RGB)
        self.process = self.hand.process(self.imgRGB)
        #print(process.multi_hand_landmarks)

        if(self.process.multi_hand_landmarks):
            for handLms in self.process.multi_hand_landmarks :
                if (draw == True):
                    self.mpDraw = draw_landmarks(cam,handLms,self.mpHands.HAND_CONNECTIONS)

        return cam    

    def lookForPosition(self,cam,handNo = 0, draw = True):
        
        self.LandmarkList = []
        if self.process.multi_hand_landmarks:
            myHand = self.process.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                #print(id,lm)
                    h, w , c = cam.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    self.LandmarkList.append([id,cx,cy])
                    
                    if(id == 4 or id == 8 or id == 12 or id == 16 or id == 20 and draw == True):
                        cv2.circle(cam, (cx,cy), 10, (255,0,255), cv2.FILLED)
    
        return self.LandmarkList

    def lookForFinger(self):
        finger = []

        if(self.LandmarkList [self.tipID [0]][1] < self.LandmarkList[self.tipID[0]-1][1]):
            finger.append(1)
        else:
            finger.append(0)

        for id in range(1,5):
            if(self.LandmarkList [self.tipID[id]][2] < self.LandmarkList[self.tipID[id]-2][2]):
                finger.append(1)
            else :
                finger.append(0)
        return finger

def main():
    pastTime = 0
    currentTime = 0
    vid = cv2.VideoCapture(0)
    detect = handDetect()

    while(True):
        working, cam = vid.read()
        cam = detect.lookForHands(cam)
        LandmarkList = detect.findCoordinate(cam)
        #if len(LandmarkList) != 0:
            #print(LandmarkList[])

        currentTime = time.time()
        fps = 1/(currentTime - pastTime)
        pastTime = currentTime

        cv2.putText(cam,str(int(fps)),(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
        cv2.imshow('openCV', cam)
    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release()
    cv2.destroyAllWindows()
        



if __name__ == "__main__":
    main()

