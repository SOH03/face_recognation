import cv2
import face_recognition
import numpy as np
import os

path=r"F:\0PENCV\pic"
images=[]
classes=[]
pathlist=os.listdir(path=path)
print(pathlist)
for cls in pathlist:
    img=cv2.imread(f"{path}\{cls}")
    images.append(img)
    classes.append(os.path.splitext(cls)[0])
    print(classes)
def encodings(images):
    encodes=[]
    for img in images:
        cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode=face_recognition.face_encodings(img)[0]
        encodes.append(encode)
    return encodes
encodelist=encodings(images)

cap=cv2.VideoCapture(0)
while True:
    ret,frame=cap.read()
    imgs=cv2.resize(frame,(0,0),None,fx=0.25,fy=0.25)
    imgs=cv2.cvtColor(imgs,cv2.COLOR_BGR2RGB)
    frame=cv2.flip(frame,1)

    faceloc=face_recognition.face_locations(imgs)
    frameencode=face_recognition.face_encodings(imgs,faceloc)

    for enc,loc in zip(frameencode,faceloc):
        mat=face_recognition.compare_faces(encodelist,enc)
        dis=face_recognition.face_distance(encodelist,enc)
        ind=np.argmin(dis)
        if mat[ind]:
            name=classes[ind]
            print(name)
            y1,x2,y2,x1=loc
            y1,x2,y2,x1=y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,255),2) 
            cv2.rectangle(frame,(x1,y2-35),(x2,y2),(255,0,255),-1) 
            cv2.putText(frame,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),1) 
     
    cv2.imshow("webcam",frame) 
 
    if cv2.waitKey(1) & 0xFF==ord("q"): 
        break 
 
cap.release() 
cv2.destroyAllWindows()
    
    