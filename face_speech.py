import os
import cv2
import numpy as np
import face_recognition as fr
import pyttsx3

path = 'Images'
my_list = os.listdir(path)

imgs = []
class_names = []
for i in my_list:
    img_path = os.path.join(path, i)
    img_r = cv2.imread(img_path)
    imgs.append(img_r)
    class_names.append(i.split('.')[0])
    
def encode_face(images):
    encoded_list = []
    for i in imgs:
        imgg = cv2.cvtColor(i, cv2.COLOR_BGR2RGB)
        face_loc = fr.face_locations(imgg)
        if len(face_loc) == 0:
            print("No face found in one image")
            continue
        face_encode = fr.face_encodings(imgg, face_loc)[0]
        encoded_list.append(face_encode)
    return encoded_list

encoded_known_face = encode_face(imgs)

txt_sp=pyttsx3.init()
voices=txt_sp.getProperty('voices')
txt_sp.setProperty('voice',voices[1].id)
txt_sp.setProperty('volume',1.0)

video=cv2.VideoCapture(0)
while True:
    sucess,img=video.read()
    if sucess==False:
        break
    img_1=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    face_loc_v=fr.face_locations(img_1)
    face_enc_v=fr.face_encodings(img_1,face_loc_v)
    for enc,loc in zip(face_enc_v,face_loc_v):
        matches=fr.compare_faces(encoded_known_face,enc)
        face_dis=fr.face_distance(encoded_known_face,enc)
        match_index=np.argmin(face_dis)
        name='unknown'
        if matches[match_index]==True:
            name=class_names[match_index]
            print(name)
        txt_sp.say(name)
        txt_sp.runAndWait()
            
        y1,x2,y2,x1=loc
        cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),2)
        cv2.putText(img,name,(x1,y1),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
        
    cv2.imshow('Frame',img)
    if cv2.waitKey(1)&0XFF==ord('q'):
        break
video.release()
cv2.destroyAllWindows()