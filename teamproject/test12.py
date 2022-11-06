import cv2
import numpy as np
import os
print(cv2.__version__)  # 4.6.0


path = "C:\study/teamproject\LeeJungJae"
file_list = os.listdir(path)

file_list[0]
len(file_list)
print(len(file_list))   # 201

file_name_list = []

for i in range(len(file_list)):
    file_name_list.append(file_list[i].replace("", ""))
print(file_name_list)
print(file_name_list[0])
print(cv2.data.haarcascades)    #C:\Users\AIA\AppData\Roaming\Python\Python39\site-packages\cv2\data\

# def Cutting_face_save(image, name):
#     if img is None:
#         print('Wrong path: ', path)
#     else: 
#         face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#         # eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'Haarcascade_eye.xml')
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
#         for (x,y,w,h) in faces:
#             cv2.rectangle(image, (x,y), (x+w, y+h), (255,0,0), 2)
#             cropped = image[y:y+h, x:x+w]
#             resize = cv2.resize(cropped, (250,250))
#             # cv2.imshow("crop&resize", resize)
#             cv2.waitKey(0)
#             cv2.destroyAllWindows()
            
#             # 이미지 저장하기
#             cv2.imwrite(f"LeeJungJae/image/{name}.jpg", resize)

def Cutting_face_save(image, name):

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    # eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'Haarcascade_eye.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x,y,w,h) in faces:
        cv2.rectangle(image, (x,y), (x+w, y+h), (255,0,0), 2)
        cropped = image[y:y+h, x:x+w]
        resize = cv2.resize(cropped, (250,250))
        # cv2.imshow("crop&resize", resize)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # 이미지 저장하기
        cv2.imwrite(f"C:/study/teamproject/LeeJungJae/{name}", resize)
                    
for name in file_name_list:
    img = cv2.imread("C:/study/teamproject/LeeJungJae/"+name+"")
    print(img)
    Cutting_face_save(img, name)
