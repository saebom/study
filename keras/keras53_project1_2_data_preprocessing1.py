import os
import glob
import pandas

#파일 이름 변경하기
"""
file_path = 'D:/study_data/_project1/img/fashion_img/total_train/total/'
# file_path = 'D:/study_data/_project1/img/fashion_img/paris_val/'
file_names = os.listdir(file_path)
file_names

# file_names = glob.glob('D:/study_data/_project1/img/2019FW/london/2019FW_train/*.jpg')
# file_names.sort(key=os.path.getctime, reverse=False)
import natsort
file_names = natsort.natsorted(file_names, reverse=False)


i = 8137
# for f in file_names:
#     print(f)
#     src = os.path.join(file_names, f)
#     dst = str(i) + '.jpg'
#     dst = os.path.join(f, dst)
#     os.rename(src, dst)
#     i += 1

for name in file_names:
    src = os.path.join(file_path, name)
    dst = str(i) + '.jpg'
    dst = os.path.join(file_path, dst)
    os.rename(src, dst)
    i += 1
"""    

# 파일 사이즈 줄이기
import glob
from PIL import Image
files = glob.glob('D:/study_data/_project1/img/fashion_img/paris_train/*')    

for f in files:
    title, ext = os.path.splitext(f)
    if ext in ['.jpg', '.png']:
        img = Image.open(f)
        img_resize = img.resize((int(img.width/4), int(img.height/4)))
        img_resize.save(title + ext)


   