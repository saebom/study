import os

#파일 이름 변경하기

file_path = 'D:\study_data\_project1\img/fashion_img/01_paris_total\정제된 데이터'
file_names = os.listdir(file_path)
file_names


i = 1958
for name in file_names:
    src = os.path.join(file_path, name)
    dst = str(i) + '.jpg'
    dst = os.path.join(file_path, dst)
    os.rename(src, dst)
    i += 1
    

# 파일 사이즈 줄이기
# import glob
# from PIL import Image
# files = glob.glob('D:/study_data/_project1/img/fashion_img/00_paris_total/*')    

# for f in files:
#     title, ext = os.path.splitext(f)
#     if ext in ['.jpg', '.png']:
#         img = Image.open(f)
#         img_resize = img.resize((int(img.width/2), int(img.height/2)))
#         img_resize.save(title + ext)


   