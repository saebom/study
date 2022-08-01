from urllib.request import urlretrieve
from selenium import webdriver


#Selenium으로 이미지 크롤링
import os

# path = 'C:/chromedriver_win32/chromedriver.exe' #웹드라이버가 있는 경로
# driver = webdriver.Chrome(path) #웹드라이버가 있는 경로에서 Chrome을 가져와 실행

options = webdriver.ChromeOptions() 
options.add_experimental_option("excludeSwitches", ["enable-logging"])
driver = webdriver.Chrome(options=options, 
                          executable_path=r'C:/chromedriver_win32/chromedriver.exe') 


#img 다운_폴더지정 또는 생성
img_folder_path = 'D:/study_data/_project1/img/2019FW/milan/bottega_veneta/'   #이미지 저장 폴더
if not os.path.isdir(img_folder_path):
    os.mkdir(img_folder_path)
    
    
#vogue.com 접속
site_path = "https://www.vogue.com/fashion-shows/fall-2019-ready-to-wear/bottega-veneta#gallery-collection"
driver.get(site_path)



#<road more> 버튼 해제하여 road함
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

try:
    buttons = WebDriverWait(driver, 10).until(
        EC.visibility_of_all_elements_located((By.XPATH, 
                                               "//div[@class='ContentWithCTAWrapper-rfTCP elyUNQ']\
                                               //button[@class='BaseButton-aqvbG ButtonWrapper-dOIhno mrDWN dupjQb button button--primary-pair GalleryThumbnailExpandButton-fxnwEB eMgbgA']")))
                                                # 사진이미지와 button이 있는 class와 button의 class를 입력해 줌
except TimeoutException:
    print("no read more")
else:
    for button in buttons:
        button.click()
     
        
#이미지 elements path 잡아 줌 
imgs = driver.find_elements(By.XPATH, '//*[@id="gallery-collection"]/div/div[1]/div/div/a/figure/span/picture/img')
result = []
img_path = '/html/body/div[1]/div/div/main/div[5]/div[1]/div/section[1]/div/div[1]/div/div[2]/a/figure/span/picture/img'


# 이미지 추출
for img in imgs:   
    print(img.get_attribute('src'))
    result.append(img.get_attribute('src')) 
  
     
# 이미지 다운로드
cnt = 0
for image in result:
    cnt += 1
    urlretrieve(image, img_folder_path + f'{cnt}.jpg')
print("Saved!") 
        
