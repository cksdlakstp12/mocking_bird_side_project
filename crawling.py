from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time
import requests
from PIL import Image
from io import BytesIO
import os

def download_images(keyword, num_images):
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service)
    
    driver.get('https://images.google.com')
    
    search_box = driver.find_element(By.NAME, 'q')
    search_box.send_keys(keyword)
    search_box.send_keys(Keys.RETURN)
    
    SCROLL_PAUSE_TIME = 1
    last_height = driver.execute_script("return document.body.scrollHeight")

    while num_images > 0:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(SCROLL_PAUSE_TIME)
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height
    
    images = driver.find_elements(By.CLASS_NAME, "YQ4gaf")
    count = 0

    if not os.path.exists(keyword):
        os.makedirs(keyword)

    for image in images:
        if count >= num_images:
            break
        try:
            image_id = image.get_attribute("id")
            if image_id and image_id.startswith("dimg_"):
                image.click()
                time.sleep(2)  # 이미지가 로드될 시간을 충분히 줌
                img_url = driver.find_element(By.XPATH, '/html/body/div[7]/div/div/div/div/div/div/c-wiz/div/div[2]/div[2]/div/div[2]/c-wiz/div/div[3]/div[1]/a/img[1]').get_attribute("src")
                print(img_url)
                if 'http' in img_url:
                    img_data = requests.get(img_url).content
                    img = Image.open(BytesIO(img_data))
                    img.save(f"{keyword}/{keyword}_{count + 1}.jpg")
                    count += 1
        except Exception as e:
            print(f"Error downloading image {count + 1}: {e}")

    driver.quit()

# 예제 실행
download_images('sunset', 10)
