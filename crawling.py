from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options

import time
import requests
from PIL import Image
from io import BytesIO
import os

def download_images(save_dir, keyword, num_images):
    # chrome_options = Options()
    # chrome_options.add_argument('--headless')
    # chrome_options.add_argument('--disable-gpu')
    
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service)
    # driver = webdriver.Chrome(service=service, options=chrome_options)
    
    driver.get('https://www.google.com/imghp?hl=kr')
    
    search_box = driver.find_element(By.NAME, 'q')
    search_box.send_keys(keyword)
    search_box.send_keys(Keys.RETURN)
    
    SCROLL_PAUSE_TIME = 2
    last_height = driver.execute_script("return document.body.scrollHeight")

    try:
        see_more_button = driver.find_element(By.XPATH, '//*[@id="rso"]/div/div/div[2]/div[2]/div[4]/div[2]/a')
        if see_more_button:
            see_more_button.click()
            time.sleep(2)
    except Exception as e:
        print("No 'See more' button found or could not click it:", e)

    while True:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(SCROLL_PAUSE_TIME)
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height
    
    images = driver.find_elements(By.CSS_SELECTOR, "div.H8Rx8c")
    count = 0

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for image in images:
        if count >= num_images:
            break
        try:
            image.click()
            time.sleep(2)
            
            img_xpath = '//*[@id="Sva75c"]/div[2]/div[2]/div/div[2]/c-wiz/div/div[3]/div[1]/a/img[1]'
            img_element = driver.find_element(By.XPATH, img_xpath)
            img_url = img_element.get_attribute("src")

            if img_url:
                img_data = requests.get(img_url).content
                img = Image.open(BytesIO(img_data))
                img.save(os.path.join(save_dir, f"{count + 1}.jpg"))
                count += 1
        except Exception as e:
            print(f"Error downloading image {count + 1}: {e}")

    driver.quit()

if __name__ == "__main__":
    download_images('data/fu', 'Dalong Fu actor', 500)
