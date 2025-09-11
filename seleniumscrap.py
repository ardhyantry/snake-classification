from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import urllib.request
import time
import os

# Daftar kata kunci
keywords = [
    "Naja sputatrix (Ular Kobra Jawa)",
    "Ophiophagus hannah (King Cobra)",
    "Bungarus candidus (Ular Weling)",
    "Python reticulatus (Sanca Kembang)",
    "Coelognathus radiatus (Ular Tikus)",
    "Dendrelaphis pictus (Ular Pucuk)"
]

# Konfigurasi Selenium
options = webdriver.ChromeOptions()
options.add_argument('--headless')  # Jalankan tanpa UI browser
options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
driver = webdriver.Chrome(options=options)

# Fungsi untuk mendownload gambar
def download_images(keyword, max_images=10):
    folder_name = keyword.replace(' ', '_').replace('(', '').replace(')', '')
    save_path = f'downloaded_images/{folder_name}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Buka Google Images
    driver.get('https://www.google.com/imghp')
    time.sleep(2)  # Tunggu halaman dimuat

    # Cari kata kunci
    try:
        search_box = driver.find_element(By.NAME, 'q')
        search_box.clear()
        search_box.send_keys(keyword)
        search_box.send_keys(Keys.ENTER)
        time.sleep(3)  # Tunggu hasil pencarian
    except Exception as e:
        print(f"Error searching for {keyword}: {e}")
        return

    # Scroll untuk memuat lebih banyak gambar
    for _ in range(3):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(5)  # Tunggu lebih lama untuk memastikan gambar dimuat

    # Ambil semua elemen gambar
    images = driver.find_elements(By.CSS_SELECTOR, 'img.YQ4gaf')  # Update selector
    print(f"Found {len(images)} images for {keyword}")

    # Download gambar
    for i, img in enumerate(images[:max_images]):
        try:
            img_url = img.get_attribute('src')
            print(f"Found URL for {keyword}: {img_url}")
            if img_url and 'http' in img_url and not img_url.startswith('data:image'):
                urllib.request.urlretrieve(img_url, f'{save_path}/image_{i}.jpg')
                print(f"Downloaded {keyword} - image_{i}.jpg")
            else:
                print(f"Skipping invalid URL for {keyword}: {img_url}")
        except Exception as e:
            print(f"Error downloading {keyword} - image_{i}: {e}")

# Buat folder utama
if not os.path.exists('downloaded_images'):
    os.makedirs('downloaded_images')

# Loop melalui setiap kata kunci
for keyword in keywords:
    print(f"Scraping images for: {keyword}")
    download_images(keyword, max_images=10)

# Tutup browser
driver.quit()