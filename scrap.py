from icrawler.builtin import GoogleImageCrawler, BingImageCrawler ,
import os

keywords = [
    "Naja sputatrix (Ular Kobra Jawa)",
    "Ophiophagus hannah (King Cobra)",
    "Bungarus candidus (Ular Weling)",
    "Python reticulatus (Sanca Kembang)",
    "Coelognathus radiatus (Ular Tikus)",
    "Dendrelaphis pictus (Ular Pucuk)"
]

for kw in keywords:
    folder = f'images/{kw}'
    os.makedirs(folder, exist_ok=True)

    # Download from Google (waits until finished)
    google_crawler = GoogleImageCrawler(storage={'root_dir': folder})
    google_crawler.crawl(keyword=kw, max_num=500)

    # After Google is done, get already downloaded file names
    existing_files = set(os.listdir(folder))

    # Download from Bing, skipping duplicates
    class NoDuplicateBingCrawler(BingImageCrawler):
        def download(self, task, default_ext, timeout=5, max_retry=3, overwrite=False):
            filename = task['file_url'].split('/')[-1].split('?')[0]
            if filename in existing_files:
                return None
            return super().download(task, default_ext, timeout, max_retry, overwrite)

    bing_crawler = NoDuplicateBingCrawler(storage={'root_dir': folder})
    bing_crawler.crawl(keyword=kw, max_num=500)
