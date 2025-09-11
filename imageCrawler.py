from icrawler.builtin import GoogleImageCrawler
from types import MethodType
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
    # Create folder for each keyword
    folder = os.path.join('datasets', kw.split(" ")[0].lower())
    os.makedirs(folder, exist_ok=True)

    # Extract genus name and convert to lowercase
    genus = kw.split(" ")[0]
    prefix = genus.lower()

    # Initialize the crawler
    google_crawler = GoogleImageCrawler(storage={'root_dir': folder})

    # Attach prefix and counter to the crawler instance
    google_crawler.prefix = prefix
    google_crawler.counter = 0

    # Define a custom get_filename method
    def custom_get_filename(self, task):
        self.counter += 1
        return f"{self.prefix}_{self.counter}"

    # Bind the custom method to the instance
    google_crawler.get_filename = MethodType(custom_get_filename, google_crawler)

    # Start crawling
    google_crawler.crawl(
        keyword=kw,
        max_num=100000,
        file_idx_offset=0
    )