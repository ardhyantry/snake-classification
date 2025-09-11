from icrawler.builtin import BingImageCrawler
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
    folder = os.path.join('datasetsBing', kw.split(" ")[0].lower())
    os.makedirs(folder, exist_ok=True)

    # Extract genus name and convert to lowercase
    genus = kw.split(" ")[0]
    prefix = genus.lower()

    # Initialize the crawler
    bing_crawler = BingImageCrawler(storage={'root_dir': folder})

    # # Attach prefix and counter to the crawler instance
    # bing_crawler.prefix = prefix
    # bing_crawler.counter = 0

    # # Define a custom get_filename method
    # def custom_get_filename(self, task):
    #     self.counter += 1
    #     return f"{self.prefix}_{self.counter}"

    # # Bind the custom method to the instance
    # bing_crawler.get_filename = custom_get_filename.__get__(bing_crawler)

    # Start crawling
    bing_crawler.crawl(
        keyword=kw,
        max_num=100000,
        file_idx_offset=0
    )