import os
from PIL import Image
import imagehash

# Direktori gambar
folder_path = 'C:\\Users\\Ardhyan\\Documents\\Code TA\\DATASET\\VENOMOUS'  # ganti dengan path datasetmu
hashes = {}
duplicates = []

for filename in os.listdir(folder_path):
    if filename.endswith(('.png', '.jpg', '.jpeg')):
        file_path = os.path.join(folder_path, filename)
        try:
            img = Image.open(file_path)
            img_hash = imagehash.phash(img)  # perceptual hash

            if img_hash in hashes:
                print(f"Duplikat ditemukan: {filename} == {hashes[img_hash]}")
                duplicates.append(file_path)
            else:
                hashes[img_hash] = filename
        except Exception as e:
            print(f"Error saat membuka {filename}: {e}")

# Hapus file duplikat
for dup in duplicates:
    os.remove(dup)
    print(f"Hapus: {dup}")
