import numpy as np
import os
import shutil
import math

root = 'D:/projects/brain_tumour_detection/data/Testing'
numberOfImages = {}

# Count images in each class folder
for dir in os.listdir(root):
    numberOfImages[dir] = len(os.listdir(os.path.join(root, dir)))

print(numberOfImages)

def dataSplit(new_folder, split):
    if not os.path.exists(new_folder):
        os.mkdir(new_folder)

        for dir in os.listdir(root):

            dest_dir = os.path.join(new_folder, dir)
            os.makedirs(dest_dir, exist_ok=True)

            count = math.floor(split * numberOfImages[dir]) - 5
            selected_imgs = np.random.choice(
                a=os.listdir(os.path.join(root, dir)),
                size=count,
                replace=False
            )

            for img in selected_imgs:
                src = os.path.join(root, dir, img)
                dst = os.path.join(dest_dir, img)
                shutil.copy(src, dst)
                os.remove(src)

    else:
        print("Folder already exists:", new_folder)


dataSplit("Validation", 0.5)
