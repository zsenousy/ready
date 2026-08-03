import pathlib
import numpy as np
from PIL import Image
import tifffile

tif_masks = pathlib.Path("C:/Users/Kofi Quarshie/downloads/ready/datasets/s-natural/s-natural/mask-withoutskin-noglasses1")
masks_out = pathlib.Path("C:/Users/Kofi Quarshie/downloads/ready/datasets/s-natural/s-natural/mask-withoutskin-noglasses1_png")
masks_out.mkdir(parents=True, exist_ok=True)
tif_images = pathlib.Path("C:/Users/Kofi Quarshie/downloads/ready/datasets/s-natural/s-natural/synthetic1")
images_out = pathlib.Path("C:/Users/Kofi Quarshie/downloads/ready/datasets/s-natural/s-natural/synthetic1_png")
images_out.mkdir(parents=True, exist_ok=True)

count = 0
for tif_file in sorted(tif_masks.iterdir()):
    if tif_file.suffix == ".tif":
        mask_array = tifffile.imread(tif_file)
        mask_array = np.squeeze(mask_array)
        mask_array = mask_array.astype(np.uint8)
        mask = Image.fromarray(mask_array)
        mask.save(masks_out / tif_file.with_suffix(".png").name)
        count += 1

for tif_img in sorted(tif_images.iterdir()):
    if tif_img.suffix == ".tif":
        img_array = tifffile.imread(tif_img)
        img_array = np.squeeze(img_array)
        img_array = img_array.astype(np.uint8)
        image = Image.fromarray(img_array)
        image.save(images_out / tif_img.with_suffix(".png").name)
        count += 1


print("Converted:", count, "mask and images files to .PNG format")
