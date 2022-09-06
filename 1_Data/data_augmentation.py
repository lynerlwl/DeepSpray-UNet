import os
from PIL import Image

target = "f_01213-edit"

img_ori = Image.open(f"image/{target}.png").convert('RGB')
msk_ori = Image.open(f"mask/{target}.png")

degree_list = [d * 30 for d in range(0, 12)]

new_folder = ["images", "masks"]
for i in new_folder:
    path = i
    CHECK_FOLDER = os.path.isdir(path)
    if not CHECK_FOLDER:
        os.makedirs(path)

for degree in degree_list:
    img_ori_rotate = img_ori.rotate(degree)
    file_name = f"images/{target}_{degree}.png"
    img_ori_rotate.save(file_name)
    
    mask_ori_rotate = msk_ori.rotate(degree)
    file_name = f"masks/{target}_{degree}.png"  
    mask_ori_rotate.save(file_name)