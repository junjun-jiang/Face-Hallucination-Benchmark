import os
from PIL import Image
image_GT = "/data/CelebA/val/HR"
image_LR = "/CelebA/val/HR_x16"


files = sorted(os.listdir(image_GT))
os.makedirs(image_LR, exist_ok=True)
print(len(files))
for i in range(len(files)):
    print(files[i])
    if files[i][-4:] != ".png":
        continue
    name = os.path.splitext(files[i])[0]
    newname = name + ".png"
    pic = Image.open(os.path.join(image_GT, files[i]))
    hr_shape = (128, 128)
    lr_shape = (8, 8)
    lr_img = pic.resize(lr_shape, resample=Image.BICUBIC) 
#   lr_img= lr_img.resize(hr_shape, resample=Image.BICUBIC)  
    lr_img.save(os.path.join(image_LR, newname))

