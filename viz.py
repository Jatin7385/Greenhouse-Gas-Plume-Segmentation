from PIL import Image
import os
import matplotlib.pyplot as plt

path = 'C:\\Users\\Jatin\\Desktop\\Engineering\\Space General Advisory Council\\Small Satellites Project Group\\Work\\U-Net_Segmentation_Pet_Dataset\\STARCOP_train_easy\\STARCOP_train_easy\\ang20190922t192642_r4578_c217_w151_h151\\'
files = os.listdir(path)
for file in files:
    try:
        print("Reading file : " , file)
        im = Image.open(path + file)
        plt.imshow(im)
        plt.show()
    except Exception as e:
        print("Exception in file {file} : " , e)