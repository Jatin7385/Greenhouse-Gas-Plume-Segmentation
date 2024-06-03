import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

'''
File contains the helper functions
'''

def Load():
    path = "./STARCOP_train_easy/STARCOP_train_easy/"
    files = os.listdir(path)

    image_dataset = []
    mask_dataset = []

    unsuccesful = 0

    for file in files:
        try:
            file1 = "/TOA_AVIRIS_640nm.tif"
            file2 = "/labelbinary.tif"
            image_dataset.append(np.array(Image.open(path + file + file1)))
            mask_dataset.append(np.array(Image.open(path + file + file2)))
        except Exception as e:
            unsuccesful+=1
            print("EXCEPTION in  {file} : " , e)
    
    print("Image and Mask shape : " , image_dataset[0].shape)
    print(f"Image Dataset size : {len(image_dataset)}")
    print(f"Mask dataset size : {len(mask_dataset)}")
    print(f"No. of images not loaded  : {unsuccesful}")
    

    return image_dataset, mask_dataset

    # print(files)

    # image_dataset = os.listdir(path1)
    # mask_dataset = os.listdir(path2)
    
    # print(image_dataset, mask_dataset)


    # return image_dataset, mask_dataset


# def Preprocess_AVIRIS(img, mask):
#     '''
#     Makes patches of target shape, to increase dataset size
#     '''
    

    
#     # Define the desired shape
#     target_shape_img = [128, 128]
#     target_shape_mask = [128, 128]




def load_and_process(display, n):
    img, mask = Load()

    print("Data Loaded....")


    if(display):
        for i in range(n):
            fig, arr = plt.subplots(1, 2, figsize=(15, 15), sharex=True, sharey=True)
            arr[0].imshow(img[i])
            arr[0].set_title('Image '+ str(i))
            arr[1].imshow(mask[i])
            arr[1].set_title('Masked Image '+ str(i))
            plt.show()


    # If processing required uncomment next line
    
    # img, mask = Preprocess_AVIRIS(img, mask)
    
    # print("Data Processed....")

    # if(display):
    #     for i in range(n):
    #         fig, arr = plt.subplots(1, 2, figsize=(15, 15), sharex=True, sharey=True)
    #         arr[0].imshow(img[i])
    #         arr[0].set_title('Processed Image '+ str(i))
    #         arr[1].imshow(mask[i])
    #         arr[1].set_title('Processed Masked Image '+ str(i))
    #         plt.show()


    return img, mask

    


# Load()
# Load("./images","./images")