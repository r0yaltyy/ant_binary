from glob import glob
import cv2
import numpy as np
import os

def preprocess(img, input_shape, letter_box=True):
    """img:         input image in numpy array
       input_shape: [height, width] of input image, this is the target shape for the model
       letter_box:  control whether to apply letterbox resizing """
    if letter_box:
        img_h, img_w, _ = img.shape                    #img is opened with opencv, in shape(h, w, c), this is the original image shape
        new_h, new_w = input_shape[0], input_shape[1]  # desired input shape for the model
        offset_h, offset_w = 0, 0                      # initialize the offset
        if (new_w / img_w) <= (new_h / img_h):         # if the resizing scale of width is lower than that of height
            new_h = int(img_h * new_w / img_w)         # get a new_h that is with the same resizing scale of width
            offset_h = (input_shape[0] - new_h) // 2   # update the offset_h 
        else:
            new_w = int(img_w * new_h / img_h)         # if the resizing scale of width is higher than that of height, update new_w
            offset_w = (input_shape[1] - new_w) // 2   # update the offset_w
        resized = cv2.resize(img, (new_w, new_h))      # get resized image using new_w and new_h
        img = np.full((input_shape[0], input_shape[1], 3), 127, dtype=np.uint8) # initialize a img with pixel value 127, gray color
        img[offset_h:(offset_h + new_h), offset_w:(offset_w + new_w), :] = resized 
    else:
        img = cv2.resize(img, (input_shape[1], input_shape[0]))

    return img

directory_in= ['/home/alex/ant_binary/train/aggressive','/home/alex/ant_binary/train/nentral','/home/alex/ant_binary/validation/aggressive','/home/alex/ant_binary/validation/nentral']
directory_out= ['/home/alex/ant_binary/train2/aggressive','/home/alex/ant_binary/train2/nentral','/home/alex/ant_binary/validation2/aggressive','/home/alex/ant_binary/validation2/nentral']
 
for current_dir in directory_out:
    for item in current_dir:
        if item.endswith(".jpg"):
            os.remove(os.path.join(current_dir, item))
img_num = 0
dir_num = 0
for current_dir in directory_in:
    for filename in os.listdir(current_dir):
        if filename.endswith(".jpg"):
            img = cv2.imread(current_dir + "/" + filename)
            new_img = preprocess(img, (50,50), True)
            cv2.imwrite(directory_out[dir_num] + "/" + "img" + str(img_num) + ".jpg",new_img)
            img_num = img_num + 1
            continue
        else:
            continue
    dir_num = dir_num + 1
