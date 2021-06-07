#script to convert VOTT CSV export to YOLO Darknet format
#Place script in the same folder as vott-csv-export and run

import numpy as np
import pandas as pd
import glob
import os

csv_vott_export= glob.glob('./*.csv')
assert len(csv_vott_export) == 1, 'Folder should cointain only 1 csv file'

thisdir=os.getcwd()

annot_df= pd.read_csv(os.path.join(thisdir, csv_vott_export[0]))

# https://github.com/AlexeyAB/darknet/blob/master/scripts/voc_label.py 

def convert(size, box):
    dw = 1./(size[1]) #cv2 img.shape returns h,w,channels 
    dh = 1./(size[0]) #size = cv2.imread img.shape
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

# Define your labels
label_dict= {'weed':0}

import cv2

def get_img_size(img_path): 
  img= cv2.imread(img_path)
  height, width, _ = img.shape 
  # return width, height #X,Y
  return img.shape

total_imgs= annot_df['image'].unique()

for img_name in total_imgs:
  temp_img_df= annot_df[img_name == annot_df['image']].copy()
  img_shape= get_img_size(img_name)
  # print(temp_img_df)
  with open(f"{temp_img_df['image'][temp_img_df.first_valid_index()][:-4]}.txt", "w") as f:
    for i,row in temp_img_df.iterrows():   
      x,y,w,h = convert(img_shape, [row['xmin'], row['xmax'], row['ymin'], row['ymax']] )   
      f.write(f"{label_dict[row['label']]} {x} {y} {w} {h}\n")

with open(f"_darknet.labels", "w") as f:
  for classes in label_dict:
    f.write(f'{classes}\n')