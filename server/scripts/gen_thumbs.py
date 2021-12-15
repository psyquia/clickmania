import cv2
import numpy as np
from tqdm import tqdm
import os

im_dir = '../static/artwork/face-terrors/' 
out_dir = f'{im_dir}/thumb/'

if not os.path.exists(im_dir):
    os.makedirs(im_dir)
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

for filename in tqdm(os.listdir(im_dir)):
  if not filename.endswith(".png"):
    continue

  # Read image
  image = cv2.imread(f'{im_dir}/{filename}')

  # resize
  result = cv2.resize(image, (64,64), interpolation=cv2.INTER_NEAREST)

  fname = f'{out_dir}/{filename}'

  # Save the image
  cv2.imwrite(fname, result)