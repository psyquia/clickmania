import cv2
import numpy as np
from cv2 import dnn_superres
from tqdm import tqdm
import names
import os

# im_dir = '../output/128-r/' 
im_dir = '../output/almost-human' 
out_dir = '../output/almost-human/sharp'

if not os.path.exists(im_dir):
    os.makedirs(im_dir)
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# Create an SR object
sr = dnn_superres.DnnSuperResImpl_create()

# Read the desired model
path = "EDSR_x3.pb"
sr.readModel(path)

# Set the desired model and scale to get correct pre- and post-processing
sr.setModel("edsr", 3)

for filename in tqdm(os.listdir(im_dir)):
  if not filename.endswith(".png"):
    continue

  # Read image
  image = cv2.imread(f'{im_dir}/{filename}')

  # Upscale the image
  result = sr.upsample(image)

  filter = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
  # Applying cv2.filter2D function on our Logo image
  result=cv2.filter2D(result,-1,filter)

  # generate random name
  fname = f'{out_dir}/{names.get_first_name()}.png'
  while os.path.isfile(fname):
    fname = f'{out_dir}/{names.get_first_name()}.png'

  # Save the image
  cv2.imwrite(fname, result)