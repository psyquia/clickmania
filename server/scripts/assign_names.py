import names
import os
from tqdm import tqdm

to_change_path = '../static/artwork/face-terrors' 
other_path = '../static/artwork/almost-human' 

for filename in tqdm(os.listdir(to_change_path)):
  if not filename.endswith(".png"):
    continue

  fn = names.get_first_name()
  fname = f'{to_change_path}/{fn}.png'
  fname_other_path = f'{other_path}/{fn}.png'

  while os.path.isfile(fname) or os.path.isfile(fname_other_path):
    fn = names.get_first_name()
    fname = f'{to_change_path}/{fn}.png'
    fname_other_path = f'{other_path}/{fn}.png'

  os.rename(f'{to_change_path}/{filename}', fname)