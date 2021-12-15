import os
import json
import numpy as np

artwork_path = '../static/artwork'

col_pk = 1
art_pk = 1

collection_map = {}

fixture = [] # output json fixture

for col in os.listdir(artwork_path):
  desc = ''
  with open(f'{artwork_path}/{col}/desc.txt', 'r') as file:
    desc = file.read().rstrip()
  # add collection to fixture
  fixture.append({
    'model':'art.collection',
    'pk': col_pk,
    'fields': {
      'name': col,
      'display_name': ' '.join([word.capitalize() for word in col.split('-')]),
      'description': desc
    }
  })
  collection_map[col] = col_pk
  col_pk+=1

for col,col_pk in collection_map.items():
  col_path = f'{artwork_path}/{col}'
  for im in os.listdir(col_path):
    if not im.endswith(".png"):
      continue
    # add art to fixture
    fixture.append({
      'model':'art.art',
      'pk': art_pk,
      'fields': {
        'title': im.split('.')[0],
        'filename': im,
        'rarity': int(np.random.choice(np.arange(1, 6), p=[0.3, 0.3, 0.25, 0.1, 0.05])),
        'collection': col_pk
      }
    })
    art_pk+=1

out = open("../fixture.json", "w")
out.write(json.dumps(fixture, indent=4, sort_keys=False))
out.close()