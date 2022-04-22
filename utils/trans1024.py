from PIL import Image
import os
from pathlib import Path
import copy
import json
from PIL import Image, ImageOps
def read_json(filename):
    with Path(filename).open(encoding='utf8') as handle:
        ann = json.load(handle)
    return ann
json_dir = '/opt/ml/input/data/ICDAR17_Korean/ufo/real_final.json'
img_dir = '/opt/ml/input/data/ICDAR17_Korean/images_origin'  # 원본 이미지 경로
img_dst_dir = '/opt/ml/input/data/ICDAR17_Korean/images'  # resize한 이미지 저장 경로

data = read_json(json_dir)
resize = 1024

new_json = {'images': {}}
for name in data['images'].keys():
    image_path = os.path.join(img_dir, name)
    img = Image.open(image_path)
    img = ImageOps.exif_transpose(img)
    h, w = img.height, img.width
    ratio = resize / max(h, w)
    if w > h:
        img = img.resize((resize, int(h * ratio)), Image.BILINEAR)
        new_w = resize
        new_h = int(h * ratio)
    else:
        img = img.resize((int(w * ratio), resize), Image.BILINEAR)
        new_w = int(w * ratio)
        new_h = resize
        
		##############
    new_name = name  # 새로운 json에 images key값으로 저장될 이름
    new_json['images'][new_name] = copy.deepcopy(data['images'][name])
    new_json['images'][new_name]['img_w'] = new_w
    new_json['images'][new_name]['img_h'] = new_h
    for k in new_json['images'][new_name]['words'].keys():
        # new_json['images'][new_name]['words'][k]['illegibility'] = False  # 안 하실 분은 주석 처리 해주세요
        for i in range(len(new_json['images'][new_name]['words'][k]['points'])):
            for j in range(2):
                new_json['images'][new_name]['words'][k]['points'][i][j] *= ratio
            
    if not os.path.exists(img_dst_dir):
        os.mkdir(img_dst_dir)
    dst_path = os.path.join(img_dst_dir, name)
    img.save(dst_path)

# 수정된 json 저장
with open('/opt/ml/input/data/ICDAR17_Korean/ufo/real_final1024.json', 'w') as outfile:
    json.dump(new_json, outfile)
