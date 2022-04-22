import numpy as np

import os
import os.path as osp
import json
from glob import glob
from tqdm import tqdm

from PIL import Image

from torch.utils.data import DataLoader, ConcatDataset, Dataset


########## 다운받은 Aihub data 디렉토리 경로 ##########
SRC_JSON = '/opt/ml/input/data/Aihub_korea/'

########## 새로 저장할 디렉토리 경로 ##########
DST_JSON = '/opt/ml/input/data/total/'

NUM_WORKERS = 32

IMAGE_EXTENSIONS = {'.gif', '.jpg', '.png','.JPG','.jpeg'}
LABEL_EXTENSIONS = {'.json'}

def maybe_mkdir(x):
    if not osp.exists(x):
        os.makedirs(x,exist_ok=True)

class AIhubDataset(Dataset):
    def __init__(self, image_dir, label_dir,copy_images_to=None):
        image_paths = {x for x in glob(osp.join(image_dir, '**'),recursive=True) if osp.splitext(x)[1] in
                       IMAGE_EXTENSIONS}
        label_paths = {x for x in glob(osp.join(label_dir, '**'),recursive=True) if osp.splitext(x)[1] in
                       LABEL_EXTENSIONS}
        assert len(image_paths) == len(label_paths)

        sample_ids, samples_info = list(), dict()
        for image_path in image_paths:
            sample_id = osp.splitext(osp.basename(image_path))[0]

            label_path = osp.join(label_dir,image_path.split('/')[7],sample_id) + '.json'
            words_info = self.parse_label_file(label_path)

            sample_ids.append(sample_id)
            samples_info[sample_id] = dict(image_path=image_path, label_path=label_path,
                                           words_info=words_info)
        
        self.sample_ids, self.samples_info = sample_ids, samples_info

        self.copy_images_to = copy_images_to
    

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sample_info = self.samples_info[self.sample_ids[idx]]

        image_fname = osp.basename(sample_info['image_path'])
        image = Image.open(sample_info['image_path'])
        img_w, img_h = image.size
        
        if self.copy_images_to:
            maybe_mkdir(self.copy_images_to)
            image.save(osp.join(self.copy_images_to, osp.basename(sample_info['image_path'])))

        sample_info_ufo = dict(img_h=img_h, img_w=img_w, words=sample_info['words_info'], tags=None,
                               license_tag=None)
        
        return image_fname, sample_info_ufo

    def parse_label_file(self,label_path):
        def rearrange_points(points):
            start_idx = np.argmin([np.linalg.norm(p, ord=1) for p in points])
            if start_idx != 0:
                points = np.roll(points, -start_idx, axis=0).tolist()
            return points

        with open(label_path) as f:
            image = json.load(f)
        
        words_info = dict()
        words = image['annotations']
        for i in range(len(words)):
            x = words[i]['bbox'][0]
            y = words[i]['bbox'][1]
            x2 = words[i]['bbox'][2]
            y2 = words[i]['bbox'][3]
            points = [[x,y],[x+x2,y],[x+x2,y+y2],[x,y+y2]]
            points = rearrange_points(points)

            transcription = words[i]['text']
            words_info[i] = dict(
                points=points, transcription=transcription, language=None,
                illegibility=False, orientation=None, word_tags=None
            ) 
        return words_info

def main():
    dst_image_dir = osp.join(DST_JSON, 'images')
    # dst_image_dir = None

    Aihub_train = AIhubDataset(osp.join(SRC_JSON,'images'),
                             osp.join(SRC_JSON, 'ufo','2.책표지'),
                             copy_images_to=dst_image_dir)

    anno = dict(images=dict())
    with tqdm(total=len(Aihub_train)) as pbar:
        for batch in DataLoader(Aihub_train, num_workers=NUM_WORKERS, collate_fn=lambda x: x):
            image_fname, sample_info = batch[0]
            anno['images'][image_fname] = sample_info
            pbar.update(1)
    
    ufo_dir = osp.join(DST_JSON, 'ufo')
    maybe_mkdir(ufo_dir)
    with open(osp.join(ufo_dir, 'train.json'), 'w') as f:
        json.dump(anno, f, indent=4)


if __name__ == '__main__':
    main()