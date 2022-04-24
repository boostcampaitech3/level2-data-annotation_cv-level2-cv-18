import splitfolders
import os
import json
from glob import glob

seed = 1234

###### 분할할 디렉토리 ######
split_dir = '/opt/ml/input/data/ICDAR19'

###### 저장할 디렉토리 ######
save_dir = '/opt/ml/input/data/split'

if not os.path.exists(save_dir):
        os.makedirs(save_dir)

###### ratio를 원하시는대로 변경해주시면 됩니다. ######
############# ratio = (train,valid,test) ############
splitfolders.ratio(split_dir, output=save_dir, seed=seed, ratio=(0.9, 0.1))

# json split
def split_json(data):
    split_anno_list = []
    for anno in {os.path.basename(x) for x in glob(os.path.join(save_dir,str(data)+'/images/**'))}:
        split_anno_list.append(anno)
    
    with open(os.path.join(split_dir,'ufo/train.json')) as f:
        anno = json.load(f)
    
    anno_dict = dict(images=dict())
    for image in anno['images']:
            if image in split_anno_list:
                anno_dict['images'][image] = anno['images'][image]

    with open(os.path.join(save_dir,str(data),'ufo',str(data)+'.json'),'w') as f:
        json.dump(anno_dict,f,indent=4)
    
    return anno_dict

def main():
    train_json = split_json('train')
    val_json = split_json('val')

    print('train_images :', len(glob(os.path.join(save_dir,'train/images/**'))))
    print('valid_images :', len(glob(os.path.join(save_dir,'val/images/**'))))

    print('train_json :', len(train_json['images']))
    print('valid_json :', len(val_json['images']))

if __name__ == '__main__':
    main()