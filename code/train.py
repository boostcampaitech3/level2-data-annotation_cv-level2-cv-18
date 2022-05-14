import os
import os.path as osp
import time
import math
from datetime import timedelta
from argparse import ArgumentParser

import torch
from torch import cuda
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from tqdm import tqdm

from east_dataset import EASTDataset
from dataset import SceneTextDataset
from model import EAST
from PIL import ImageFile

from detect import get_bboxes
from deteval import calc_deteval_metrics
import numpy as np

ImageFile.LOAD_TRUNCATED_IMAGES = True

def parse_args():
    parser = ArgumentParser()

    # Conventional args
    parser.add_argument('--train_data_dir', type=str,
                        default=os.environ.get('SM_CHANNEL_TRAIN', '../input/data/split/train'))
    parser.add_argument('--valid_data_dir', type=str,
                        default=os.environ.get('SM_CHANNEL_TRAIN', '../input/data/split/val'))

    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR',
                                                                        'trained_models'))

    parser.add_argument('--device', default='cuda' if cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=4)

    parser.add_argument('--image_size', type=int, default=1024)
    parser.add_argument('--input_size', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--save_interval', type=int, default=5)

    args = parser.parse_args()

    if args.input_size % 32 != 0:
        raise ValueError('`input_size` must be a multiple of 32')

    return args

def make_bboxes(box_dict):
    bboxes = dict()
    bbox_list = []
    for image in box_dict['images']:
        for word in box_dict['images'][image]['words']:  
            bbox_list.append(box_dict['images'][image]['words'][word]['points'])
        bboxes[image] = bbox_list
    
    return bboxes

def do_training(train_data_dir, valid_data_dir, model_dir, device, image_size, input_size, num_workers, batch_size,
                learning_rate, max_epoch, save_interval):

    train_dataset = SceneTextDataset(train_data_dir, split='train', image_size=image_size, crop_size=input_size)
    train_dataset = EASTDataset(train_dataset)
    valid_dataset = SceneTextDataset(valid_data_dir, split='val', image_size=image_size, crop_size=input_size)
    valid_dataset = EASTDataset(valid_dataset)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EAST()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[max_epoch // 2], gamma=0.1)
    dataloaders = {
        "train" : train_loader,
        "valid" : valid_loader}

    max_hmean = 0
    for epoch in range(max_epoch):
        gt_bboxes, pred_bboxes, trans = [], [], []
        
        for phase in ['train','valid']:
            epoch_loss, epoch_start = 0, time.time()
            if phase == 'train':
                model.train()
                num_batches = math.ceil(len(train_dataset) / batch_size)
            else:
                model.eval()
                num_batches = math.ceil(len(valid_dataset) / batch_size)

            with tqdm(total=num_batches) as pbar:
                pbar.set_description('[Epoch {}]'.format(epoch + 1))
                for img, gt_score_map, gt_geo_map, roi_mask in dataloaders[phase]:
                    with torch.set_grad_enabled(phase=='train'):
                        loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)
                        optimizer.zero_grad()

                        if phase == "valid":
                            orig_sizes = []
                            for image in img:
                                orig_sizes.append(image.shape[1:3])
                            gt_bbox = []
                            pred_bbox = []
                            tran = []
                            with torch.no_grad():
                                pred_score_map, pred_geo_map = model.forward(img.to(device))
                            
                            for gt_score, gt_geo, pred_score, pred_geo, orig_size in zip(gt_score_map.cpu().numpy(), gt_geo_map.cpu().numpy(), pred_score_map.cpu().numpy(), pred_geo_map.cpu().numpy(), orig_sizes):
                                gt_bbox_angle = get_bboxes(gt_score, gt_geo)
                                pred_bbox_angle = get_bboxes(pred_score, pred_geo)
                                if gt_bbox_angle is None:
                                    gt_bbox_angle = np.zeros((0, 4, 2), dtype=np.float32)
                                    tran_angle = []
                                else:
                                    gt_bbox_angle = gt_bbox_angle[:, :8].reshape(-1, 4, 2)
                                    gt_bbox_angle *= max(orig_size) / input_size
                                    tran_angle = ['null' for _ in range(gt_bbox_angle.shape[0])]
                                if pred_bbox_angle is None:
                                    pred_bbox_angle = np.zeros((0, 4, 2), dtype=np.float32)
                                else:
                                    pred_bbox_angle = pred_bbox_angle[:, :8].reshape(-1, 4, 2)
                                    pred_bbox_angle *= max(orig_size) / input_size
                                    
                                tran.append(tran_angle)
                                gt_bbox.append(gt_bbox_angle)
                                pred_bbox.append(pred_bbox_angle)

                            gt_bboxes.extend(gt_bbox)
                            pred_bboxes.extend(pred_bbox)
                            trans.extend(tran)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                        loss_val = loss.item()
                        epoch_loss += loss_val

                    pbar.update(1)
                    val_dict = {
                        'Cls loss': extra_info['cls_loss'], 'Angle loss': extra_info['angle_loss'],
                        'IoU loss': extra_info['iou_loss']
                    }
                    pbar.set_postfix(val_dict)
                    
            if phase == 'train':
                scheduler.step()

            print('[Epoch : {}] [Phase : {}] Mean loss: {:.4f} | Elapsed time: {}'.format(
                epoch+1, phase, epoch_loss / num_batches, timedelta(seconds=time.time() - epoch_start)))

            if phase == 'valid':
                img_len = len(valid_dataset)
                pred_bboxes_dict, gt_bboxes_dict, trans_dict = dict(), dict(), dict()
                for img_num in range(img_len):
                    pred_bboxes_dict[f'img_{img_num}'] = pred_bboxes[img_num]
                    gt_bboxes_dict[f'img_{img_num}'] = gt_bboxes[img_num]
                    trans_dict[f'img_{img_num}'] = trans[img_num]
                
                deteval_dict = calc_deteval_metrics(pred_bboxes_dict, gt_bboxes_dict, trans_dict)
                metric_dict = deteval_dict['total']
                precision = metric_dict['precision']
                recall = metric_dict['recall']
                hmean = metric_dict['hmean']
                print('[Epoch : {}] [Phase : {}] precision : {:.5f} recall : {:.5f} hmean : {:.5f}'.format(
                    epoch+1,phase,precision,recall,hmean))
                

                if hmean > max_hmean:
                    max_hmean = hmean
                    if not osp.exists(model_dir):
                        os.makedirs(model_dir)

                    ckpt_fpath = osp.join(model_dir, 'best.pth')
                    torch.save(model.state_dict(), ckpt_fpath)
            
            if (epoch + 1) % save_interval == 0:
                if not osp.exists(model_dir):
                    os.makedirs(model_dir)

                ckpt_fpath = osp.join(model_dir, 'latest.pth')
                torch.save(model.state_dict(), ckpt_fpath)

def main(args):
    do_training(**args.__dict__)


if __name__ == '__main__':
    args = parse_args()
    main(args)
