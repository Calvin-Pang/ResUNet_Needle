from dataset import NeedleDataset
import os
import argparse
from models import ResYNet
from torch.utils.data.dataloader import DataLoader
import torch
import os
import yaml
from tqdm import tqdm
from utils import *
torch.manual_seed(42)


def val(model, val_loader, config, exp_dir):
    model.eval()
    save_img_dir = os.path.join(exp_dir, 'test_imgs')
    if not os.path.exists(save_img_dir): os.mkdir(save_img_dir)
    with torch.no_grad():
        if config['model'] =='join':
            img_tem_id = 0
            for imgs, img_id in tqdm(val_loader, desc  ='Testing...', leave = False):
                img_tem_id += 1
                imgs = imgs.cuda()
                pred_mask, pred_class = model(imgs)
                binary_pred_mask = (pred_mask >= 0.5).float()
                transforms.ToPILImage()(binary_pred_mask[0]).save(os.path.join(save_img_dir, str(img_id[0]) + '.png'))

        if config['model'] =='seg':
            for imgs, img_id in tqdm(val_loader, desc  ='Testing...', leave = False):
                imgs = imgs.cuda()
                pred_mask = model(imgs)
                binary_pred_mask = (pred_mask >= 0.5).float()
                transforms.ToPILImage()(binary_pred_mask[0]).save(os.path.join(save_img_dir, str(img_id[0]) + '.png'))




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--exp_name')
    args = parser.parse_args()
    
    if not os.path.exists('./save'): os.mkdir('./save')
    exp_dir = os.path.join('./save', args.exp_name)
    if not os.path.exists(exp_dir): os.mkdir(exp_dir)


    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    print('Config loaded!')
    if config['model'] =='join': seg, cls = True, True
    elif config['model'] =='seg': seg, cls = True, False
    
    model = ResYNet(seg = seg, cls = cls, embed_dim = 64, in_channels = 1, num_classes = 1, res = config['res']).cuda()
    print('model:', config['model'])
    ckpt = torch.load(os.path.join(exp_dir, 'best_model.pt'))
    model.load_state_dict(ckpt)
    
    meta_file = config['meta_file']

    test_dataset = NeedleDataset(meta_file = meta_file,
                                img_dir = config['test_set']['img_dir'],
                                mask_dir = None,
                                mode = 'test',
                                augment = False)
    test_loader = DataLoader(dataset = test_dataset, batch_size = 1, shuffle = False)


    val(model, test_loader, config, exp_dir)