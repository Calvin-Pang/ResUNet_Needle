from dataset import NeedleDataset
import os
import argparse
from models import ResYNet
from torch.utils.data.dataloader import DataLoader
from torch.optim import Adam
import torch.nn as nn
import torch
import os
import yaml
import copy
from torch.optim.lr_scheduler import MultiStepLR
from dice_loss import DiceLoss, Sensitivity, StraightnessLoss
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
from utils import *
torch.manual_seed(42)
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

def train(model, train_loader, optimizer, config):
    model.train()
    loss_record = []
    celoss_record = []
    diceloss_record = []
    clsloss_record = []
    straightloss_record = []
    num_batch = len(train_loader)
    celoss_fn, diceloss_fn, straightloss_fn = nn.BCELoss(), DiceLoss(), StraightnessLoss()
    if config['model'] =='join':
        for imgs, masks, labels, img_id in tqdm(train_loader, desc = 'Training...', leave = False):
            imgs, masks, labels = imgs.cuda(), masks.cuda(), labels.float().unsqueeze(-1).cuda()
            pred_mask, pred_class = model(imgs)

            celoss = celoss_fn(pred_mask, masks)
            diceloss = diceloss_fn(pred_mask, masks)
            straightloss = straightloss_fn(pred_mask)
            clsloss = celoss_fn(pred_class, labels)
            loss = config['seg_lambda'] * (celoss + config['dice_lambda'] * diceloss + config['straight_lambda'] * straightloss) + clsloss
                        
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_value_(model.parameters(), 0.1)
            optimizer.step()

            loss_record.append(loss.item())
            celoss_record.append(celoss.item())
            diceloss_record.append(diceloss.item())
            straightloss_record.append(straightloss.item())
            clsloss_record.append(clsloss.item())
        return sum(loss_record) / num_batch, sum(celoss_record) / num_batch, sum(diceloss_record) / num_batch, sum(straightloss_record) / num_batch, sum(clsloss_record) / num_batch

    if config['model'] =='seg':
        for imgs, masks, labels, img_id in tqdm(train_loader, desc = 'Training...', leave = False):
            imgs, masks = imgs.cuda(), masks.cuda()
            pred_mask = model(imgs)
            celoss = celoss_fn(pred_mask, masks)
            diceloss = diceloss_fn(pred_mask, masks)
            straightloss = straightloss_fn(pred_mask)
            loss = celoss + config['dice_lambda'] * diceloss + config['straight_lambda'] * straightloss
                    
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_value_(model.parameters(), 0.1)
            optimizer.step()

            loss_record.append(loss.item())
            celoss_record.append(celoss.item())
            diceloss_record.append(diceloss.item())
            straightloss_record.append(straightloss.item())
        return sum(loss_record) / num_batch, sum(celoss_record) / num_batch, sum(diceloss_record) / num_batch, sum(straightloss_record) / num_batch

def val(model, val_loader, config, exp_dir, epoch_id):
    model.eval()
    sen_fn = Sensitivity()
    loss_record = []
    dice_score_record = []
    sensitivity_record = []
    pred_label_record = []
    gt_label_record = []
    save_img_dir = os.path.join(exp_dir, 'val_imgs')
    if not os.path.exists(save_img_dir): os.mkdir(save_img_dir)
    save_img_epoch_dir = os.path.join(save_img_dir, 'epoch_' + str(epoch_id))
    if not os.path.exists(save_img_epoch_dir): os.mkdir(save_img_epoch_dir)
    celoss_fn, diceloss_fn, straightloss_fn = nn.BCELoss(), DiceLoss(), StraightnessLoss()
    with torch.no_grad():
        if config['model'] =='join':
            for imgs, masks, labels, img_id in tqdm(val_loader, desc  ='Testing...', leave = False):
                imgs, masks, labels = imgs.cuda(), masks.cuda(), labels.float().unsqueeze(-1).cuda()
                pred_mask, pred_class = model(imgs)
                celoss = celoss_fn(pred_mask, masks)
                diceloss = diceloss_fn(pred_mask, masks)
                clsloss = celoss_fn(pred_class, labels)
                straightloss = straightloss_fn(pred_mask)
                loss = config['seg_lambda'] * (celoss + config['dice_lambda'] * diceloss + config['straight_lambda'] * straightloss) + clsloss
                loss_record.append(loss.item())

                binary_pred_mask = (pred_mask >= 0.5).float()
                binary_pred_class = (pred_class >= 0.5).float()
                dice_score_record.append((1 - diceloss_fn(binary_pred_mask, masks)).item())
                sensitivity_record.append(sen_fn(binary_pred_mask, masks).item())
                pred_label_record.append(binary_pred_class.item())
                gt_label_record.append(labels.item())
                draw_mask(imgs[0], binary_pred_mask[0], os.path.join(save_img_epoch_dir, str(img_id[0]) + '.png'))

            loss_avg = sum(loss_record) / len(val_loader)
            dice_score_avg = sum(dice_score_record) / len(val_loader)
            sensitivity_score_avg = sum(sensitivity_record) / len(val_loader)
            acc = accuracy_score(pred_label_record, gt_label_record)
            f1 = f1_score(pred_label_record, gt_label_record)
            return loss_avg, dice_score_avg, sensitivity_score_avg, acc, f1

        if config['model'] =='seg':
            for imgs, masks, labels, img_id in tqdm(val_loader, desc  ='Testing...', leave = False):
                imgs, masks, labels = imgs.cuda(), masks.cuda(), labels.float().unsqueeze(-1).cuda()
                pred_mask = model(imgs)
                celoss = celoss_fn(pred_mask, masks)
                diceloss = diceloss_fn(pred_mask, masks)
                straightloss = straightloss_fn(pred_mask)
                loss = celoss + config['dice_lambda'] * diceloss + config['straight_lambda'] * straightloss
                loss_record.append(loss.item())
                
                binary_pred_mask = (pred_mask >= 0.5).float()
                dice_score_record.append((1 - diceloss_fn(binary_pred_mask, masks)).item())
                sensitivity_record.append(sen_fn(binary_pred_mask, masks).item())
                draw_mask(imgs[0], binary_pred_mask[0], os.path.join(save_img_epoch_dir, str(img_id[0]) + '.png'))

            loss_avg = sum(loss_record) / len(val_loader)
            dice_score_avg = sum(dice_score_record) / len(val_loader)
            sensitivity_score_avg = sum(sensitivity_record) / len(val_loader)
            return loss_avg, dice_score_avg, sensitivity_score_avg


def main(model, train_loader, val_loader, config, exp_dir):
    epoch, lr = config['epoch'], config['lr']
    txt_record = os.path.join(exp_dir, 'record.txt')
    best_model = copy.deepcopy(model)
    best_score = 0
    optimizer = Adam(model.parameters(), lr = lr) #, weight_decay = 2e-5)
    lr_scheduler = MultiStepLR(optimizer, milestones = config['multi_lr_milestones'], gamma = 0.5)
    best_dice_score, best_sensitivity, best_cls_acc, best_cls_f1 = 0, 0, 0, 0
    for i in range(epoch):
        epoch_id = i + 1
        lr = lr_scheduler.get_last_lr()[0]
        
        if config['model'] =='join':
            train_loss, celoss, diceloss, straightloss, clsloss = train(model, train_loader, optimizer, config)
            with open(txt_record, 'a') as f:
                print(f'epoch {epoch_id}/{epoch}: lr={lr:.3e}',
                      'train loss = {:.6f}'.format(train_loss), 
                      'CE loss = {:.6f}'.format(celoss), 
                      'Dice loss = {:.6f}'.format(diceloss), 
                      'Straight loss = {:.6}'.format(straightloss),
                      'Classification loss = {:.6f}'.format(clsloss), 
                      file = f)
                print(f'epoch {epoch_id}/{epoch}: lr={lr:.3e}',
                      'train loss = {:.6f}'.format(train_loss), 
                      'CE loss = {:.6f}'.format(celoss), 
                      'Dice loss = {:.6f}'.format(diceloss),
                      'Straight loss = {:.6}'.format(straightloss),
                      'Classification loss = {:.6f}'.format(clsloss))
            if epoch_id % config['val_every'] == 0:
                val_loss, dice_score, sensitivity, cls_acc, cls_f1 = val(model, val_loader, config, exp_dir, epoch_id)
                with open(txt_record, 'a') as f:
                    print(f'epoch {epoch_id}/{epoch}: lr={lr:.3e}',
                        'val loss = {:.6f}'.format(val_loss), 
                        'Dice score = {:.6f}'.format(dice_score), 
                        'Seg sensitivity = {:.6f}'.format(sensitivity), 
                        'Classification acc = {:.6f}'.format(cls_acc), 
                        'Classification F1 = {:.6f}'.format(cls_f1), 
                        file = f)
                    print(f'epoch {epoch_id}/{epoch}: lr={lr:.3e}',
                        'val loss = {:.6f}'.format(val_loss), 
                        'Dice score = {:.6f}'.format(dice_score), 
                        'Seg sensitivity = {:.6f}'.format(sensitivity), 
                        'Classification acc = {:.6f}'.format(cls_acc), 
                        'Classification F1 = {:.6f}'.format(cls_f1))   
                if dice_score + sensitivity > best_score: 
                    best_score = dice_score + sensitivity
                    best_dice_score = dice_score
                    best_sensitivity = sensitivity
                    best_model.load_state_dict(copy.deepcopy(model.state_dict()))
                    checkpoint_name = os.path.join(exp_dir, 'best_model.pt')
                    torch.save(best_model.state_dict(), checkpoint_name) 
                             
        elif config['model'] =='seg':
            train_loss, celoss, diceloss, straightloss = train(model, train_loader, optimizer, config)
            with open(txt_record, 'a') as f:
                print(f'epoch {epoch_id}/{epoch}: lr={lr:.3e}',
                      'train loss = {:.6f}'.format(train_loss), 
                      'CE loss = {:.6f}'.format(celoss), 
                      'Dice loss = {:.6f}'.format(diceloss), 
                      'Straight loss = {:.6}'.format(straightloss),
                      file = f)
                print(f'epoch {epoch_id}/{epoch}: lr={lr:.3e}',
                      'train loss = {:.6f}'.format(train_loss), 
                      'CE loss = {:.6f}'.format(celoss), 
                      'Dice loss = {:.6f}'.format(diceloss),
                      'Straight loss = {:.6}'.format(straightloss))
            if epoch_id % config['val_every'] == 0:
                val_loss, dice_score, sensitivity = val(model, val_loader, config, exp_dir, epoch_id)
                with open(txt_record, 'a') as f:
                    print(f'epoch {epoch_id}/{epoch}: lr={lr:.3e}',
                        'val loss = {:.6f}'.format(val_loss), 
                        'Dice score = {:.6f}'.format(dice_score), 
                        'Seg sensitivity = {:.6f}'.format(sensitivity), 
                        file = f)
                    print(f'epoch {epoch_id}/{epoch}: lr={lr:.3e}',
                        'val loss = {:.6f}'.format(val_loss), 
                        'Dice score = {:.6f}'.format(dice_score), 
                        'Seg sensitivity = {:.6f}'.format(sensitivity)) 
                if dice_score + sensitivity > best_score: 
                    best_score = dice_score + sensitivity
                    best_dice_score = dice_score
                    best_sensitivity = sensitivity
                    best_model.load_state_dict(copy.deepcopy(model.state_dict()))
                    checkpoint_name = os.path.join(exp_dir, 'best_model.pt')
                    torch.save(best_model.state_dict(), checkpoint_name) 
            
        lr_scheduler.step()  
    with open(txt_record, 'a') as f:
        print('Best dice score = {:.6f}'.format(best_dice_score), 
            'Best seg sensitivity = {:.6f}'.format(best_sensitivity), 
            file = f)
        print('Best dice score = {:.6f}'.format(best_dice_score), 
            'Best seg sensitivity = {:.6f}'.format(best_sensitivity))




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

    meta_file = config['meta_file']
    train_dataset = NeedleDataset(meta_file = meta_file,
                                img_dir = config['train_set']['img_dir'],
                                mask_dir = config['train_set']['mask_dir'],
                                mode = 'train',
                                augment = True)
    train_loader = DataLoader(dataset = train_dataset, batch_size = config['batch_size'], shuffle = True)

    val_dataset = NeedleDataset(meta_file = meta_file,
                                img_dir = config['val_set']['img_dir'],
                                mask_dir = config['val_set']['mask_dir'],
                                mode = 'train',
                                augment = False)
    val_loader = DataLoader(dataset = val_dataset, batch_size = 1, shuffle = False)


    main(model, train_loader, val_loader, config, exp_dir)