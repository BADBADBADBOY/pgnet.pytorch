import torch
from pgnet_model import PGnet
from pgnet_dataload import PGDataLoad,alignCollate
from pgnet_loss import PGLoss
import cv2
import numpy as np
import torch.nn.functional as F
from logger import Logger


def adjust_learning_rate(config, optimizer, epoch):
    if epoch in config['optimizer_decay']['schedule']:
        adjust_lr = optimizer.param_groups[0]['lr'] * config['optimizer_decay']['gama']
        for param_group in optimizer.param_groups:
            param_group['lr'] = adjust_lr 
            
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
        
model = PGnet().cuda()
train_dataset = PGDataLoad('/src/notebooks/imagenet/total_text/train','/src/notebooks/imagenet/total_text/train/train_new.txt')
train_data_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=10,
            shuffle=True,
            num_workers=12,
            collate_fn = alignCollate(),
            drop_last=True,
            pin_memory=True)

model_dict = torch.load('./pgnet_model.pth')
model.load_state_dict(model_dict)

criterion = PGLoss()
optimizer = torch.optim.Adam(lr=0.001,params=model.parameters())

config = {}
config['optimizer_decay'] ={}
config['optimizer_decay']['schedule'] = [250,500,550]
config['optimizer_decay']['gama'] = 0.1

logbin = Logger('./train_log.txt')
logbin.set_names(['total loss','score loss','border loss','direc loss','ctc loss'])

for i in range(600):
    loss_total = AverageMeter()
    loss_score = AverageMeter()
    loss_border = AverageMeter()
    loss_direction = AverageMeter()
    loss_ctc = AverageMeter()
    
    adjust_learning_rate(config, optimizer, i)
    
    for (train_iter,data) in enumerate(train_data_loader): 
        for key in data.keys():
            data[key] = data[key].cuda()
        imgs = data['images']
        out = model(imgs)
        
        ##### show img
        index = 0
        show_img = imgs[index].cpu().numpy().transpose((1, 2, 0))
        show_img[:, :, 2] *= (255.0 * 0.229)
        show_img[:, :, 1] *= (255.0 * 0.224)
        show_img[:, :, 0] *= (255.0 * 0.225)
        show_img[:, :, 2] += 0.485 * 255
        show_img[:, :, 1] += 0.456 * 255
        show_img[:, :, 0] += 0.406 * 255
        
        cv2.imwrite('img.jpg',show_img.astype(np.uint8))
        cv2.imwrite('f_score_512.jpg',F.interpolate(out['f_score'],(512,512))[index,0].detach().cpu().numpy()*255)
        cv2.imwrite('f_score.jpg',out['f_score'][index,0].detach().cpu().numpy()*255)
        ######
        
        loss = criterion(out,data)
        optimizer.zero_grad()
        loss['loss'].backward()
        optimizer.step()
        
        loss_total.update(loss['loss'].item())
        loss_score.update(loss['score_loss'].item())
        loss_border.update(loss['border_loss'].item())
        loss_direction.update(loss['direction_loss'].item())
        loss_ctc.update(loss['ctc_loss'].item())
        
        if train_iter%5==0:
            log = ''
            log+='{}/{}/{}/{}'.format(i,600,train_iter,len(train_data_loader))+' | '
            for key in loss.keys():
                log += '{}:{:.2f} | '.format(key,loss[key].item())
            log+= ' | Lr:{:.8f}'.format(optimizer.param_groups[0]['lr'])
            print(log)
            
    print('epoch_loss:{:.2f} | epoch_score_loss:{:.2f} | epoch_border_loss:{:.2f} |  \
          epoch_dire_loss:{:.2f} | epoch_ctc_loss:{:.2f}'.format(loss_total.avg,loss_score.avg,loss_border.avg,loss_direction.avg,loss_ctc.avg))
    logbin.append([loss_total.avg,loss_score.avg,loss_border.avg,loss_direction.avg,loss_ctc.avg])
    torch.save(model.state_dict(),'./pgnet_model_total_text.pth')
        

