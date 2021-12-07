from pgnet_utils import DecodeImage,E2ELabelEncodeTest,E2EResizeForTest,NormalizeImage,ToCHWImage,KeepKeys
import os
from pgnet_model import PGnet
import numpy as np
import torch
from pg_postprocess import PGPostProcess
import cv2
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
def draw_e2e_res(dt_boxes, strs, img, img_name):
    if len(dt_boxes) > 0:
        src_im = img
        for box, str in zip(dt_boxes, strs):
            box = box.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(src_im, [box], True, color=(255, 255, 0), thickness=2)
            cv2.putText(
                src_im,
                str,
                org=(int(box[0, 0, 0]), int(box[0, 0, 1])),
                fontFace=cv2.FONT_HERSHEY_COMPLEX,
                fontScale=0.7,
                color=(0, 255, 0),
                thickness=1)
        save_det_path = './e2e_results/'
        if not os.path.exists(save_det_path):
            os.makedirs(save_det_path)
        save_path = os.path.join(save_det_path, os.path.basename(img_name))
        cv2.imwrite(save_path, src_im)
        print("The e2e Image saved in {}".format(save_path))
        
        
img_path = '/src/notebooks/imagenet/total_text/test/rgb'
args = {'max_side_len':736,'valid_set':'totaltext'}
files = os.listdir(img_path)
DecodeImage_bin = DecodeImage()
E2EResizeForTest_bin = E2EResizeForTest(args)
NormalizeImage_bin = NormalizeImage(scale= 1./255.,mean =[ 0.485, 0.456, 0.406 ],std=[ 0.229, 0.224, 0.225 ],order='hwc')
ToCHWImage_bin = ToCHWImage()
KeepKeys_bin = KeepKeys([ 'image', 'shape'])
post_process_class = PGPostProcess('./ic15_dict.txt','totaltext',0.5,'fast')

model = PGnet().cuda()
model_dict = torch.load('./pgnet_model_total_text.pth')
# import pdb
# pdb.set_trace()
model.load_state_dict(model_dict)
model.eval()
for file in files:
    with open(os.path.join(img_path,file), 'rb') as f:
        img = f.read()
        data = {'image': img}
    batch = DecodeImage_bin(data)
    batch = E2EResizeForTest_bin(batch)
    batch = NormalizeImage_bin(batch)
    batch = ToCHWImage_bin(batch)
    batch = KeepKeys_bin(batch)
    images = np.expand_dims(batch[0], axis=0)
    shape_list = np.expand_dims(batch[1], axis=0)
    with torch.no_grad():
        images = torch.Tensor(images).cuda()
        preds = model(images)
        post_result = post_process_class(preds, shape_list)
        points, strs = post_result['points'], post_result['texts']
    src_img = cv2.imread(os.path.join(img_path,file))
    draw_e2e_res(points, strs, src_img, os.path.join(img_path,file))