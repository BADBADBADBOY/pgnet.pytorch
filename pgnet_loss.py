from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from pgnet_utils import pre_process
import itertools


class DiceLoss(nn.Module):
    def __init__(self,eps=1e-6):
        super(DiceLoss,self).__init__()
        self.eps = eps
    def forward(self,pre_score,gt_score,train_mask):
        pre_score = pre_score.contiguous().view(pre_score.size()[0], -1)
        gt_score = gt_score.contiguous().view(gt_score.size()[0], -1)
        train_mask = train_mask.contiguous().view(train_mask.size()[0], -1)

        pre_score = pre_score * train_mask
        gt_score = gt_score * train_mask

        a = torch.sum(pre_score * gt_score, 1)
        b = torch.sum(pre_score * pre_score, 1) + self.eps
        c = torch.sum(gt_score * gt_score, 1) + self.eps
        d = (2 * a) / (b + c)
        dice_loss = torch.mean(d)
        return 1 - dice_loss


def gather_nd(params,indices):
    """
    :param params: (Tensor) - the source Tensor
    :param indices: (LongTensor) - the indices of elements to gather
    :return output: (Tensor) – the destination tensor
    """
    assert indices.dtype == torch.int64, f"indices must be torch.LongTensor, got {indices.dtype}"
    assert indices.shape[-1] <= len(params.shape), f'The last dimension of indices can be at most the rank ' \
                                                   f'of params ({len(params.shape)})'

    # Define the output shape. According to the  documentation of tf.gather_nd, this is:
    # "indices.shape[:-1] + params.shape[indices.shape[-1]:]"
    output_shape = indices.shape[:-1] + params.shape[indices.shape[-1]:]

    # Initialize the output Tensor as an empty one.
    output = torch.zeros(size=output_shape, device=params.device, dtype=params.dtype)

    # indices_to_fill is a list of tuple containing the indices to fill in `output`
    indices_to_fill = list(itertools.product(*[range(x) for x in output_shape[:-1]]))

    # Loop over the indices_to_fill and fill the `output` Tensor
    for idx in indices_to_fill:
        index_value = indices[idx]

        if len(index_value.shape) == 0:
            index_value = torch.Tensor([0, index_value.item()])

        value = params[index_value.view(-1, 1).tolist()].view(-1)
        output[idx] = value
    return output

class PGLoss(nn.Module):
    def __init__(self,
                 tcl_bs=64,
                 max_text_length=50,
                 max_text_nums=30,
                 pad_num=36,
                 eps=1e-6,
                 **kwargs):
        super(PGLoss, self).__init__()
        self.tcl_bs = tcl_bs
        self.max_text_nums = max_text_nums
        self.max_text_length = max_text_length
        self.pad_num = pad_num
        self.dice_loss = DiceLoss(eps=eps)

    def border_loss(self, f_border, l_border, l_score, l_mask):
        l_border_split, l_border_norm = l_border[:,0:-1,:],l_border[:,-1:,:]
        f_border_split = f_border
        b, c, h, w = l_border_norm.shape
        l_border_norm_split = l_border_norm.expand([b, 4 * c, h, w])
        b, c, h, w = l_score.shape
        l_border_score = l_score.expand([b, 4 * c, h, w])
        b, c, h, w = l_mask.shape
        l_border_mask = l_mask.expand([b, 4 * c, h, w])
        border_diff = l_border_split - f_border_split
        abs_border_diff = torch.abs(border_diff)
        border_sign = abs_border_diff < 1.0
        border_sign = border_sign.float()
        border_sign.stop_gradient = True
        border_in_loss = 0.5 * abs_border_diff * abs_border_diff * border_sign + \
                         (abs_border_diff - 0.5) * (1.0 - border_sign)
        border_out_loss = l_border_norm_split * border_in_loss
        border_loss = torch.sum(border_out_loss * l_border_score * l_border_mask) / \
                      (torch.sum(l_border_score * l_border_mask) + 1e-5)
        return border_loss

    def direction_loss(self, f_direction, l_direction, l_score, l_mask):
        l_direction_split, l_direction_norm = l_direction[:,0:-1,:],l_direction[:,-1:,:]
        f_direction_split = f_direction
        b, c, h, w = l_direction_norm.shape
        l_direction_norm_split = l_direction_norm.expand([b, 2 * c, h, w])
        b, c, h, w = l_score.shape
        l_direction_score = l_score.expand([b, 2 * c, h, w])
        b, c, h, w = l_mask.shape
        l_direction_mask = l_mask.expand([b, 2 * c, h, w])
        direction_diff = l_direction_split - f_direction_split
        abs_direction_diff = torch.abs(direction_diff)
        direction_sign = abs_direction_diff < 1.0
        direction_sign = direction_sign.float()
        direction_sign.stop_gradient = True
        direction_in_loss = 0.5 * abs_direction_diff * abs_direction_diff * direction_sign + \
                            (abs_direction_diff - 0.5) * (1.0 - direction_sign)
        direction_out_loss = l_direction_norm_split * direction_in_loss
        direction_loss = torch.sum(direction_out_loss * l_direction_score * l_direction_mask) / \
                         (torch.sum(l_direction_score * l_direction_mask) + 1e-5)
        return direction_loss

    def ctcloss(self, f_char, tcl_pos, tcl_mask, tcl_label, label_t):
        f_char = f_char.permute([0, 2, 3, 1])
        tcl_pos = torch.reshape(tcl_pos, [-1, 3])
        tcl_pos = tcl_pos.long()
        f_tcl_char = gather_nd(f_char, tcl_pos)
        f_tcl_char = torch.reshape(f_tcl_char,[-1, 64, 37])  # len(Lexicon_Table)+1
        f_tcl_char_fg, f_tcl_char_bg = f_tcl_char[:,:,0:-1],f_tcl_char[:,:,-1:]
        f_tcl_char_bg = f_tcl_char_bg * tcl_mask + (1.0 - tcl_mask) * 20.0
        b, c, l = tcl_mask.shape
        tcl_mask_fg = tcl_mask.expand([b, c, 36 * l])
        tcl_mask_fg.stop_gradient = True
        f_tcl_char_fg = f_tcl_char_fg * tcl_mask_fg + (1.0 - tcl_mask_fg) * (-20.0)
        f_tcl_char_mask = torch.cat([f_tcl_char_fg, f_tcl_char_bg], 2)
        f_tcl_char_ld = f_tcl_char_mask.permute(1, 0, 2)
        N, B, _ = f_tcl_char_ld.shape
        input_lengths = torch.Tensor([N] * B).long()

        f_tcl_char_ld = f_tcl_char_ld.log_softmax(2).requires_grad_()
        cost = F.ctc_loss(log_probs=f_tcl_char_ld,
            targets=tcl_label,
            input_lengths=input_lengths,
            target_lengths=label_t,
            blank=self.pad_num,
            reduction='none')
        cost = cost.mean()
        return cost

    def forward(self, predicts, labels):
        images, tcl_maps, tcl_label_maps, border_maps \
        , direction_maps, training_masks, label_list, pos_list, pos_mask = labels['images'],labels['tcl_maps'],labels['tcl_label_maps'],labels['border_maps'] \
        ,labels['direction_maps'],labels['training_masks'],labels['label_list'],labels['pos_list'],labels['pos_mask']
        # for all the batch_size
        pos_list, pos_mask, label_list, label_t = pre_process(
            label_list.cpu(), pos_list.cpu(), pos_mask.cpu(), self.max_text_length,
            self.max_text_nums, self.pad_num, self.tcl_bs)
        pos_list, pos_mask, label_list, label_t = pos_list.cuda(), pos_mask.cuda(), label_list.cuda(), label_t.cuda()
        f_score, f_border, f_direction, f_char = predicts['f_score'], predicts['f_border'], predicts['f_direction'], \
                                                 predicts['f_char']
        score_loss = self.dice_loss(f_score, tcl_maps, training_masks)
        border_loss = self.border_loss(f_border, border_maps, tcl_maps,
                                       training_masks)
        direction_loss = self.direction_loss(f_direction, direction_maps,
                                             tcl_maps, training_masks)
        ctc_loss = self.ctcloss(f_char, pos_list, pos_mask, label_list, label_t)
        loss_all = score_loss + border_loss + direction_loss + 5 * ctc_loss

        losses = {
            'loss': loss_all,
            "score_loss": score_loss,
            "border_loss": border_loss,
            "direction_loss": direction_loss,
            "ctc_loss": ctc_loss
        }
        return losses
