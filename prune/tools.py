import torch.nn as nn
import torch
from copy import deepcopy
import numpy as np
from utils.general import (LOGGER, NCOLS,  coco80_to_coco91_class, non_max_suppression, scale_coords, xywh2xyxy)
from tqdm import tqdm
from utils.torch_utils import time_sync
from pathlib import Path
from val_prune import process_batch
from utils.metrics import ap_per_class
from models.common import C3, SPPF, Bottleneck
import os
from utils.general import init_seeds
from torch.cuda import amp
import val
from models.common import *
from models.yolo import *


RANK = int(os.getenv('RANK', -1))
init_seeds(1 + RANK)


# 获得除exclude之外的BN层的权重和偏置
def gather_bn_weights(model, exclude=[] ,mode = 1):
    weights = []
    if mode == 1:
        for name, m in model.named_modules():
            if isinstance(m, nn.BatchNorm2d) and name not in exclude:
                weights.extend(list(m.weight.data.abs().clone()))
        weights = torch.tensor(weights)
        return weights
    elif mode == 2:
        weights = []
        bias = []
        for name, m in model.named_modules():
            if isinstance(m, nn.BatchNorm2d) and name not in exclude:
                weights.append((name, m.weight.data))
                bias.append((name, m.bias.data))
        return weights, bias

def update_BN_grad(model, s):
    exclude = get_exclude(model)
    for name, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d) and name not in exclude:
            m.weight.grad.data.add_(s * torch.sign(m.weight.data))

def get_thresh(model, global_percent = 0.0):
    exclude = get_exclude(model)
    bn_weights = gather_bn_weights(model, exclude)
    sorted_bn = torch.sort(bn_weights)[0]
    thre_index = int(len(sorted_bn) * global_percent)
    thresh = sorted_bn[thre_index]
    return thresh

def get_mask(module_list, thresh = 0.0):
    for module in module_list:
        w_copy = module.bn.weight.data.abs().clone()
        mask = w_copy >= thresh
        num_retain = torch.sum(mask)
        if torch.sum(mask) == 0:
            mask[0] = True
        module.out_mask = mask
        module.remain_ratio = num_retain / mask.shape[0]

def get_exclude(model):
    exclude = []
    for name ,m in model.named_modules():
        if isinstance(m, SPPF):
            exclude.append(name + '.cv1.bn')
        if isinstance(m,nn.Upsample):
            split = name.split('.')
            conv_before_upsample = split[0] + '.' + str(eval(split[1]) - 1) + '.bn'
            exclude.append(conv_before_upsample)
    return exclude


def pruning(model, Copy = True, global_percent = 0.0):
    thresh = get_thresh(model, global_percent)
    # 先根据BN层的权重初步获取每个模块的out_mask
    exc = []
    for m in model.model:
        if isinstance(m, nn.Upsample):
            exc.append(m.i - 1)
    for m in model.model:
        # 得到CBS模块的out_mask
        if m.type == 'models.common.Conv':
            # Upsample之前的卷积不进行剪枝
            if m.i not in exc:
                get_mask([m],thresh)
            else:
                get_mask([m])
        elif m.type == 'models.common.C3':
            # 得到C3结构的out_mask
            for sub_name, sub_module in m.sub_mod.items():
                if sub_name in ['cv1','cv2', 'cv3']:
                    get_mask([sub_module], thresh)
                elif sub_name is 'm':
                    for bottleneck in sub_module:
                        get_mask([bottleneck.cv1, bottleneck.cv2],thresh)
            # C3结构的out_mask是其中的cv3的out_mask
            m.out_mask = m.cv3.out_mask

        elif m.type == 'models.common.SPPF':
            # 得到SPPF结构的out_mask
            get_mask([m.cv1])
            get_mask([m.cv2],thresh)
            m.out_mask = m.cv2.out_mask
        elif m.type == 'models.common.Concat':
            if isinstance(m.f, list):
                m.out_mask = torch.cat((model.model[m.f[0] + m.i].out_mask,model.model[m.f[1]].out_mask),0)
        elif m.type == 'torch.nn.modules.upsampling.Upsample':
            m.out_mask = model.model[m.i + m.f].out_mask

    # 获得模型中个模块的input_mask，并根据一些规则修改out_mask
    for m in model.model:
        if m.type == 'models.common.Conv':
            if m.i == 0:
                m.input_mask = torch.ones(3) > 0
            else:
                m.input_mask = model.model[m.f + m.i].out_mask
        elif m.type == 'models.common.C3':
            C3_input_mask = model.model[m.f + m.i].out_mask
            m.input_mask = C3_input_mask
            # 上路分支的mask
            C3_top_out_mask = []
            for sub_name, sub_module in m.sub_mod.items():
                if sub_name == 'cv1':
                    sub_module.input_mask = C3_input_mask
                    C3_top_out_mask.append(sub_module.out_mask)
                elif sub_name == 'cv2':
                    sub_module.input_mask = C3_input_mask
                    C3_down_out_mask = sub_module.out_mask
                elif sub_name == 'm':
                    for bottleneck in sub_module:
                        C3_top_out_mask.append(bottleneck.cv2.out_mask)
                        if bottleneck.shortcut is True:
                            shortcut = True
                        else:
                            shortcut = False
                    if shortcut:
                        top_out_mask = torch.cat(C3_top_out_mask, 0).reshape(len(C3_top_out_mask),-1)
                        top_out_mask = torch.sum(top_out_mask, dim=0) > 0
                    else:
                        # 如果C3结构中的bottleneck结构没有shortcut连接，则上路分支是最后一个bottleneck的out_mask
                        top_out_mask = bottleneck.cv2.out_mask

                elif sub_name == 'cv3':
                    # 在第0维度上进行拼接，组成cv3的input_mask
                    sub_module.input_mask = torch.cat((top_out_mask, C3_down_out_mask),0)

            for sub_name, sub_module in m.sub_mod.items():
                if sub_name == 'cv1':
                    if shortcut:
                        sub_module.out_mask = top_out_mask
                    cv1_out_mask = sub_module.out_mask
                elif sub_name == 'm':
                    if shortcut:
                        for bottleneck in sub_module:
                            bottleneck.cv1.input_mask = top_out_mask
                            bottleneck.cv2.input_mask = bottleneck.cv1.out_mask
                            bottleneck.cv2.out_mask = top_out_mask
                    else:
                        for index, bottleneck in enumerate(sub_module):
                            if index == 0:
                                bottleneck.cv1.input_mask = cv1_out_mask
                            else:
                                bottleneck.cv1.input_mask = sub_module[index-1].cv2.out_mask
                            bottleneck.cv2.input_mask = bottleneck.cv1.out_mask

        elif m.type == 'models.common.SPPF':
            input_mask = model.model[m.f + m.i].out_mask
            m.cv1.input_mask = input_mask
            m.input_mask = input_mask
            m.cv2.input_mask = torch.cat([m.cv1.out_mask] * 4, 0)
        elif m.type == 'models.yolo.Detect':
            m.input_mask = []
            for i in range(len(m.f)):
                input_mask = model.model[m.f[i]].out_mask
                m.input_mask.append(input_mask)


    # 以上只是获得mask以及remain_ratio，下面会真正进行剪枝
    if Copy:
        pruned_model = deepcopy(model)
    else:
        pruned_model = model

    return doing_pruning(pruned_model)
    # print('params reduce ratio: {:.2f}%,flops reduce ratio: {:.2f}%'.format(100 * (params_before-params_after)/params_before,
    #                                                                100 * (flops_before - flops_after)/flops_before))

def doing_pruning(pruned_model):
    for m in pruned_model.model:
        if m.type == 'models.common.Conv':
            m.doing_pruning()
        elif m.type == 'models.common.C3':
            for sub_name, sub_module in m.sub_mod.items():
                if sub_name in ['cv1', 'cv2', 'cv3']:
                    sub_module.doing_pruning()
                elif sub_name == 'm':
                    for bottleneck in sub_module:
                        bottleneck.cv1.doing_pruning()
                        bottleneck.cv2.doing_pruning()
        elif m.type == 'models.common.SPPF':
            m.cv1.doing_pruning()
            m.cv2.doing_pruning()
        elif m.type == 'models.yolo.Detect':
            m.doing_pruning()
    params_after, flops_after = pruned_model.info()
    return pruned_model,flops_after
# is_training开启时该方法只是对模型在训练中先预剪枝查看效果,并未实际影响model的参数，关闭时就是正常的剪枝
# rand_remain_ratio:[最大值，最小值]
def prune(model, deepcopy=False, global_percent = 0.0):
    pruned_model,flops_compact = pruning(model,deepcopy,global_percent)
    return pruned_model, flops_compact


def update_yaml(d, model):
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)
    ch = [3]
    c2 = ch[-1]
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):
        m = eval(m) if isinstance(m, str) else m  # eval strings
        n = max(round(n * gd), 1) if n > 1 else n  # depth gain
        named_m_base = "model.{}".format(i)
        if m is Conv:
            args[-1] = model.model[i].remain_ratio
        elif m is SPPF:
            # 如果args[-1]是一个浮点数，那么就单指cv1的输出通道
            if isinstance(args[-1],float) or isinstance(args[-1],list) :
                temp = []
                temp.append(0.5 * model.model[i].cv1.remain_ratio)  # 0.5是原本的cv1的输出通道相对输入通道的缩放比例
                temp.append(model.model[i].cv2.remain_ratio)
                args[-1] = temp
        elif m is C3:
            args[-3][0] = 0.5 * model.model[i].cv1.remain_ratio
            args[-3][1] = 0.5 * model.model[i].cv2.remain_ratio
            args[-1] = model.model[i].cv3.remain_ratio
            for index, bottleneck in enumerate(model.model[i].m):
                # 添加bottleneck中的cv1和cv2输出比例，其中因为
                temp = []
                temp.append(bottleneck.cv1.remain_ratio)
                temp.append(bottleneck.cv2.remain_ratio)
                args[-2][index] = temp
    return d


# 迭代20次更新BN层统计信息
def adaptive_bn(compact_model,train_loader,device,cuda):
    with torch.no_grad():
        for i, (imgs, targets, paths, _) in tqdm(enumerate(train_loader), total=len(train_loader)):
            imgs = imgs.to(device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0

            # Forward
            with amp.autocast(enabled=cuda):
                pred = compact_model(imgs)  # forward
                # loss, loss_items = compute_loss(pred, targets.to(device))  # loss scaled by batch_size

            if i > 20:
                break

def adaptive_bn_eval(compact_model,train_loader, test_loader, device,data_dict, opt):
    compact_model.train()
    compact_model.half().float()
    cuda = device.type != 'cpu'
    # 通过不进行梯度更新改变模型的BN层信息，迭代训练共20次
    adaptive_bn(compact_model,train_loader,device,cuda)
    # 然后进行测验，这就是Eagleeye算法的核心思想
    results, _, _ = val.run(data_dict,
                            batch_size=opt.batch_size * 2,
                            imgsz=opt.imgsz,
                            model=compact_model,
                            conf_thres= 0.001,
                            iou_thres= 0.60,  # best pycocotools results at 0.65
                            single_cls=opt.single_cls,
                            dataloader=test_loader,
                            save_json=False,
                            plots=False)  # val best model with plots

    return results[2]

def get_prune_mAP(data, model, dataloader,  conf_thres=0.001, iou_thres=0.6, augment=False, save_hybrid = False,half=True):
    training = model is not None
    if training:  # called by train.py
        device, pt = next(model.parameters()).device, True  # get model device, PyTorch model

        half &= device.type != 'cpu'  # half precision only supported on CUDA
        model.half() if half else model.float()

    # Configure
    model.eval()
    is_coco = isinstance(data.get('val'), str) and data['val'].endswith('coco/val2017.txt')  # COCO dataset
    nc = data['nc'] # number of classes
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()
    seen = 0
    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    class_map = coco80_to_coco91_class() if is_coco else list(range(1000))
    s = ('%20s' + '%11s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    dt, p, r, f1, mp, mr, map50, map = [0.0, 0.0, 0.0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class = [], [], [], []
    pbar = tqdm(dataloader, desc=s, ncols=NCOLS, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar
    for batch_i, (im, targets, paths, shapes) in enumerate(pbar):
        t1 = time_sync()
        if pt:
            im = im.to(device, non_blocking=True)
            targets = targets.to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        nb, _, height, width = im.shape  # batch size, channels, height, width
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        out, train_out = model(im) if training else model(im, augment=augment, val=True)  # inference, loss outputs
        dt[1] += time_sync() - t2

        # NMS
        targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
        lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
        t3 = time_sync()
        out = non_max_suppression(out, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=False)
        dt[2] += time_sync() - t3

        # Metrics
        for si, pred in enumerate(out):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            path, shape = Path(paths[si]), shapes[si][0]
            seen += 1

            if len(pred) == 0:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Predictions
            predn = pred.clone()
            scale_coords(im[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # native-space pred

            # Evaluate
            if nl:
                tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                scale_coords(im[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
                correct = process_batch(predn, labelsn, iouv)
            else:
                correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool)
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))  # (correct, conf, pcls, tcls)

    # Compute metrics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats, plot=False, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    pf = '%20s' + '%11i' * 2 + '%11.3g' * 4  # print format
    LOGGER.info(pf % ('all', seen, nt.sum(), mp, mr, map50, map))
    model.float()
