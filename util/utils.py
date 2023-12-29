r""" Helper functions """
import random

import torch
import numpy as np
import cv2
from matplotlib import pyplot as plt
import os

def fix_randseed(seed):
    r""" Set random seeds for reproducibility """
    if seed is None:
        seed = int(random.random() * 1e5)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def mean(x):
    return sum(x) / len(x) if len(x) > 0 else 0.0


def to_cuda(batch):
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.cuda()
    return batch


def to_cpu(tensor):
    return tensor.detach().clone().cpu()

###########################################################showout
path = '/home/liuyu/code/SHSNet/savefig/'

def showout(map, index,name,count):
    feature_save_path = os.path.join(path, str(name))
    if not os.path.exists(feature_save_path):
        os.makedirs(feature_save_path)
    pred = map.squeeze()
    pred = pred.detach().cpu().numpy()
    plt.imshow(pred, interpolation='bicubic',cmap='gray')
    plt.axis('off')
    save_name = feature_save_path + '/' +str(count)+ '_' + str(index) + '.png'
    plt.savefig(save_name)
    plt.show()

def show_image(map, name, count):
    path = '/home/liuyu/code/SHSNet/savefig/'
    feature_save_path = os.path.join(path, str(name))
    if not os.path.exists(feature_save_path):
        os.makedirs(feature_save_path)
    plt.imshow(map)
    plt.axis('off')
    save_name = feature_save_path + '/' +str(count)+ '_' + 'image' + '.jpg'
    plt.savefig(save_name)

def show(img,mask,overlay_color):
    # blue 51 102 255
    # green 204 255 204
    # yellow 255,230,153
    overlay_color = overlay_color
    transparency = 0.5
    plt.ion()
    img = img
    mask = mask
    mask = mask // np.max(mask)
    im_over = np.ndarray(img.shape)
    im_over[:, :, 0] = (1 - mask) * img[:, :, 0] + mask * (
            overlay_color[0] * transparency + (1 - transparency) * img[:, :, 0])
    im_over[:, :, 1] = (1 - mask) * img[:, :, 1] + mask * (
            overlay_color[1] * transparency + (1 - transparency) * img[:, :, 1])
    im_over[:, :, 2] = (1 - mask) * img[:, :, 2] + mask * (
            overlay_color[2] * transparency + (1 - transparency) * img[:, :, 2])

    return im_over.astype(np.uint8)

def save_result(support,s_label,query,q_label,query_pred,supp_pred,i_iter,group,q_heat,s_heat):
    result_dir = os.path.join(path, 'result',str(group),'pred')
    result_dir1 = os.path.join(path, 'result', str(group),'heatmap')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    if not os.path.exists(result_dir1):
        os.makedirs(result_dir1)
    support = support.squeeze().detach().cpu().numpy()
    s_label = s_label.squeeze().detach().cpu().numpy()
    query = query.squeeze().detach().cpu().numpy()
    q_label = q_label.squeeze().detach().cpu().numpy()
    cv2.imwrite('%s/%s_Sb.png' % (result_dir, i_iter), support)
    cv2.imwrite('%s/%s_qb.png' % (result_dir, i_iter), query)
    overlay_color = [0, 255, 0]
    overlay_color1 = [255,0, 0]
    support_save = show(support,s_label,overlay_color)
    cv2.imwrite('%s/%s_Sa.png' % (result_dir, i_iter), support_save)
    query_save =show(query,q_label,overlay_color)
    cv2.imwrite('%s/%s_qa.png' % (result_dir, i_iter), query_save)
    quer_presave =show(query,query_pred,overlay_color1)
    cv2.imwrite('%s/%s_qpred.png' % (result_dir, i_iter), quer_presave)
    supp_presave = show(support, supp_pred, overlay_color1)
    cv2.imwrite('%s/%s_spred.png' % (result_dir, i_iter), supp_presave)
    cv2.imwrite('%s/%s_sheat.png' % (result_dir1, i_iter), s_heat)
    cv2.imwrite('%s/%s_qheat.png' % (result_dir1, i_iter), q_heat)

def vis_heatmap(img,feature,name,count):
    result_dir1 = os.path.join(path, str(name))
    if not os.path.exists(result_dir1):
        os.makedirs(result_dir1)
    img = img.squeeze().detach().cpu().numpy()
    img_size = img.shape[1]
    # feature = torch.softmax(feature,1)
    feature = (feature ** 2).sum(1)
    # feature = feature.squeeze(0)[0]

    for j in range(feature.size(0)):
        fm = feature[j].detach().cpu().numpy()
        fm = 1-fm
        #  activate map
        fm = cv2.resize(fm, (img_size, img_size))
        fm = 255 * (fm - np.min(fm)) / (
                np.max(fm) - np.min(fm) + 1e-12
        )

        # bbox = localize_from_map(fm, threshold_ratio=1.0)
        fm = np.uint8(np.floor(fm))
        fm = cv2.applyColorMap(fm, cv2.COLORMAP_JET)


        overlapped = img * 0.3 + fm * 0.7
        overlapped[overlapped > 255] = 255
        # overlapped = draw_bbox(overlapped, [bbox])
        overlapped = overlapped.astype(np.uint8)

        grid_img = 255 * np.ones(
            (img_size, 3 * img_size + 2 * 10, 3), dtype=np.uint8
        )
        grid_img[:, :img_size, :] = img[:, :, ::-1]
        grid_img[:, img_size + 10:2 * img_size + 10, :] = fm
        grid_img[:, 2 * img_size + 2 * 10:, :] = overlapped
        cv2.imshow("", grid_img)
        cv2.waitKey(0)
        cv2.imwrite('%s/%s_heat.png' % (path, count), grid_img)