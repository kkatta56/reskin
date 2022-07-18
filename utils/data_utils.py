import os 

import numpy as np
import torch

expt_dict = {
    2: 'const_depth_snake',
    3: 'shear_lines'
}

# FIX THESE SHITS   

def get_data_dirs(dirs, sensor_ids, expt_id):
    data_root = os.path.join('./expts',expt_dict[expt_id])
    dd_list = []
    for d in dirs:
        for sid in sensor_ids:
            curr_path = os.path.join(data_root,d,'sensor_'+str(sid))
            # print(curr_path)
            if os.path.exists(curr_path):
                dd_list.append(curr_path)    
    if len(dd_list) == 0:
        print('No data found!')
        return []
    
    return dd_list

def eval_xy_accuracy(preds, labels, res=2):
    # print('preds:', preds[0])
    scaled_preds = res * torch.round((16//res)*preds).detach().numpy()
    # print(scaled_preds)
    return np.mean(np.bitwise_and.reduce(
        scaled_preds == np.around(labels[...,:2].detach().numpy()), axis=-1))

 
def get_xy_confusion_mats_depths(preds, labels, depths, res=2):
    # print(depths)
    num_pts = (16//res) + 1
    int_depths = np.around(depths.detach().numpy() * (-5.0))
    conf_labels, conf_dist, conf_number = np.zeros((11,num_pts,num_pts)),np.zeros((11,num_pts,num_pts)),np.zeros((11,num_pts,num_pts))
    for i in range(11):
        # print(i,int_depths)
        reqd_ids = (int_depths == i)
        clab, cdist, cnum = get_xy_confusion_mats(preds[reqd_ids], labels[reqd_ids], res)
        conf_labels[i] = clab
        conf_number[i] = cnum
        conf_dist[i] = cdist
        # print(sum(reqd_ids))
        # print(cdist)
    return conf_labels, conf_dist, conf_number

def get_force_confusion_depths(F_preds, F_labels, labels, depths, res=2):
    num_pts = (16//res) + 1
    int_depths = np.around(depths.detach().numpy()* (-5.0))
    conf_Fz, conf_shear = np.zeros((11,num_pts,num_pts)),np.zeros((11,num_pts,num_pts))
    for i in range(11):
        reqd_ids = (int_depths == i)
        if np.sum(reqd_ids) > 0:
            cfz, cshear = get_force_confusion(
                F_preds[reqd_ids], F_labels[reqd_ids], labels[reqd_ids])
            conf_Fz[i] = cfz
            conf_shear[i] = cshear
    
    return conf_Fz, conf_shear
    
        

# Returns a spatial 
def get_xy_confusion_mats(preds, labels, res=2):
    num_pts = (16//res) + 1
    conf_labels = np.zeros((num_pts,num_pts))
    conf_dist = np.zeros((num_pts,num_pts))
    conf_number = np.zeros((num_pts,num_pts))

    semiscale_preds = torch.round((16//res)*preds).detach().numpy()
    det_labels = np.around(labels[...,:2].detach().numpy())
    err_preds = 1 - np.bitwise_and.reduce(
        (res)*semiscale_preds == det_labels, axis=-1)
    # print((res)*semiscale_preds, det_labels)
    for label, pred, err_pred in zip(det_labels, preds.detach().numpy(), err_preds):
        conf_labels[label[0].astype(int)//res,label[1].astype(int)//res] += 1 * err_pred
        conf_number[label[0].astype(int)//res,label[1].astype(int)//res] += 1
        conf_dist[label[0].astype(int)//res,label[1].astype(int)//res] += np.linalg.norm(16*pred - label)

    return conf_labels, conf_dist, conf_number

def get_force_confusion(F_preds, F_labels, labels, res=2):
    num_pts = (16//res) + 1
    conf_Fz = np.zeros((num_pts,num_pts))
    conf_shear = np.zeros((num_pts,num_pts))

    det_labels = np.around(labels[...,:2].detach().numpy())

    for label, F_pred, F_label in zip(det_labels, F_preds.detach().numpy(), F_labels.detach().numpy()):
        conf_Fz[label[0].astype(int)//(res),label[1].astype(int)//(res)] += np.linalg.norm(F_pred[0:1] - F_label[0:1])
        if F_preds.shape[-1] == 3:
            conf_shear[label[0].astype(int)//(res),label[1].astype(int)//(res)] += np.linalg.norm(F_pred[1:] - F_label[1:])
    
    return conf_Fz, conf_shear


