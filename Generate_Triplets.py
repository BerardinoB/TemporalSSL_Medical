import numpy as np
import torch.nn as nn
import torch,lz4.frame,math,os,sys,pickle,re
from itertools import permutations
from tqdm import tqdm

def create_dict_timepoints(subj_set):
    dict_timepoints = {}
    for id_subj in split['experiment'][subj_set].keys():
        path_temp = os.path.join(datapath,id_subj[0],id_subj[1])
        list_timepoints = [f for f in os.listdir(path_temp) if 'w' in f and 
                        'e_' not in f and 
                        os.path.exists(os.path.join(path_temp,f,'flr.npy.lz4'))]
        if len(list_timepoints)<3: continue
        list_permute = list(permutations(list_timepoints,3))
        dict_timepoints[id_subj[0]+'_'+id_subj[1]] = list_permute
    return dict_timepoints

def sum_squared_diff(triplet,id_subj):
    squared_diff_list = []
    if len(id_subj.split('_'))==2:
        id_subj = id_subj.split('_')
    elif len(id_subj.split('_'))==3:
        id_subj = id_subj.split('_')
        id_subj = ['_'.join(id_subj[:2]),id_subj[-1]]
    else:
        assert False
    for i,x in enumerate(triplet[:-1]):
        with lz4.frame.open(os.path.join(datapath,id_subj[0],id_subj[1],x,'flr.npy.lz4'), 'rb') as f:
            X = np.load(f)
        with lz4.frame.open(os.path.join(datapath,id_subj[0],id_subj[1],x,'Beast.npy.lz4'), 'rb') as f:
            X_mask = np.load(f)
            X = np.multiply(X,X_mask)
            X_std = (X-np.mean(X))/np.std(X)
        for y in triplet[i+1:]:
            with lz4.frame.open(os.path.join(datapath,id_subj[0],id_subj[1],y,'flr.npy.lz4'), 'rb') as f:
                Y = np.load(f)
            with lz4.frame.open(os.path.join(datapath,id_subj[0],id_subj[1],y,'Beast.npy.lz4'), 'rb') as f:
                Y_mask = np.load(f)
            Y = np.multiply(Y,Y_mask)
            Y_std = (Y-np.mean(Y))/np.std(Y)
            squared_diff_list.append(((X_std-Y_std)**2).sum())
    return np.array(squared_diff_list).sum()


def create_dict_subj_ssd(dict_timepoints):
    dict_subj_ssd = {}
    for id_subj,list_triplets in tqdm(dict_timepoints.items()):
        n_unique = len(set([x for x_tup in dict_timepoints[id_subj] for x in x_tup]))
        dict_triplet_ssd = {}
        for i,triplet in enumerate(list_triplets):
            triplet_str = '_'.join(list(triplet))
            if n_unique!=3 or (n_unique==3 and i==0):
                ssdiff = sum_squared_diff(triplet,id_subj)
            dict_triplet_ssd[triplet_str] = ssdiff
        dict_subj_ssd[id_subj] = dict_triplet_ssd
    return dict_subj_ssd

def find_threshold(perc):
    return np.percentile([x for subj_dict in dict_subj_ssd['train'].values() for x in subj_dict.values()],q=perc)

def generate_dict_triplets(dict_subj_ssd,p_val):
    dict_triplets = {}
    for img_set,dict_ssd in dict_subj_ssd.items():
        dict_triplets[img_set] = {}
        for subj,dict_triplet_ssd in dict_ssd.items():
            list_valid_triplets = [trip for trip,val in dict_triplet_ssd.items() if val>=p_val]
            dict_triplets[img_set][subj] = list_valid_triplets
        dict_triplets[img_set] = {key:val for key,val in dict_triplets[img_set].items() if len(val)>0}
    return dict_triplets
