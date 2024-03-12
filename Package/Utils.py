from monai.networks.blocks import ResidualUnit
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
import lz4.frame,os,pickle,random,torch
from monai import transforms
from augmentation import CustomTransforms
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import operator


class Model(nn.Module):
    def __init__(self, in_channels=1, k=64, depth=4, units_per_level=2, dropout=0.2):
        assert (k % 8) == 0, "k should be a multiple of 8 for best performance"
        assert depth >= 1
        assert units_per_level >= 1

        super(Model, self).__init__()
        blocks = []

        for d in range(depth):
            if d > 0: # Add downsampling if at deeper level
                blocks.append(nn.MaxPool3d(2,2))

            for u in range(units_per_level):
                if d == 0 and u == 0: # First conv at input
                   in_ch = in_channels
                   out_ch = k
                elif d > 0 and u == 0: # First conv after downsampling
                    in_ch = k*(2**(d-1))
                    out_ch = k*(2**d)
                else:
                    in_ch = out_ch = k*(2**d)

                blocks.append(ResidualUnit(spatial_dims=3,
                                           in_channels=in_ch,
                                           out_channels=out_ch,
                                           dropout_dim=3,
                                           dropout=dropout,
                                          )
                             )

        blocks.append(nn.AdaptiveAvgPool3d(1))
        blocks.append(nn.Flatten())

        self.model  = nn.Sequential(*blocks)
        self.linear = nn.Linear(k*(2**(depth-1)), k*(2**(depth-2)))
        

    def forward(self, x):
        x = self.model(x)
        out = self.linear(x)
        # out = self.out(f)
        return out
    
class TripletSiameseNet(nn.Module):
    def __init__(self,k=64, depth=4):
        super(TripletSiameseNet, self).__init__()
        self.branch = Model()
        self.addedClassifier = nn.Linear(in_features=k*(2**(depth-2))*3, out_features=1, bias=True)

    def forward(self, x):
        out1 = self.branch(x[:, :1, :, :, :])
        out2 = self.branch(x[:, 1:2, :, :, :])
        out3 = self.branch(x[:, 2:3, :, :, :])
        out = self.addedClassifier(torch.cat([out1, out2, out3], dim=1))
        return out

def load_image(path: str):
    with lz4.frame.open(path, 'rb') as f:
        return np.load(f)

class CustomDataset(Dataset):
    def __init__(self,dict_paths,data_folder):
        self.datapath = os.path.join(dict_paths['datapath'],data_folder)
        self.id_subj_path_list = []
        with open(dict_paths['dict_triplets'], 'rb') as handle:
            self.dict_triplets = pickle.load(handle)['train']
        
    def __len__(self):
        return len(self.dict_triplets)
    
    def __getitem__(self, idx):
        id_subj = list(self.dict_triplets.keys())[idx].split('_')
        if len(id_subj)==3:
            id_subj = ['_'.join(id_subj[:2]),id_subj[2]]
        while True:
            img_triplets = random.choice(self.dict_triplets['_'.join(id_subj)]).split('_')
            if sum(map(operator.eq, img_triplets, sorted(img_triplets)))!=3:
                break
        list_img = []
        for t in img_triplets:
            subj_path = os.path.join(self.datapath,id_subj[0],id_subj[1],t,'flr.npy.lz4')
            list_img.append(load_image(subj_path))
            subj_mask_path = os.path.join(self.datapath,id_subj[0],id_subj[1],t,'Beast.npy.lz4')
            list_img.append(load_image(subj_mask_path))
        return np.array(list_img),'_'.join(img_triplets)
    
def data_preproc(data,idx,data_folder):
    DATA_FOLDER_NAME = data_folder
    # spatial_size_dict = {'loris_3x3x3_v5':[-1, 64, 80, 64]}
    spatial_size_dict = {'loris_3x3x3_v5':[-1, 64, 80, 64],
                        # 'loris_2x2x2_v4_0':[-1, 80, 112, 80],
                        'loris_1x1x3_v5':[-1, 64, 224, 176]}
    spatial_pad_dict = {'keys':['MRI','MASK'],'spatial_size': spatial_size_dict[DATA_FOLDER_NAME]}
    spatial_transform = transforms.SpatialPadd(**spatial_pad_dict)
    
    data_dict = {'MRI':data[:,idx['MRI'],:,:,:],'MASK':data[:,idx['MASK'],:,:,:]}
    data_dict = spatial_transform(data_dict)
    data_dict['MRI'] = CustomTransforms.denoise(data_dict['MRI'], data_dict['MASK'])
    data_dict['MRI'] = CustomTransforms.standardize(data_dict['MRI'], data_dict['MASK'])
    
    return data_dict['MRI']

def sort_data(data,labels,return_labels=True):
    data_sorted,labels_sorted = [],[]
    for i,l in enumerate(labels):
        l_sorted = sorted(l.split('_'))
        assert sum(map(operator.eq, l_sorted, l.split('_')))!=3
        idx_sorted = [l.split('_').index(x) for x in l_sorted]
        data_sorted.append(data[i:i+1,idx_sorted,:,:,:])
        labels_sorted.append('_'.join(l_sorted))
    if return_labels:
        return torch.cat(data_sorted,dim=0),labels_sorted
    else:
        return torch.cat(data_sorted,dim=0)

    
def load_img_triplets(dict_paths,id_subj,img_set='test'):
    with open(os.path.join(dict_paths['dict_triplets']), 'rb') as handle:
        dict_triplets = pickle.load(handle)[img_set]
    
    list_img,list_label = [],[]
    id_subj = id_subj.split('_')
    if len(id_subj)==3:
        id_subj = ['_'.join(id_subj[:2]),id_subj[2]]
    while True:
        img_triplets = random.choice(dict_triplets['_'.join(id_subj)]).split('_')
        if sum(map(operator.eq, img_triplets, sorted(img_triplets)))!=3:
            break
    for t in img_triplets:
        subj_path = os.path.join(dict_paths['datapath'],dict_paths['data_folder_name'],id_subj[0],id_subj[1],t,'flr.npy.lz4')
        list_img.append(load_image(subj_path))
        subj_mask_path = os.path.join(dict_paths['datapath'],dict_paths['data_folder_name'],id_subj[0],id_subj[1],t,'Beast.npy.lz4')
        list_img.append(load_image(subj_mask_path))
    list_label.append('_'.join(img_triplets))
    return np.array(list_img),list_label    
    

def get_dict_preds(id_subj,model,dict_paths,img_set,device='cpu'):
    idx = {'MRI':[0,2,4],'MASK':[1,3,5]}
    data, label_shuffle = load_img_triplets(dict_paths,id_subj,img_set=img_set)
    data = torch.from_numpy(np.expand_dims(data, axis=0))
    X_shuffle = data_preproc(data,idx,dict_paths['data_folder_name'])
    X_sorted,label_sorted = sort_data(X_shuffle,label_shuffle,return_labels=True)
    X_shuffle = X_shuffle.to(device)
    X_sorted = X_sorted.to(device)
    yhat_shuffle = torch.sigmoid(model(X_shuffle)).detach().cpu().item()
    yhat_sorted = torch.sigmoid(model(X_sorted)).detach().cpu().item()
    X_shuffle = X_shuffle.detach().cpu().numpy()
    X_sorted = X_sorted.detach().cpu().numpy()
    torch.cuda.empty_cache()
    return {'id_subj':id_subj,
            'yhat_shuffle':yhat_shuffle,
            'yhat_sorted':yhat_sorted,
            'label_sorted':label_sorted[0],
            'label_shuffle':label_shuffle[0]}


def evaluate(model,dict_paths,img_set='test',n_sample=100,device='cuda:0'):
    if not isinstance(model,(list,tuple)):
        model_list = [model]
    else:
        model_list = model
    print('Evaluate...')
    assert img_set in ['validate','test'], 'img_set can be either of [validate,test]'
    idx = {'MRI':[0,2,4],'MASK':[1,3,5]}
    with open(dict_paths['dict_triplets'], 'rb') as handle:
        dict_triplets = pickle.load(handle)[img_set]
    
    if n_sample=='all':
        sample_keys = list(dict_triplets.keys())
    else:
        assert isinstance(n_sample,int)
        sample_keys = random.sample(list(dict_triplets.keys()),n_sample)

    yhat,y = [],[]
    list_dict_model_pred = []
    for id_subj in tqdm(sample_keys):
        list_model_temp = []
        for model in model_list:
            list_model_temp.append(get_dict_preds(id_subj,model,dict_paths,img_set,device=device))
        list_dict_model_pred.append(list_model_temp)
    yhat_shuffle,yhat_sorted = [],[]
    for list_dict_pred in list_dict_model_pred:
        yhat_shuffle_temp,yhat_sorted_temp = [],[]
        for dict_res in list_dict_pred:
            yhat_shuffle_temp.append(dict_res['yhat_shuffle'])
            yhat_sorted_temp.append(dict_res['yhat_sorted'])
        yhat_shuffle.append(yhat_shuffle_temp)
        yhat_sorted.append(yhat_sorted_temp)
    yhat = list(np.array(yhat_shuffle).mean(axis=1)) + list(np.array(yhat_sorted).mean(axis=1))
    y = list(np.zeros(len(yhat_shuffle_temp))) + list(np.ones(len(yhat_sorted_temp)))
        
    return roc_auc_score(y,yhat)
    
    
