import comet_ml,os
from comet_ml import Experiment
import numpy as np
import pandas as pd
import torch.nn as nn
import torch,lz4.frame,math,os,sys,pickle,re
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.optim as optim
from random import shuffle
from sklearn.metrics import roc_auc_score
os.chdir('/usr/local/data/bbera/Shuffle_Learn/Experiments/CC_Perc_0/Package')
from Utils import TripletSiameseNet,evaluate,get_dict_preds


DATA_FOLDER_NAME = 'loris_3x3x3_v5'
DATAPATH = os.path.join('/usr/local/faststorage/datasets/',DATA_FOLDER_NAME)
PROJECT_NAME = 'Shuffle_Learn'
SERVER_NAME = 'CC'
PERCENTILE = 0
EXPERIMENT = f'Perc_{PERCENTILE}'
DEVICE = 'cuda:0'

BASEPATH = '/usr/local/data/bbera/'
DATAPATH = '/usr/local/faststorage/datasets/'

CHECKPOINT_PATH = os.path.join(BASEPATH,f'Shuffle_Learn/Experiments/{SERVER_NAME}_Perc_{PERCENTILE}/checkpoints')
SAVED_RESULTS_PATH = os.path.join(BASEPATH,PROJECT_NAME,'Saved_Results')

dict_paths = {'checkpoints':CHECKPOINT_PATH,'saved_results':SAVED_RESULTS_PATH,
              'dict_triplets':os.path.join(SAVED_RESULTS_PATH,'dict_triplets',f'dict_triplets_perc{PERCENTILE}.pickle'),
              'basepath':BASEPATH,'experiment':EXPERIMENT,'server_name':SERVER_NAME,
              'project_name':PROJECT_NAME,'datapath':DATAPATH,'data_folder_name':DATA_FOLDER_NAME}

if DEVICE=='cuda:0':
    list_checkpoint_files = os.listdir(CHECKPOINT_PATH)[:440]
elif DEVICE=='cuda:1':
    list_checkpoint_files = os.listdir(CHECKPOINT_PATH)[440:]
    
def get_auc_val(file_checkpoint,subj_set):
    if isinstance(file_checkpoint,str):
        file_checkpoint_list = [file_checkpoint]
    else:
        file_checkpoint_list = file_checkpoint
    model_list = []
    for file_checkpoint in file_checkpoint_list:
        checkpoint = torch.load(os.path.join(CHECKPOINT_PATH,file_checkpoint),map_location='cpu')
        model = TripletSiameseNet()
        model.load_state_dict({k[len('module.'):]: v for k, v in checkpoint['model'].items()})
        # model.load_state_dict(checkpoint['model'])
        model =  model.to(DEVICE)
        model_list.append(model)

    return evaluate(model_list,dict_paths,img_set=subj_set,n_sample='all',device=DEVICE)

    
# dict_validate_auc = {}
# for file_checkpoint in list_checkpoint_files:
#     if 'checkpoint_' not in file_checkpoint:continue
#     tot_iter = int(file_checkpoint.split('_')[-1].replace('.pth.tar',''))
#     if tot_iter<5000: continue
    
#     auc_val = get_auc_val(file_checkpoint,subj_set='validate')
#     dict_validate_auc[file_checkpoint] = auc_val
#     print(file_checkpoint,' - ',auc_val)
    
#     device_str = DEVICE.replace(':','_')
#     with open(f'/usr/local/data/bbera/Shuffle_Learn/Saved_Results/dict_validate_auc_{device_str}.pickle', 'wb') as handle:
#         pickle.dump(dict_validate_auc, handle, protocol=pickle.HIGHEST_PROTOCOL)


with open('/usr/local/data/bbera/Shuffle_Learn/Saved_Results/dict_validate_auc_cuda_0.pickle', 'rb') as handle:
    dict_validate_auc_cuda_0 = pickle.load(handle)
with open('/usr/local/data/bbera/Shuffle_Learn/Saved_Results/dict_validate_auc_cuda_1.pickle', 'rb') as handle:
    dict_validate_auc_cuda_1 = pickle.load(handle)

dict_validate_auc = dict_validate_auc_cuda_0.copy()
dict_validate_auc.update(dict_validate_auc_cuda_1)

best_val_res = {}
for file_checkpoint,auc_val in dict_validate_auc.items():
    if len(best_val_res)<100:
        best_val_res[file_checkpoint] = auc_val
    else:
        key_min = min(best_val_res,key=best_val_res.get)
        del best_val_res[key_min]
        best_val_res[file_checkpoint] = auc_val
        assert len(best_val_res)==100

# dict_test_auc = {}
# for file_checkpoint in tqdm(best_val_res.keys()):
#     auc_test = get_auc_val(file_checkpoint,subj_set='test')
#     dict_test_auc[file_checkpoint] = auc_test
#     print(file_checkpoint,' - ',auc_test)
    
# device_str = DEVICE.replace(':','_')
# with open(f'/usr/local/data/bbera/Shuffle_Learn/Saved_Results/dict_test_auc_{device_str}.pickle', 'wb') as handle:
#     pickle.dump(dict_test_auc, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(dict_paths['dict_triplets'], 'rb') as handle:
    dict_triplets = pickle.load(handle)['test']
    

dict_res = {'id_subj':[],
            'yhat_shuffle':[],
            'yhat_sorted':[],
            'label_sorted':[],
            'label_shuffle':[],
            'tot_iter':[]}
for file_checkpoint in best_val_res.keys():
    print(f'############# {file_checkpoint} #############')
    tot_iter = int(file_checkpoint.split('_')[-1].replace('.pth.tar',''))
    model = TripletSiameseNet()
    checkpoint = torch.load(os.path.join(CHECKPOINT_PATH,file_checkpoint),map_location='cpu')
    model.load_state_dict({k[len('module.'):]: v for k, v in checkpoint['model'].items()})
    model =  model.to(DEVICE)
    sample_keys = list(dict_triplets.keys())
    for id_subj in tqdm(sample_keys):
        dict_preds = get_dict_preds(id_subj,model,dict_paths,img_set='test',device=DEVICE)
        for key in dict_preds.keys():
            dict_res[key].append(dict_preds[key])
        dict_res['tot_iter'].append(tot_iter)
    
    pd.DataFrame.from_dict(dict_res).to_csv('/usr/local/data/bbera/Shuffle_Learn/Saved_Results/df_res_test.csv')
    
    
########### plotting and statistical analysis ################

df_res_test = pd.read_csv('/usr/local/data/bbera/Shuffle_Learn/Saved_Results/df_res_test.csv')
df_clinic = pd.read_csv('/usr/local/data/bbera/Clinical_csv_generation_data/df_clinic.csv')

df_clinic.columns
df_clinic['CLINICAL_EDSS_Score_Observed'].unique()

df_res_test = df_res_test[~df_res_test['id_subj'].isin(df_clinic.loc[df_clinic['SUBJECT_Trial_Arm']=='MBP8298','subj_id'].values)]
df_clinic_test = df_clinic[df_clinic['subj_id'].isin(df_res_test['id_subj'].values)]

list_mean,list_tot_iter = [],[]
for tot_iter in df_res_test['tot_iter'].unique():
    df_temp = df_res_test[(df_res_test['tot_iter']==tot_iter)]
    list_mean.append(df_temp.apply(lambda x: 1 if x['yhat_shuffle']<x['yhat_sorted'] else 0,axis=1).mean())
    list_tot_iter.append(tot_iter)

tot_iter = list_tot_iter[np.argmax(list_mean)]
df_temp = df_res_test[(df_res_test['tot_iter']==tot_iter)]
df_temp.apply(lambda x: 1 if x['yhat_shuffle']<x['yhat_sorted'] else 0,axis=1).mean()



df_temp['contain_w000'] = df_temp['label_sorted'].apply(lambda x: 1 if 'w000' in x else 0).values
df_temp['Label_correct'] = df_temp.apply(lambda x: 1 if x['yhat_shuffle']<x['yhat_sorted'] else 0,axis=1).values

df_temp.groupby('contain_w000').agg({'Label_correct':np.mean})
df_temp['contain_w000'].value_counts()

# df_temp['correct_w000_w001'] = 

def check_first_two_w(x):
    x_split = x.split('_')
    if int(x_split[0].replace('w',''))<int(x_split[1].replace('w','')):
        return True
    else:
        return False

df_temp['fist_two_w_sorted'] = df_temp['label_shuffle'].apply(lambda x: 1 if check_first_two_w(x) else 0)
df_temp.groupby('fist_two_w_sorted').agg({'Label_correct':np.mean})
df_temp['fist_two_w_sorted'].value_counts()

df_temp.loc[df_temp['fist_two_w_sorted']==1,'label_shuffle']

check_first_two_w(df_temp['label_shuffle'].values[0])




res = []
for treat_arm in df_clinic['SUBJECT_Trial_Arm'].unique():
    if treat_arm not in df_clinic_test['SUBJECT_Trial_Arm'].values:continue
    list_subj_id = list(set(df_clinic.loc[df_clinic['SUBJECT_Trial_Arm']==treat_arm,'subj_id'].values))
    df_temp = df_res_test[(df_res_test['tot_iter']==tot_iter)&(df_res_test['id_subj'].isin(list_subj_id))]
    df_mean = df_temp.apply(lambda x: 1 if x['yhat_shuffle']<x['yhat_sorted'] else 0,axis=1)
    print(treat_arm,df_mean.mean(),df_mean.shape[0])
    res.append(df_mean.mean())

np.mean(res)

for sex in df_clinic['SUBJECT_Sex'].unique():
    if sex not in df_clinic_test['SUBJECT_Sex'].values:continue
    list_subj_id = list(set(df_clinic.loc[df_clinic['SUBJECT_Sex']==sex,'subj_id'].values))
    df_temp = df_res_test[(df_res_test['tot_iter']==tot_iter)&(df_res_test['id_subj'].isin(list_subj_id))]
    df_mean = df_temp.apply(lambda x: 1 if x['yhat_shuffle']<x['yhat_sorted'] else 0,axis=1)
    print(sex,df_mean.mean(),df_mean.shape[0])


df_clinic['SUBJECT_Screening_Visit_MS_Type'].value_counts()
df_clinic['MS_groups'] = df_clinic['SUBJECT_Screening_Visit_MS_Type'].apply(lambda x: 'RR' if x in ['RRMS','RMS'] else 'SPP')
df_clinic_test['MS_groups'] = df_clinic_test['SUBJECT_Screening_Visit_MS_Type'].apply(lambda x: 'RR' if x in ['RRMS','RMS'] else 'SPP')
for ms in df_clinic['MS_groups'].unique():
    if ms not in df_clinic_test['MS_groups'].values:continue
    list_subj_id = list(set(df_clinic.loc[df_clinic['MS_groups']==ms,'subj_id'].values))
    df_temp = df_res_test[(df_res_test['tot_iter']==tot_iter)&(df_res_test['id_subj'].isin(list_subj_id))]
    df_mean = df_temp.apply(lambda x: 1 if x['yhat_shuffle']<x['yhat_sorted'] else 0,axis=1)
    print(ms,df_mean.mean(),df_mean.shape[0])

df_clinic['CDP_010d_EDSS_Status_for_Trial'].unique()
for cdp in df_clinic['CDP_010d_EDSS_Status_for_Trial'].unique():
    if cdp not in df_clinic_test['CDP_010d_EDSS_Status_for_Trial'].values:continue
    list_subj_id = list(set(df_clinic.loc[df_clinic['CDP_010d_EDSS_Status_for_Trial']==cdp,'subj_id'].values))
    df_temp = df_res_test[(df_res_test['tot_iter']==tot_iter)&(df_res_test['id_subj'].isin(list_subj_id))]
    df_mean = df_temp.apply(lambda x: 1 if x['yhat_shuffle']<x['yhat_sorted'] else 0,axis=1)
    print(cdp,df_mean.mean(),df_mean.shape[0])



np.percentile(df_clinic_test['REFERENCE_T25FW_Mean_sec'].dropna().values,q=range(0,110,10))

df_temp = df_res_test[df_res_test['tot_iter']==tot_iter]
y_true = df_temp.apply(lambda x: 1 if x['yhat_shuffle']<x['yhat_sorted'] else 0,axis=1)
df_clin_temp = df_clinic_test.loc[df_clinic_test['subj_id'].isin(df_temp['id_subj'].values),'REFERENCE_T25FW_Mean_sec'].dropna()
y_true = y_true[df_clin_temp.index].values
yhat = df_clin_temp['REFERENCE_T25FW_Mean_sec'].values
roc_auc_score(y_true,yhat)

yhat.shape


for t25fw in df_clinic['REFERENCE_T25FW_Mean_sec'].unique():
    if t25fw not in df_clinic_test['REFERENCE_T25FW_Mean_sec'].dropna().values:continue
    list_subj_id = list(set(df_clinic.loc[df_clinic['REFERENCE_T25FW_Mean_sec']==t25fw,'subj_id'].values))
    df_temp = df_res_test[(df_res_test['tot_iter']==tot_iter)&(df_res_test['id_subj'].isin(list_subj_id))]
    df_mean = df_temp.apply(lambda x: 1 if x['yhat_shuffle']<x['yhat_sorted'] else 0,axis=1)
    print(t25fw,df_mean.mean(),df_mean.shape[0])

df_clinic['subj_id'].unique().shape
df_res_test['id_subj'].unique().shape

df_clinic['REFERENCE_T25FW_Mean_sec'].isna().sum()/df_clinic.shape[0]



df_clinic_test['REFERENCE_T25FW_Mean_sec'].shape
