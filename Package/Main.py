import numpy as np
import torch.nn as nn
import torch,os
from tqdm import tqdm
from Utils import CustomDataset,TripletSiameseNet,sort_data,data_preproc,evaluate
from torch.utils.data import DataLoader
import torch.optim as optim
from torch import autocast

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import argparse

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Training SSL Model")
    
    parser.add_argument("--checkpoint_path", help="Path to save checkpoints", default='./checkpoint')
    parser.add_argument("--epochs", help="N epochs", default=200)
    parser.add_argument("--cuda_id", help="devise for training", default=0)
    parser.add_argument("--batch_size", help="batch size for training", default=32)
    parser.add_argument("--learning_rate", help="default 3e-4", default=3e-4)
    parser.add_argument("--parallel", help="parallelize training", default=False)
    parser.add_argument("--n_sample_eval", help="N of sample to use for validation", default="all")
    parser.add_argument("--resume_checkpoint", help="Resume Training", default=None)

    # Parse the command-line arguments
    args = parser.parse_args()

    # Access the parsed arguments
    DATAPATH = './Data'
    SAVED_RESULTS_PATH = './Results'
    CHECKPOINT_PATH = args.checkpoint_path
    EPOCH = int(args.epochs)
    device = 'cuda:'+str(args.cuda_id)
    BATCH_SIZE = int(args.batch_size)
    LEARNING_RATE = args.learning_rate
    N_SAMPLE_VAL = args.n_sample_eval
    CHECKPOINT = args.resume_checkpoint
    PARALLEL = True if isinstance(args.cuda_id,list) else False

    dict_paths = {'checkpoints':CHECKPOINT_PATH,'saved_results':SAVED_RESULTS_PATH,
              'dict_triplets':os.path.join(SAVED_RESULTS_PATH,'dict_triplets',f'dict_triplets.pickle'),
              'datapath':DATAPATH,}

    dataset = CustomDataset(dict_paths,data_folder=DATAPATH)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = TripletSiameseNet()
    
    if CHECKPOINT!=None:
        checkpoint = torch.load(os.path.join(CHECKPOINT_PATH,CHECKPOINT),map_location={'cuda:0':'cuda:0'})
        try:
            model.load_state_dict(checkpoint['model'])
        except:
            model.load_state_dict({k[len('module.'):]: v for k, v in checkpoint['model'].items()})
    
    if PARALLEL:
        model = nn.DataParallel(model,device_ids = args.cuda_id)
    
    bce_loss = torch.nn.BCEWithLogitsLoss()
    opt = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model = model.to(device)

    if CHECKPOINT!=None:
        opt.load_state_dict(checkpoint['opt'])
    
    scaler = torch.cuda.amp.GradScaler()
    idx = {'MRI':[0,2,4],'MASK':[1,3,5]}
    iter_tot = -1
    for epoch in range(EPOCH):
        for data,labels in tqdm(train_loader):
            if data.shape[0]!=BATCH_SIZE:continue
            iter_tot += 1
            
            X_shuffle = data_preproc(data,idx,data_folder=DATAPATH)
            X_sorted,labels_sorted = sort_data(X_shuffle,labels)
            
            X = torch.cat([X_sorted,X_shuffle],dim=0).to(device)
            y = torch.from_numpy(np.vstack([np.ones((BATCH_SIZE,1)),np.zeros((BATCH_SIZE,1))])).to(device)
            
            model = model.to(device=device)
            model = model.train()
            model.zero_grad()
            
            with autocast(device_type='cuda', dtype=torch.float16):
                yhat = model(X)
                loss = bce_loss(yhat,y)
                
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            
            if iter_tot%100==0 and iter_tot>100:
                model.eval()

                auc_val = evaluate(model,dict_paths,data_folder=DATAPATH,img_set='validate',n_sample=N_SAMPLE_VAL,device=device)

                torch.save({
                        'epoch':epoch,
                        'iter_tot':iter_tot,
                        'model':model.cpu().state_dict(),
                        'opt':opt.state_dict(),
                    },os.path.join(CHECKPOINT_PATH,'best_model_checkpoint.pth.tar'))








# DATA_FOLDER_NAME = 'loris_3x3x3_v5'
# DATAPATH = os.path.join('/usr/local/faststorage/datasets/',DATA_FOLDER_NAME)
# BATCH_SIZE = 16
# EPOCH = 300
# LEARNING_RATE = 3e-4
# PROJECT_NAME = 'Shuffle_Learn_git'
# SERVER_NAME = 'Odyssey'
# PERCENTILE = 0
# EXPERIMENT = f'Perc_{PERCENTILE}'

# BASEPATH = '/usr/local/data/bbera/'
# DATAPATH = '/usr/local/faststorage/datasets/'
    
# perc_str = str(PERCENTILE).replace('.','_')
# CHECKPOINT_PATH = os.path.join(BASEPATH,f'{PROJECT_NAME}/Experiments/{SERVER_NAME}_Perc_{PERCENTILE}/checkpoints')
# SAVED_RESULTS_PATH = os.path.join(BASEPATH,PROJECT_NAME,'Saved_Results')
# PARALLEL = True
# CUDA_INDEX = 0
# DEVICE_ID = [0,1]
# COMET_ML_LOGGING = False
# RESUME_TRAINING = False
# N_SAMPLE_VAL = 100

# dict_paths = {'checkpoints':CHECKPOINT_PATH,'saved_results':SAVED_RESULTS_PATH,
#               'dict_triplets':os.path.join(SAVED_RESULTS_PATH,'dict_triplets',f'dict_triplets_perc{PERCENTILE}.pickle'),
#               'basepath':BASEPATH,'experiment':EXPERIMENT,'server_name':SERVER_NAME,
#               'project_name':PROJECT_NAME,'datapath':DATAPATH,'data_folder_name':DATA_FOLDER_NAME}

# path_experiment_key = os.path.join(CHECKPOINT_PATH,f'experiment_{EXPERIMENT.lower()}_key.pickle') 

# if COMET_ML_LOGGING:
#     if RESUME_TRAINING:
#         assert os.path.exists(path_experiment_key), f'No experiment key found in {path_experiment_key}'
#         with open(path_experiment_key, 'rb') as handle:
#             EXPERIMENT_KEY = pickle.load(handle)
        
#         experiment = comet_ml.ExistingExperiment(
#             api_key="o6ADebaoKVqiUM2SjkATfllQC",
#             experiment_key=EXPERIMENT_KEY[PROJECT_NAME]
#         )
#     else:
#         experiment = Experiment(
#                     api_key="o6ADebaoKVqiUM2SjkATfllQC",
#                     project_name=PROJECT_NAME,
#                     workspace="bbarile",
#                 )
#         experiment.set_name(SERVER_NAME+'_'+EXPERIMENT)
#         EXPERIMENT_KEY = experiment.get_key()
#         with open(path_experiment_key, 'wb') as handle:
#             pickle.dump({PROJECT_NAME:EXPERIMENT_KEY}, handle, protocol=pickle.HIGHEST_PROTOCOL)

# # dict_paths['dict_triplets'] = os.path.join(dict_paths['dict_triplets'],f'dict_triplets_train_{EXPERIMENT.lower()}.pickle')
# dataset = CustomDataset(dict_paths,data_folder=DATA_FOLDER_NAME)
# train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# idx = {'MRI':[0,2,4],'MASK':[1,3,5]}

# if PARALLEL:
#     device = f'cuda:{DEVICE_ID[0]}'
# else:
#     device = 'cuda:'+str(CUDA_INDEX)

# model = TripletSiameseNet()

# if isinstance(DEVICE_ID,list):
#     dict_map_checkpoint = {f'cuda:{DEVICE_ID[0]}': device}
# elif isinstance(DEVICE_ID,str):
#     dict_map_checkpoint = {f'cuda:{DEVICE_ID}': device}


# if RESUME_TRAINING:
#     checkpoint = torch.load(os.path.join(CHECKPOINT_PATH,'checkpoint_'+EXPERIMENT+'iter_tot_1600.pth.tar'),map_location=dict_map_checkpoint)
#     try:
#         model.load_state_dict(checkpoint['model'])
#     except:
#         model.load_state_dict({k[len('module.'):]: v for k, v in checkpoint['model'].items()})
    
    
# if PARALLEL:
#     model = nn.DataParallel(model,device_ids = DEVICE_ID)
    
# bce_loss = torch.nn.BCEWithLogitsLoss()
# opt = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# model = model.to(device)

# if RESUME_TRAINING:
#     opt.load_state_dict(checkpoint['opt'])

# scaler = torch.cuda.amp.GradScaler()

# iter_tot = -1
# for epoch in range(EPOCH):
#     for data,labels in tqdm(train_loader):
#         if data.shape[0]!=BATCH_SIZE:continue
#         iter_tot += 1
        
#         X_shuffle = data_preproc(data,idx,data_folder=DATA_FOLDER_NAME)
#         X_sorted,labels_sorted = sort_data(X_shuffle,labels)
        
#         X = torch.cat([X_sorted,X_shuffle],dim=0).to(device)
#         y = torch.from_numpy(np.vstack([np.ones((BATCH_SIZE,1)),np.zeros((BATCH_SIZE,1))])).to(device)
        
#         model = model.to(device=device)
#         model = model.train()
#         model.zero_grad()
        
#         with autocast(device_type='cuda', dtype=torch.float16):
#             yhat = model(X)
#             loss = bce_loss(yhat,y)
            
#         scaler.scale(loss).backward()
#         scaler.step(opt)
#         scaler.update()
        
#         if COMET_ML_LOGGING and iter_tot%100==0 and iter_tot>100:
#             model.eval()

#             auc_val = evaluate(model,dict_paths,data_folder=DATA_FOLDER_NAME,img_set='validate',n_sample=N_SAMPLE_VAL,device=device)
            
#             experiment.log_metrics({
#                 'BCE_loss_train': loss.detach().cpu().item(),
#                 'AUC_validate': auc_val
#             })
        
        # if iter_tot%100==0 and iter_tot>100:
        #     torch.save({
        #             'epoch':epoch,
        #             'iter_tot':iter_tot,
        #             'model':model.cpu().state_dict(),
        #             'opt':opt.state_dict(),
        #         },os.path.join(CHECKPOINT_PATH,'checkpoint_'+EXPERIMENT+'iter_tot_'+str(iter_tot)+'.pth.tar'))
