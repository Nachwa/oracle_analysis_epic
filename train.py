import json, argparse
import numpy as np
import time
from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision import transforms
from torch.autograd import Variable
import torchvision.utils as vutils
from tensorboardX import SummaryWriter

from epic_db import EPIC_Dataset
from models import *
from utils.cyc_lr import CyclicLR

np.random.seed(2222)
torch.manual_seed(2222)

def prepare_label_hotencoding(labels, n_classes, dtype='int'):
    n_samples = len(labels)
    categorials = np.zeros((n_samples , n_classes), dtype)
    for l in labels:
        categorials[np.arange(n_samples), l] = 1
    return categorials

def get_classes_weights(loader):
    weights = torch.zeros(n_verbs).cuda()
    for batch in loader: 
        verbs = batch['verb_class']
        for i in range(verbs.shape[0]):
            weights[verbs[i]] += 1
    return  weights #/ len(loader)

def confusion_matrix(loader):
    confusion_mat = torch.zeros(n_verbs, n_verbs)
    for batch in loader: 
        verbs = batch['verb_class']
        X = Variable(torch.Tensor(objects)).cuda() if args.cuda else Variable(torch.Tensor(objects))
        
        #pass img to the model
        out = net(X)
        
        predictions = torch.argmax(out, dim=-1)
        confusion_mat[verbs, predictions] += 1
    return confusion_mat

def iteration_step(epoch, dataname, update_weights=True):
    #set model in training mode
    if update_weights: 
        net.train()
        loader = train_img_loader
    else:
        net.eval()
        loader = valid_img_loader
    
    #prepare evaluation metrics
    loss_lst, acc, batch_count = [], 0, 0
    for i_batch, data in enumerate(tqdm(loader)):
        verbs   = data['verb_class']
        #objects_list = data['objects_list_unique']  #object set exp MLP1, MLP2, MLP3
        #objects_list = data['objects_list_sorted']  #Ordered list exp MLP2
        #objects_list_hot   = data['objects_list_hot']
        objects_data = data[dataname]

        #for cyclicLR
        #scheduler.batch_step() 
        
        X = Variable(torch.Tensor(objects_data)).cuda() 
        Y = Variable(torch.LongTensor(verbs)).cuda()
       
        #pass img to the model
        out = net(X)

        if tr_weights is not None:         
            loss_fn = nn.CrossEntropyLoss(tr_weights) if update_weights else nn.CrossEntropyLoss(vl_weights)
        else: 
            loss_fn = nn.CrossEntropyLoss()

        loss = loss_fn(out, Y)

        if update_weights:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        #compute accuracy 
        loss_lst.append(loss.item())
        acc +=  torch.sum(Y == torch.argmax(out, dim=-1)).item()
        batch_count += len(X)

    acc /= batch_count
    return np.mean(loss_lst), acc


def save_checkpoint(state, filename):
    """Save checkpoint if a new best is achieved"""
    torch.save(state, filename)  # save checkpoint

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--expname', dest='expname', type=str,
                        default='exp_unnamed', help='name of experiment for tensorboard')
    parser.add_argument('--epochs', dest='epochs', type=int, 
                        default=50, help='number of epochs to run')
    parser.add_argument('--batch_size', dest='batch_size', type=int, 
                        default=64, help='batch size')
    return parser.parse_args()

if __name__ == '__main__':
    
    args = get_args()

    home_db_dir = '/media/nachwa/48d9ff99-04f7-4a80-ae30-8bd5a89069f8/Datasets/epic_kitchen/'
    work_db_dir =  '/media/naboubak/Maxtor/EPIC_KITCHENS_2018/' 
    db_root_dir = work_db_dir

    train_db = EPIC_Dataset(db_root_dir, training=True)
    valid_db = EPIC_Dataset(db_root_dir, training=False)
    
    train_img_loader = torch.utils.data.DataLoader(train_db, 
                                                batch_size=args.batch_size, 
                                                shuffle=True, 
                                                num_workers=64)
                                                #collate_fn=epic_db.collate_var_length)

    valid_img_loader = torch.utils.data.DataLoader(valid_db,
                                                batch_size=args.batch_size, 
                                                shuffle=True, 
                                                num_workers=64)
                                                #, collate_fn=epic_db.collate_var_length)

    noun_keywords, verb_keywords = train_db.noun_dict, train_db.verb_dict
    n_nouns, n_verbs = train_db.n_nouns, train_db.n_verbs

    print('Computing classes weights .. ')
    tr_weights = None #get_classes_weights(train_img_loader).cuda() if args.cuda else get_classes_weights(train_img_loader)
    vl_weights = None #get_classes_weights(valid_img_loader).cuda() 
    #print(vl_weights, torch.max(vl_weights), torch.sum(vl_weights))
    print('Initialize network archtitcure .. ')

    
    experiments = { 
        #modelname: (args, exp_name, data_toload, lr)
        #______________________
        #EXP1: Object unique set
         #'mlp_1'  : ([[n_nouns, 100, 500, 500, 100, n_verbs]], 'MLP_Mdl', 0.001), #29.46
         #'mlp_2'  : ([[n_nouns, 300, 200, 150, n_verbs]], 'MLP_Mdl', 0.001),  #28.46
         #'mlp_3'  : ([[n_nouns, 100, 50, 100, n_verbs]], 'MLP_Mdl', 0.001), #28.07
         #______________________
        #EXP2: Object list sorted 'ordered_object
         #'mlp_2.1'  : ([[train_db.max_sequence_of_objects, 300, 200, 150, n_verbs]], 'MLP_Mdl', 'objects_list_sorted', 0.001),  #24.09
         #'mlp_2.2'  : ([[train_db.max_sequence_of_objects, 100, 50, 100, n_verbs]], 'MLP_Mdl', 'objects_list_sorted', 0.001), #25.24
         #'rnn_2.1'  : ([train_db.max_sequence_of_objects, n_verbs], 'simple_rnn_2', 'objects_list_sorted', 10e-4), #27.04
         #'lstm_2.1'  : ([train_db.max_sequence_of_objects, n_verbs, 3, 128, False], 'simple_lstm_2', 'objects_list_sorted', 10e-4), #27.68
         #'bilstm_2.1'  : ([train_db.max_sequence_of_objects, n_verbs, 3, 128, True], 'simple_lstm_2', 'objects_list_sorted', 10e-4), #29.05
         #'tcn_2.1'  : ([train_db.max_sequence_of_objects, n_verbs], 'simple_tcn_2', 'objects_list_sorted', 10e-4) #29.90
         #______________________
        #EXP3: Object list ordered per frame
         #'mlp_2.1'  : ([[train_db.max_sequence_of_frames, 300, 200, 150, n_verbs]], 'MLP_Mdl', 'objects_list_sorted_perframe', 0.001),  #
         #'mlp_2.2'  : ([[train_db.max_sequence_of_frames, 100, 50, 100, n_verbs]], 'MLP_Mdl', 'objects_list_sorted_perframe', 0.001), #
         #'rnn_3.1'  : ([train_db.max_sequence_of_frames*n_nouns, n_verbs], 'simple_rnn_2', 'objects_list_sorted_perframe', 10e-4), #35.16
         #'lstm_3.1'  : ([train_db.max_sequence_of_frames*n_nouns, n_verbs, 3, 128, False], 'simple_lstm_2', 'objects_list_sorted_perframe', 10e-4), #34.01
         #'bilstm_3.1'  : ([train_db.max_sequence_of_frames*n_nouns, n_verbs, 3, 128, True], 'simple_lstm_2', 'objects_list_sorted_perframe', 10e-4), #34.77
         #'tcn_3.1'  : ([train_db.max_sequence_of_frames, n_verbs], 'simple_tcn_3', 'objects_list_sorted_perframe_noreshape', 10e-4) #32.44
         #'tcn_3.2'  : ([train_db.max_sequence_of_frames, n_verbs], 'simple_tcn_skip_2', 'objects_list_sorted_perframe_noreshape', 10e-4), #35.00
        
         'rnn_3.2'  : ([train_db.max_sequence_of_frames*n_nouns, n_verbs, 3], 'simple_rnn_2', 'objects_list_sorted_perframe', 10e-4), #
         'lstm_3.2'  : ([train_db.max_sequence_of_frames*n_nouns, n_verbs, 1, 128, False], 'simple_lstm_2', 'objects_list_sorted_perframe', 10e-4), #
         
    #______________________
        #EXP4: Objects per frame and time
         #'rnn_4.1'  : ([train_db.max_sequence_of_frames*(n_nouns+4), n_verbs], 'simple_rnn_2', 'objects_per_frame_with_time', 10e-4), #36.74
         #'lstm_4.1'  : ([train_db.max_sequence_of_frames*(n_nouns+4), n_verbs, 3, 128, False], 'simple_lstm_2','objects_per_frame_with_time', 10e-4), #32.83
         #'bilstm_4.1'  : ([train_db.max_sequence_of_frames*(n_nouns+4), n_verbs, 3, 128, True], 'simple_lstm_2', 'objects_per_frame_with_time', 10e-4), #32.19
         #'tcn_4.1'  : ([train_db.max_sequence_of_frames, n_verbs], 'simple_tcn_3', 'objects_list_sorted_perframe_noreshape', 10e-4),#32.02
         #'tcn_4.2'  : ([train_db.max_sequence_of_frames, n_verbs], 'simple_tcn_skip_2', 'objects_per_frame_with_time_noreshape', 10e-4) #36.98
         #'lstm_4.2'  : ([train_db.max_sequence_of_frames*(n_nouns+4), n_verbs, 1, 128, False], 'simple_lstm_2', 'objects_per_frame_with_time', 10e-4), #36.19
         #'bilstm_4.2'  : ([train_db.max_sequence_of_frames*(n_nouns+4), n_verbs, 1, 128, True], 'simple_lstm_2', 'objects_per_frame_with_time', 10e-4), #35.98
        
        'rnn_4.2'  : ([train_db.max_sequence_of_frames*(n_nouns+4), n_verbs, 3], 'simple_rnn_2', 'objects_per_frame_with_time', 10e-4), #
        
    #______________________
        #EXP5: Objects per frame and time and object scoring center
         #'rnn_5.1'  : ([train_db.max_sequence_of_frames*(n_nouns+4), n_verbs], 'simple_rnn_2', 'objects_score_per_frame_with_time', 10e-4), #34.92
         #'lstm_5.2'  : ([train_db.max_sequence_of_frames*(n_nouns+4), n_verbs, 1, 128, False], 'simple_lstm_2', 'objects_score_per_frame_with_time', 10e-4), #33.35
         #'bilstm_5.2'  : ([train_db.max_sequence_of_frames*(n_nouns+4), n_verbs, 1, 128, True], 'simple_lstm_2', 'objects_score_per_frame_with_time', 10e-4), #34.63
         #'tcn_5.2'  : ([train_db.max_sequence_of_frames, n_verbs], 'simple_tcn_skip_2', 'objects_score_per_frame_with_time_noreshape', 10e-4), #36.77
         #'lstm_5.1'  : ([train_db.max_sequence_of_frames*(n_nouns+4), n_verbs, 3, 128, False], 'simple_lstm_2','objects_score_per_frame_with_time', 10e-4), #31.64
         #'bilstm_5.1'  : ([train_db.max_sequence_of_frames*(n_nouns+4), n_verbs, 3, 128, True], 'simple_lstm_2', 'objects_score_per_frame_with_time', 10e-4), #31.53
         #'tcn_5.1'  : ([train_db.max_sequence_of_frames, n_verbs], 'simple_tcn_3', 'objects_score_per_frame_with_time_noreshape', 10e-4),#33.23
         #'rnn_5.2'  : ([train_db.max_sequence_of_frames*(n_nouns+4), n_verbs, 3], 'simple_rnn_2', 'objects_score_per_frame_with_time', 10e-4), #35.13
    #______________________
        #EXP6: Objects per frame and time and object scoring with hand
        #  'rnn_6.1'  : ([train_db.max_sequence_of_frames*(n_nouns+4), n_verbs], 'simple_rnn_2', 'objects_score_per_frame_with_time', 10e-4), #34.25, 30.18
        #  'lstm_6.2'  : ([train_db.max_sequence_of_frames*(n_nouns+4), n_verbs, 1, 128, False], 'simple_lstm_2', 'objects_score_per_frame_with_time', 10e-4), #34.12

         'bilstm_6.2'  : ([train_db.max_sequence_of_frames*(n_nouns+4), n_verbs, 1, 128, True], 'simple_lstm_2', 'objects_score_per_frame_with_time', 10e-4), #33.86
        
        #  'tcn_6.2'  : ([train_db.max_sequence_of_frames, n_verbs], 'simple_tcn_skip_2', 'objects_score_per_frame_with_time_noreshape', 10e-4), #32.23
        #  'lstm_6.1'  : ([train_db.max_sequence_of_frames*(n_nouns+4), n_verbs, 3, 128, False], 'simple_lstm_2','objects_score_per_frame_with_time', 10e-4), #29.58
        #  'bilstm_6.1'  : ([train_db.max_sequence_of_frames*(n_nouns+4), n_verbs, 3, 128, True], 'simple_lstm_2', 'objects_score_per_frame_with_time', 10e-4), #30.05
        #  'tcn_6.1'  : ([train_db.max_sequence_of_frames, n_verbs], 'simple_tcn_3', 'objects_score_per_frame_with_time_noreshape', 10e-4),#33.12
        #  'rnn_6.2'  : ([train_db.max_sequence_of_frames*(n_nouns+4), n_verbs, 3], 'simple_rnn_2', 'objects_score_per_frame_with_time', 10e-4), #33.88
    #______________________
        #EXP7: Objects per frame and time with state
        # 'rnn_7.1'  : ([train_db.max_sequence_of_frames*(n_nouns+4), n_verbs], 'simple_rnn_2', 'objects_state_per_frame_with_time', 10e-4), #54.35 , v6=56.56
        # 'lstm_7.2'  : ([train_db.max_sequence_of_frames*(n_nouns+4), n_verbs, 1, 128, False], 'simple_lstm_2', 'objects_state_per_frame_with_time', 10e-4), #56.37
        # 'bilstm_7.2'  : ([train_db.max_sequence_of_frames*(n_nouns+4), n_verbs, 1, 128, True], 'simple_lstm_2', 'objects_state_per_frame_with_time', 10e-4), #57.38
        # 'tcn_7.2'  : ([train_db.max_sequence_of_frames, n_verbs], 'simple_tcn_skip_2', 'objects_state_per_frame_with_time_noreshape', 10e-4), #72.90
        # 'lstm_7.1'  : ([train_db.max_sequence_of_frames*(n_nouns+4), n_verbs, 3, 128, False], 'simple_lstm_2','objects_state_per_frame_with_time', 10e-4), #56.22
        # 'bilstm_.1'  : ([train_db.max_sequence_of_frames*(n_nouns+4), n_verbs, 3, 128, True], 'simple_lstm_2', 'objects_state_per_frame_with_time', 10e-4), #58.36

        'tcn_7.1'  : ([train_db.max_sequence_of_frames, n_verbs], 'simple_tcn_3', 'objects_state_per_frame_with_time_noreshape', 10e-4),#
        'rnn_7.2'  : ([train_db.max_sequence_of_frames*(n_nouns+4), n_verbs, 3], 'simple_rnn_2', 'objects_state_per_frame_with_time', 10e-4), #
         
        }

    str_res = ''
    for exp_name in experiments: 
        arguments, mdl_name, dataname,lr = experiments[exp_name]
        print('Loading Model : ', mdl_name, ' for exp : ', exp_name, ' with args : ', arguments)
        mdl = eval(mdl_name)
        net = mdl(*arguments).cuda()
        print(net)

        optimizer = torch.optim.Adam(net.parameters(), lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5) 

        epoch_n, best_acc = 0, 0

        print('Start training .. ')
        for epoch in range(args.epochs):

            train_loss, train_acc = iteration_step(epoch = epoch_n, dataname=dataname, update_weights=True)   
            print(f'epoch ({epoch_n:2d}): train [loss = {train_loss:.03f}, acc = {train_acc*100:2.02f}]')
            
            valid_loss, valid_acc = iteration_step(epoch = epoch_n, dataname=dataname, update_weights=False)
            print(f'valid [loss = {valid_loss:.03f}, acc = {valid_acc*100:2.02f}]')
            
            scheduler.step(valid_loss)

            if best_acc <  valid_acc * 100: 
                best_acc = valid_acc * 100
                save_checkpoint({
                        'epoch': epoch_n,
                        'net_arch' : str(net),
                        'state_dict': net.state_dict(),
                        'best_acc': best_acc,
                        'optimizer' : optimizer.state_dict(),
                        'model_name' : mdl_name,
                        }, filename=f'./new_runs/{args.expname}_{exp_name}_best_model.pth')
            
            epoch_n += 1

        str_res += f'Best model for {exp_name} is {best_acc:2.02f} \n'
        print(str_res)
    
