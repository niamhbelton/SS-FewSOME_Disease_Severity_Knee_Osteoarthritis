
import torch
import os
import numpy as np
import pandas as pd
import argparse
import torch.nn.functional as F
import torch.optim as optim
import random
import time
from datasets.oa_knee import oa
from torch.utils.data import DataLoader
from model import *
from evaluate import *
import torchvision.transforms as transforms
from sklearn.metrics import roc_curve, auc, roc_auc_score, precision_recall_fscore_support, f1_score
from utils import *
import torch.multiprocessing
from train import *
torch.multiprocessing.set_sharing_strategy('file_system')



TRAIN_PLATEAU_EPOCH = 400

def ss_training(args, model_temp_name_ss, N, epochs, num_ss, shots, self_supervised, semi, seed = None, eval_epoch = 1): #trains the model and evaluates every 10 epochs for all seeds OR trains the model for a specific number of epochs for specified seed

  val_dataset =  oa(args.data_path, task = 'test_on_train', train_info_path = args.train_ids_path)
  if args.ss_test:
      test_dataset =  oa(args.data_path, task = args.task)
  else:
      test_dataset = None

  if seed == None:
     seeds =[1001, 138647, 193, 34, 44, 71530, 875688, 8765, 985772, 244959]
  else:
      seeds =[seed]

  current_epoch = 0
  for seed in seeds:
      model = ALEXNET_nomax_pre().to(args.device)
      train_dataset =  oa(args.data_path, task='train', stage='ss', N = N, shots = shots, semi = semi, self_supervised = self_supervised, num_ss = num_ss, augmentations = args.augmentations, normal_augs = args.normal_augs, train_info_path = args.train_ids_path, seed = seed)
      train(train_dataset, val_dataset, N, model, epochs, seed, eval_epoch, shots, model_name_temp_ss + '_seed_' + str(seed), args, current_epoch, metric='centre_mean', patches =True, test_dataset = test_dataset )
      del model

  return './outputs/dfs/ss/', './outputs/logs/ss/'




def dclr_training(args, model_temp_name_stage, stage, pseudo_label_ids, epochs, num_ss, current_epoch, self_supervised = 1, semi= 0):

    val_dataset =  oa(args.data_path, task = 'test_on_train', train_info_path = args.train_ids_path)
    if args.stage3_test:
        test_dataset =  oa(args.data_path, task = args.task)
    else:
        test_dataset = None

    if stage == 'stage2':
        seeds =[1001, 138647, 193, 34, 44, 71530, 875688, 8765, 985772, 244959]
    else:
        seeds =[ 1001, 138647, 193, 34, 44]
    for seed in seeds:
        model = vgg16().to(args.device)
        train_dataset =  oa(args.data_path, task='train', stage='stage3', semi = semi, self_supervised = self_supervised, num_ss = num_ss, augmentations = args.augmentations, normal_augs = args.normal_augs, train_info_path = args.train_ids_path, seed = seed, pseudo_label_ids = pseudo_label_ids)
        N = train_dataset.N
        shots = train_dataset.shots
        if isinstance(current_epoch, dict):
            ep = current_epoch[str(seed)]
        else:
            ep=current_epoch
        train(train_dataset, val_dataset, N, model, epochs, seed, args.eval_epoch, shots, model_temp_name_stage + '_seed_' + str(seed) +  '_N_' + str(N), args, ep, metric='w_centre', patches = False, test_dataset = test_dataset )
        del model
    current_epoch = get_best_epoch('./outputs/logs/' + stage + '/', last_epoch = current_epoch, metric='ref_centre')

    return current_epoch, './outputs/dfs/' + stage + '/'


def parse_arguments():
    parser = argparse.ArgumentParser()

    #what stages to perform
    parser.add_argument('--train_ss', type=int, default=1)
    parser.add_argument('--stage2', type=int, default = 1)
    parser.add_argument('--stage3', type=int, default = 1)
    parser.add_argument('--stage_severe_pred', type=int, default = 1)


    #the same for all models
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--lr', type=float, default=1e-6)
    parser.add_argument('--bs', type=int, default=1)
    parser.add_argument('--k', type=int, default=1)
    parser.add_argument('--train_ids_path', type=str, default='../meta/')
    parser.add_argument('--task', type=str, default='test')
    parser.add_argument('--eval_epoch', type=int, default=1)
    parser.add_argument('--data_path', type=str, default='/home/niamh/knee/archive-4/')
    parser.add_argument('--model_name', type=str, default='mod_1')
    parser.add_argument('--augmentations', type=str, default="crop, cutpaste")
    parser.add_argument('--normal_augs', type=str, default="sharp, bright, jitter")
    parser.add_argument('--seed', type=int, default=1001)


    #details of data if not running train_ss
    parser.add_argument('--stage1_path_to_anom_scores', type=str, default = './outputs/dfs/ss/')
    parser.add_argument('--stage_severe_pred_path_to_anom_scores', type=str, default = './outputs/dfs/stage_severe_pred/')
    parser.add_argument('--stage2_path_to_anom_scores', type=str, default = './outputs/dfs/stage2/')
    parser.add_argument('--stage3_path_to_anom_scores', type=str, default = './outputs/dfs/stage3/')
    parser.add_argument('--stage1_path_to_logs', type=str, default = './outputs/logs/ss/')
    parser.add_argument('--stage_severe_pred_path_to_logs', type=str, default = './outputs/logs/stage_severe_pred/')
    parser.add_argument('--stage2_path_to_logs', type=str, default = './outputs/logs/stage2/')
    parser.add_argument('--stage3_path_to_logs', type=str, default = './outputs/logs/stage3/')


    #epochs and N for each stage
    parser.add_argument('--ss_N', type=int, default=30)
    parser.add_argument('--ss_epochs', type=int, default=300)
    parser.add_argument('--stage_severe_pred_N', type=int, default=30)
    parser.add_argument('--stage_severe_pred_epochs', type=int, default=990)
    parser.add_argument('--stage2_N', type=int, default=30)
    parser.add_argument('--stage2_epochs', type=int, default=1000)
    parser.add_argument('--stage3_epochs', type=int, default=1000)

    #patching parameters for ss stage
    parser.add_argument('--padding', type=int, default=0)
    parser.add_argument('--patchsize', type=int, default=3)
    parser.add_argument('--stride', type=int, default=1)

    #evaluate on test set or not
    parser.add_argument('--ss_test', type=int, default=1)
    parser.add_argument('--stage2_test', type=int, default = 1)
    parser.add_argument('--stage3_test', type=int, default=1)


    parser.add_argument('--save_models', type=int, default=0)
    parser.add_argument('--save_anomaly_scores', type=int, default=1)
    parser.add_argument('--meta_data_dir', type=str, default = '../meta/kxr_sq_bu00.txt' )

    parser.add_argument('--margin', type=float, default = 1.184)
    parser.add_argument('--severe_margin', type=float, default = 3.122)

    args = parser.parse_args()
    return args



if __name__ == '__main__':

  args = parse_arguments()

  torch.use_deterministic_algorithms(True, warn_only=True)
  torch.manual_seed(args.seed)
  torch.cuda.manual_seed(args.seed)
  torch.cuda.manual_seed_all(args.seed)


  if not os.path.exists('./outputs'):
       os.makedirs('./outputs')

  if not os.path.exists('./outputs/label_details/'):
       os.makedirs('./outputs/label_details/')

  if not os.path.exists('./outputs/results'):
       os.makedirs('./outputs/results')

  if not os.path.exists('./outputs/dfs'):
       os.makedirs('./outputs/dfs')

  if not os.path.exists('./outputs/models'):
       os.makedirs('./outputs/models')

  if not os.path.exists('./outputs/logs'):
       os.makedirs('./outputs/logs')

  if not os.path.exists('./outputs/oarsi'):
           os.makedirs('./outputs/oarsi')

  stages = ['ss', 'stage_severe_pred', 'stage2', 'stage3']
  for stage in stages:
      if not os.path.exists('./outputs/results/' + stage):
           os.makedirs('./outputs/results/' + stage)

      if not os.path.exists('./outputs/dfs/' + stage):
           os.makedirs('./outputs/dfs/' + stage)

      if not os.path.exists('./outputs/models/' + stage):
           os.makedirs('./outputs/models/' + stage)

      if not os.path.exists('./outputs/logs/' + stage):
           os.makedirs('./outputs/logs/' + stage)

      if not os.path.exists('./outputs/oarsi/' + stage):
               os.makedirs('./outputs/oarsi/' + stage)



  model_name_temp = args.model_name  + '_bs_' + str(args.bs) + '_task_' + str(args.task)+  '_lr_' + str(args.lr)

  if args.train_ss:
      model_name_temp_ss = stages[0] + '/' + 'ss_training_' + 'margin_' + str(args.margin) + '_' + model_name_temp  + '_N_' + str(args.ss_N)
      stage1_path_to_anom_scores, stage1_path_to_logs = ss_training(args, model_name_temp_ss, N=args.ss_N, epochs =args.ss_epochs, num_ss = args.ss_N, shots=0, self_supervised=1, semi=0)
  else:
      stage1_path_to_anom_scores = args.stage1_path_to_anom_scores
      stage1_path_to_logs = args.stage1_path_to_logs
      if args.eval_epoch == 0: #if not evaluated every ten epochs, there are no anoms scores for the best performing epoch
          for key in current_epoch.keys(): #retraining for specific epoch to write out anomaly scores
             _,_,_ = ss_training(args, model_name_temp_ss, N=args.ss_N, epochs = current_epoch[key], num_ss = args.ss_N, shots=0, self_supervised=1, semi=0, seed = key, eval_epoch = 0 )


  #stage2 is to DCLR-FewSOME_OA ITER 1
  if args.stage2:
      pseudo_label_ids = get_pseudo_labels(args.train_ids_path, stage1_path_to_anom_scores, args.data_path, margin = args.margin, metric = 'centre_mean', current_epoch=TRAIN_PLATEAU_EPOCH)
      pd.DataFrame(pseudo_label_ids).to_csv('./outputs/label_details/dclr_fewsome_OA_iter1_pseudo_anom_labels.csv')
      shots = len(pseudo_label_ids)
      model_name_temp_stage2 =  stages[2] + '/' + 'stage2_' + 'margin_' + str(args.margin) + '_' + model_name_temp + '_shots_' + str(shots)  + '_N_' + str(args.stage2_N)
      current_epoch, stage2_path_to_scores = dclr_training(args, model_name_temp_stage2, stages[2], pseudo_label_ids= pseudo_label_ids, N=args.stage2_N, epochs = args.stage2_epochs, current_epoch = TRAIN_PLATEAU_EPOCH, num_ss=0, shots=shots, self_supervised = 0, semi= 1)
  else:
      stage2_path_to_anom_scores = args.stage2_path_to_anom_scores
      stage2_path_to_logs = args.stage2_path_to_logs
      current_epoch = get_best_epoch(args.stage2_path_to_logs, last_epoch = TRAIN_PLATEAU_EPOCH, metric='ref_centre')

  #stage3 is to DCLR-FewSOME_OA ITER 2
  if args.stage3:
        pseudo_label_ids =  get_pseudo_labels(args.train_ids_path, stage2_path_to_anom_scores, args.data_path, margin = args.margin, metric = 'w_centre', current_epoch=current_epoch)
        pd.DataFrame(pseudo_label_ids).to_csv('./outputs/label_details/dclr_fewsome_OA_iter2_pseudo_anom_labels.csv')
        shots = len(pseudo_label_ids)
        model_name_temp_stage3 =  stages[3] + '/' +  'stage3_' + 'margin_' + str(margin) + '_' + model_name_temp + '_shots_' + str(shots)
        current_epoch, stage3_path_to_scores = dclr_training(args, model_name_temp_stage3, stages[3], pseudo_label_ids = pseudo_label_ids, epochs = args.stage3_epochs, num_ss=0, current_epoch = current_epoch, self_supervised = 0, semi= 1)


  if args.stage_severe_pred:
      pseudo_label_ids = get_pseudo_labels(args.train_ids_path, stage1_path_to_anom_scores, args.data_path, margin = args.severe_margin, metric = 'centre_mean', current_epoch=TRAIN_PLATEAU_EPOCH)
      pd.DataFrame(pseudo_label_ids).to_csv('./outputs/label_details/dclr_fewsome_sev_pseudo_anom_labels.csv')
      shots = len(pseudo_label_ids)
      model_name_temp_stage2 =  stages[1] + '/' + 'stage_sev_pred_' + 'margin_' + str(args.severe_margin) + '_' + model_name_temp + '_shots_' + str(shots)  + '_N_' + str(args.stage_severe_pred_N)
      current_epoch, stage2_path_to_scores = dclr_training(args, model_name_temp_stage2, stages[1], pseudo_label_ids= pseudo_label_ids, N=args.stage_severe_pred_N, epochs = args.stage_severe_pred_epochs, current_epoch = TRAIN_PLATEAU_EPOCH, num_ss=0, shots=shots, self_supervised = 0, semi= 1)
