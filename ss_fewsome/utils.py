import pandas as pd
import numpy as np
import torch
import os
import random
from scipy.ndimage.filters import uniform_filter1d
from sklearn import metrics
from sklearn.metrics import roc_curve, auc, roc_auc_score, precision_recall_fscore_support, f1_score, cohen_kappa_score, average_precision_score
from scipy import stats
import torch.nn.functional as F

def get_best_epoch(path_to_centre_dists, last_epoch, metric):
    '''
        find the epoch where the metric value begins to plateau for each file in path_to_centre_dists
    '''
    files = os.listdir(path_to_centre_dists)
    files= [f for f in files if 'on_test_set' not in f]
    best_epochs = {}
    for i,file in enumerate(files):
        logs = pd.read_csv(path_to_centre_dists + file)[metric]
        if len(logs) > 20:
            seed = file.split('_seed_')[1].split('_')[0]
            smoothF = uniform_filter1d(logs, size = 20)
            dist_zero = np.abs(0 - np.gradient(smoothF))
            if isinstance(last_epoch, dict):
                best_epochs[seed] = (np.where(dist_zero == np.min(dist_zero)) [0][0] * 10 ) + last_epoch[seed]
            else:
                best_epochs[seed] = (np.where(dist_zero == np.min(dist_zero)) [0][0] * 10 ) + last_epoch

    return best_epochs




def get_pseudo_labels(train_ids_path, path_to_anom_scores, data_path, margin, margin_upper, metric, current_epoch = None):

    files_total = os.listdir(path_to_anom_scores)
    if isinstance(current_epoch, dict):
        files=[]
        for key in current_epoch.keys():
            f = [f for f in files_total if ( ('epoch_' + str(current_epoch[key]) in f) &  ('seed_' + str(key) in f)  &    ('removed_' not in f) &    ('on_test_set_' not in f)  ) ][0]
            files.append(f)

    else:
        files = [f for f in files_total if ( ('epoch_' + str(current_epoch) in f) &   ('removed_' not in f) &    ('on_test_set_' not in f)  ) ]
        print(files)
        assert len(files) == 10



    seeds = [1001,71530,138647,875688,985772,44,34,193,244959,8765]
    for i,seed in enumerate(seeds):

        for file in files:
          if (('_' + str(seed) + '_') in file) :
            logs = pd.read_csv(path_to_anom_scores + file)

            if i ==0:
                df = logs.iloc[:,:3]

            scores = logs[['id',metric]]
            if metric == 'w_centre':
                scores.loc[:,metric] = (scores.loc[:,metric] + 2) / 4
            else:
                scores.loc[:,metric] = (scores.loc[:,metric] + 1) / 2

            df = df.merge(scores, on='id', how='left')

            df.columns = np.concatenate((df.columns.values[:-1] , np.array(['col_{}'.format(i)])))

            df['col_{}'.format(i)] = df['col_{}'.format(i)] / np.max(df['col_{}'.format(i)])



    df['av']=df.iloc[:,3:].mean(axis=1)
    df['std'] = df.iloc[:,3:-1].std(axis=1)

    t0=pd.read_csv(train_ids_path + "train_seed_1001", index_col=0)
    t1=pd.read_csv(train_ids_path + "train_seed_71530", index_col=0)
    t2=pd.read_csv(train_ids_path + "train_seed_138647", index_col=0)
    t3=pd.read_csv(train_ids_path + "train_seed_875688", index_col=0)
    t4=pd.read_csv(train_ids_path + "train_seed_985772", index_col=0)
    t5=pd.read_csv(train_ids_path + "train_seed_44", index_col=0)
    t6=pd.read_csv(train_ids_path + "train_seed_34", index_col=0)
    t7=pd.read_csv(train_ids_path + "train_seed_193", index_col=0)
    t8=pd.read_csv(train_ids_path + "train_seed_244959", index_col=0)
    t9=pd.read_csv(train_ids_path + "train_seed_8765", index_col=0)


    df['id'] = df['id'].apply(lambda x: x.split('/')[-2] + '/' + x.split('/')[-1] )
    t0['id'] = t0['id'].apply(lambda x: x.split('/')[-2] + '/' + x.split('/')[-1] )
    t0=t0.merge(df, on='id', how='left')
    t1['id'] = t1['id'].apply(lambda x: x.split('/')[-2] + '/' + x.split('/')[-1] )
    t1=t1.merge(df, on='id', how='left')
    t2['id'] = t2['id'].apply(lambda x: x.split('/')[-2] + '/' + x.split('/')[-1] )
    t2=t2.merge(df, on='id', how='left')
    t3['id'] = t3['id'].apply(lambda x: x.split('/')[-2] + '/' + x.split('/')[-1] )
    t3=t3.merge(df, on='id', how='left')
    t4['id'] = t4['id'].apply(lambda x: x.split('/')[-2] + '/' + x.split('/')[-1] )
    t4=t4.merge(df, on='id', how='left')
    t5['id'] = t5['id'].apply(lambda x: x.split('/')[-2] + '/' + x.split('/')[-1] )
    t5=t5.merge(df, on='id', how='left')
    t6['id'] = t6['id'].apply(lambda x: x.split('/')[-2] + '/' + x.split('/')[-1] )
    t6=t6.merge(df, on='id', how='left')
    t7['id'] = t7['id'].apply(lambda x: x.split('/')[-2] + '/' + x.split('/')[-1] )
    t7=t7.merge(df, on='id', how='left')
    t8['id'] = t8['id'].apply(lambda x: x.split('/')[-2] + '/' + x.split('/')[-1] )
    t8=t8.merge(df, on='id', how='left')
    t9['id'] = t9['id'].apply(lambda x: x.split('/')[-2] + '/' + x.split('/')[-1] )
    t9=t9.merge(df, on='id', how='left')

    df['count'] = 0

    for i in range(len(df)):

        for j in range(len(files)):
            if j ==0:
                if df.iloc[i, 3+j] > (margin *np.max(t0['av'])):
                    df.loc[i,'count']+=1

            if j ==1:
                if df.iloc[i, 3+j] > (margin *np.max(t1['av'])):
                    df.loc[i,'count']+=1

            if j ==2:
                if df.iloc[i, 3+j] > (margin *np.max(t2['av'])):
                    df.loc[i,'count']+=1

            if j ==3:
                if df.iloc[i, 3+j] > (margin *np.max(t3['av'])):
                    df.loc[i, 'count']+=1

            if j ==4:
                if df.iloc[i, 3+j] > (margin *np.max(t4['av'])):
                    df.loc[i,'count']+=1

            if j ==5:
                if df.iloc[i, 3+j] > (margin *np.max(t5['av'])):
                    df.loc[i,'count']+=1

            if j ==6:
                if df.iloc[i, 3+j] > (margin *np.max(t6['av'])):
                    df.loc[i,'count']+=1

            if j ==7:
                if df.iloc[i, 3+j] > (margin *np.max(t7['av'])):
                    df.loc[i,'count']+=1

            if j ==8:
                if df.iloc[i, 3+j] > (margin *np.max(t8['av'])):
                    df.loc[i,'count']+=1

            if j ==9:
                if df.iloc[i, 3+j] > (margin *np.max(t9['av'])):
                    df.loc[i,'count']+=1




    sim = pd.read_csv(train_ids_path + "sim_scores.csv")
    sim['id'] = sim['id'].apply(lambda x: x.split('/')[-2] + '/' + x.split('/')[-1] )


    df = df.merge(sim,on='id', how='left')
    training_data = pd.read_csv(train_ids_path + 'train_ids.csv')
    training_data.columns=['ind', 'id']
    training_data['id'] = training_data['id'].apply(lambda x: x.split('/')[-2] + '/' + x.split('/')[-1] )



    training_data = df.loc[df['id'].isin(training_data['id'].values.tolist())].reset_index(drop=True)
    df_notrain = df

    print(df_notrain['count'].value_counts())


    anoms=df_notrain.loc[df_notrain['sim'] < np.percentile(training_data['sim'], 95)].reset_index(drop=True)
    anoms = anoms.loc[(anoms['count'] == len(files) ].reset_index(drop=True)
    anoms = anoms.sort_values(by='av', ascending =False).reset_index(drop=True)



    anoms['id'] = anoms['id'].apply(lambda x: data_path + 'train/'+x)

    anoms['label'].value_counts().to_csv('./outputs/label_details/anoms_label_current_epoch_{}.csv'.format(current_epoch))

    return anoms['id'].tolist()






def write_results(df, results, res_name, logs_df, epoch, model, optimizer, ref_std, oarsi_res, df_rmv, results_rmv, oarsi_results_rmv, args):


      try:
          logs_df.to_csv('./outputs/logs/{}'.format(res_name))
      except:
          pass

      results.to_csv('./outputs/results/' + res_name  + '_epoch_' +str(epoch) )
      results_rmv.to_csv('./outputs/results/' + res_name  + 'train_removed_epoch_' +str(epoch) )

      oarsi_res.to_csv('./outputs/oarsi/' + res_name  + '_epoch_' +str(epoch) )
      oarsi_results_rmv.to_csv('./outputs/oarsi/' + res_name  + 'train_removed_epoch_' +str(epoch) )

      if args.save_models == 1:
        #  if (epoch>=22) & (epoch <29):
        torch.save({
          'model_state_dict': model.state_dict(),
          'optimizer_state_dict': optimizer.state_dict()
          }, './outputs/models/' + res_name  + '_epoch_' +str(epoch))

      elif args.save_models == 2:
          models = os.listdir('./outputs/models/')
          for mod in models:
             if res_name in mod:
              os.remove('./outputs/models/' + mod)

          torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
            }, './outputs/models/' + res_name  + '_epoch_' +str(epoch))



      if args.save_anomaly_scores ==1 :
          df = df.sort_values(by='centre_mean', ascending = False).reset_index(drop=True)
          df.to_csv('./outputs/dfs/' + res_name  + '_epoch_' +str(epoch))
          df_rmv = df_rmv.sort_values(by='centre_mean', ascending = False).reset_index(drop=True)
          df_rmv.to_csv('./outputs/dfs/' + res_name  + 'train_removed_epoch_' +str(epoch) )




def create_patches(features, padding,patchsize, stride):
    n=False
    if type(features).__module__ == np.__name__:
        features = torch.FloatTensor(features)
        n=True

    unfolder = torch.nn.Unfold(
                kernel_size=patchsize, stride=stride, padding=padding, dilation=1
            )
    unfolded_features = unfolder(features)
    number_of_total_patches = []
    for s in features.shape[-2:]:
            n_patches = (
                s + 2 * padding - 1 * (patchsize - 1) - 1
            ) / stride + 1

            number_of_total_patches.append(int(n_patches))
            unfolded_features = unfolded_features.reshape(
            *features.shape[:2], patchsize, patchsize, -1
        )
    unfolded_features = unfolded_features.permute(0, 4, 1, 2, 3)
    if n == True:
        unfolded_features = np.asarray(unfolded_features)


    return unfolded_features[0]



#------------------------------------------------------ evaluate helper functions ------------------------------------------------------


class get_scores():
    def __init__(self, c, mat, device):
        super(get_scores, self).__init__()

        self.c = c
        self.mat = mat
        self.device = device
        self.scores = {}


    def get_dist_metrics(self,out):
        '''
            for each data instance in the testing set, obtain the score and add it to self.scores
        '''
        dist_matrix = (1 - F.cosine_similarity(out, self.mat.to(self.device), dim=-1)).detach().cpu().numpy() #of dimensions N x number patches in stage 'ss'
        dist_matrix_centre = (1 - F.cosine_similarity(out, self.c.to(self.device), dim=-1)).detach().cpu().numpy() #of dimension number_of_patches in stage 'ss'
        scores = self.scores
        if len(scores.keys()) == 0:

            scores['norm_min']=[np.min(dist_matrix)]
            scores['max_scores']=[np.max(dist_matrix)]
            scores['mean_scores']=[np.mean(dist_matrix)]

            scores['mean_scores_min']=[np.mean (np.min(dist_matrix, axis=1))]
            scores['max_scores_min']=[np.max (np.min(dist_matrix, axis=1))]

            scores['norm_min_max']=[np.min (np.max(dist_matrix, axis=1))]
            scores['mean_scores_max']=[np.mean (np.max(dist_matrix, axis=1))]

            scores['norm_min_mean']=[np.min (np.mean(dist_matrix, axis=1))]
            scores['max_scores_mean']=[np.max (np.mean(dist_matrix, axis=1))]

            scores['centre_min']=[np.min(dist_matrix_centre)]
            scores['centre_max']=[np.max(dist_matrix_centre)]
            scores['centre_mean']=[np.mean(dist_matrix_centre)]

        else:
            scores['norm_min'].append(np.min(dist_matrix))
            scores['max_scores'].append(np.max(dist_matrix))
            scores['mean_scores'].append(np.mean(dist_matrix))

            scores['mean_scores_min'].append (np.mean (np.min(dist_matrix, axis=1)))
            scores['max_scores_min'].append (np.max (np.min(dist_matrix, axis=1)))

            scores['norm_min_max'].append (np.min (np.max(dist_matrix, axis=1)))
            scores['mean_scores_max'].append (np.mean (np.max(dist_matrix, axis=1)))

            scores['norm_min_mean'].append (np.min (np.mean(dist_matrix, axis=1)))
            scores['max_scores_mean'].append (np.max (np.mean(dist_matrix, axis=1)))

            scores['centre_min'].append(np.min(dist_matrix_centre))
            scores['centre_max'].append(np.max(dist_matrix_centre))
            scores['centre_mean'].append(np.mean(dist_matrix_centre))


        self.scores = scores


        return dist_matrix


def combine_metrics(scores_nominal, scores_anom):
    scores={}
    scores['w_min'] = ( scores_nominal['norm_min']) - ( scores_anom['norm_min'])
    scores['w_max'] = ( scores_nominal['max_scores']) - ( scores_anom['max_scores'])
    scores['w_mean'] = (scores_nominal['mean_scores']) - ( scores_anom['mean_scores'])
    scores['w_centre'] =( scores_nominal['centre_mean']) - ( scores_anom['centre_mean'])


    return pd.DataFrame(scores, columns= scores.keys())


def create_centre(mat):

    return torch.mean(mat.reshape(mat.shape[0] * mat.shape[1], mat.shape[2]), dim = 0)


def get_df(scores_nominal, scores_anom,names, labels_sev):
        scores_nominal = pd.DataFrame(scores_nominal, columns = scores_nominal.keys())

        if scores_anom != {}:
            scores = combine_metrics(scores_nominal, scores_anom)
            scores_anom = pd.DataFrame(scores_anom, columns = scores_anom.keys())
            scores_anom.columns=[col + '_anom' for col in scores_anom.columns.values.tolist() ]

            df= pd.concat([pd.DataFrame(names, columns = ['id']),pd.DataFrame(labels_sev, columns = ['label']), scores_nominal,  scores], axis =1) #scores_anom
        else:
            df= pd.concat([pd.DataFrame(names, columns = ['id']),pd.DataFrame(labels_sev, columns = ['label']), scores_nominal], axis =1) #scores_anom

        return df



def get_res(df):
    results_values = df.columns.values[2:]

    sp, auc,auc_mid,auc_mid2,auc_sev,acc0,acc1,acc2,acc3,acc4,acc, acc_total, ap, ap_mid, ap_mid2, ap_sev  = get_metrics(df, results_values[0])
    results = pd.DataFrame([[sp], [auc], [auc_mid], [auc_sev],[ap], [ap_mid], [ap_mid2], [ap_sev], [acc0],[acc1],[acc2],[acc3],[acc4],[acc], [acc_total]])
    results=results.T
    for i in range(1, len(results_values)):
        sp, auc,auc_mid,auc_mid2,auc_sev,acc0,acc1,acc2,acc3,acc4,acc, acc_total, ap, ap_mid, ap_mid2, ap_sev  = get_metrics(df, results_values[i] )
        temp =pd.DataFrame([[sp], [auc], [auc_mid], [auc_mid2], [auc_sev], [ap], [ap_mid], [ap_mid2], [ap_sev], [acc0],[acc1],[acc2],[acc3],[acc4], [acc], [acc_total]]).T
        results = pd.concat([results, temp ], axis =0 )


    results.columns= ['spearman', 'auc','auc_mid', 'auc_mid2', 'auc_sev','ap', 'ap_mid', 'ap_mid2', 'ap_sev', 'acc0','acc1','acc2','acc3','acc4','bal_acc', 'acc_not_bal'] #, 'wck']
    results.index = results_values

    return results



def get_results(scores_nominal, scores_anom, training_data,names, labels_sev):
        scores_nominal_rmv={}
        scores_anom_rmv={}
        labels_sev_rmv = []
        names_rmv = []

        for i,key in enumerate(scores_nominal.keys()):
            if scores_anom != {}:
                scores_anom[key] = np.array(scores_anom[key])
            scores_nominal[key] = np.array(scores_nominal[key])

        df= get_df(scores_nominal, scores_anom,names, labels_sev)
        results = get_res(df)
        training_data = pd.DataFrame(training_data, columns=['id'])
        outer = df.merge(training_data, on='id', how='outer', indicator=True)
        df_rmv = outer[(outer._merge=='left_only')].drop('_merge', axis=1)
        results_rmv = get_res(df_rmv)

        return df, results, df_rmv, results_rmv



def create_mat(ref_dataset, model, padding, patchsize, stride, dev, shots):


    for i in range(0, (len(ref_dataset.paths2)  )):
        img1, _, _, _,_,lab = ref_dataset.__getitem__(i)
        if i ==0:
            assert lab.item() == 0
            mat =model.forward( img1.to(dev).float()).detach().unsqueeze(0)
        else:
          if lab.item() == 0:
              mat = torch.cat((mat,  model.forward( img1.to(dev).float()).detach().unsqueeze(0) ))
          else:
              try:
                mat_anom = torch.cat((mat_anom,  model.forward( img1.to(dev).float()).detach() .unsqueeze(0) ))
              except:
                mat_anom =  model.forward( img1.to(dev).float()).detach().unsqueeze(0)


    if shots > 0:
        return mat, mat_anom
    else:
        return mat



def create_mat_patches(ref_dataset, model, padding, patchsize, stride, dev, shots):


    for i in range(0, (len(ref_dataset.paths2)  )):
        img1, _, _, _,_,lab = ref_dataset.__getitem__(i)
        if i == 0:
            assert lab.item() == 0
            mat = F.adaptive_avg_pool2d(create_patches (model.forward( img1.to(dev).float()).detach(), padding, patchsize, stride) , (1,1) )[:,:,0,0].unsqueeze(0)
        else:
            if lab.item() == 0:
                mat = torch.cat((mat, F.adaptive_avg_pool2d( create_patches ( model.forward( img1.to(dev).float()).detach() , padding,patchsize, stride) , (1,1) )[:,:,0,0].unsqueeze(0) ))
            else:
                try:
                  mat_anom = torch.cat((mat_anom, F.adaptive_avg_pool2d( create_patches ( model.forward( img1.to(dev).float()).detach() , padding,patchsize, stride) , (1,1) )[:,:,0,0].unsqueeze(0) ))
                except:
                  mat_anom =  F.adaptive_avg_pool2d( create_patches ( model.forward( img1.to(dev).float()).detach() , padding,patchsize, stride) , (1,1) )[:,:,0,0].unsqueeze(0)

    if shots > 0:
        return mat, mat_anom
    else:
        return mat





def convert(row):
    row['ID']=row['ID'].astype(int).astype('str')
    if row['SIDE'] == 1:
        row['ID'] = str(int(row['ID'])) + 'R'
    else:
        row['ID'] = str(int(row['ID'])) + 'L'

    return row



def oarsi(df, shots, meta_data_dir):
    '''
        get AUC for OARSI grading
    '''
    oarsi_res = {}
    meta = pd.read_csv(meta_data_dir, delimiter='|')
    meta=meta.drop(meta.loc[np.isnan(meta['V00XRKL'])].index.values).reset_index(drop=True)
    meta=meta.loc[meta['READPRJ'] == 15].reset_index(drop=True)
    meta=meta.drop_duplicates().reset_index(drop=True)
    meta = meta.apply(convert, axis=1)

    df['ID'] = df.apply(lambda x: x['id'].split('/')[-1].split('.')[0], axis=1)
    if shots > 0 :
        cols=['centre_mean', 'norm_min', 'w_centre']
    else:
        cols=['centre_mean', 'norm_min']
    merged = df[['ID','label'] + cols].merge(meta[['ID', 'V00XRJSL']], on =['ID'], how='left')
    merged = merged.merge(meta[['ID', 'V00XRJSM']], on =['ID'], how='left')
    merged = merged.merge(meta[['ID', 'V00XROSFL', 'V00XROSFM', 'V00XROSTL', 'V00XROSTM']], on =['ID'], how='left')

    merged['oarsi'] = 0
    merged['V00XRJSM'] = merged['V00XRJSM'].replace(np.nan, 0)
    merged['V00XRJSL'] = merged['V00XRJSL'].replace(np.nan, 0)
    merged['V00XROSFL'] = merged['V00XROSFL'].replace(np.nan, 0)
    merged['V00XROSFM'] = merged['V00XROSFM'].replace(np.nan, 0)
    merged['V00XROSTL'] = merged['V00XROSTL'].replace(np.nan, 0)
    merged['V00XROSTM'] = merged['V00XROSTM'].replace(np.nan, 0)
    merged['osteo_sum'] = merged['V00XROSFL'] + merged['V00XROSFM'] + merged['V00XROSTL'] + merged['V00XROSTM']


    merged.loc[(merged['V00XRJSL'] >= 2) | (merged['V00XRJSM'] >= 2) | (merged['osteo_sum'] >= 2)
              | ((merged['osteo_sum'] > 0) &   ((merged['V00XRJSM'] >0) | (merged['V00XRJSL'] > 0)) ), 'oarsi'] =1

    for col in cols:

        merged['binary_label'] = 0
        merged.loc[merged['oarsi'] > 0, 'binary_label'] = 1
        fpr, tpr, thresholds = roc_curve(np.array(merged['binary_label']),np.array(merged[col]))
        oarsi_res['oarsi_auc_' + col]  = metrics.auc(fpr, tpr)

    return pd.DataFrame(oarsi_res, index=['auc']).T





def get_metrics(df, score):

    res = stats.spearmanr(df[score].tolist(), df['label'].tolist())

#    wck = cohen_kappa_score(df[score], df['label'], labels = np.array([0,1,2,3,4]), weights='linear')

    thres4 = len(df.loc[df['label'] == 4]) / len(df)

    thres4_val = np.percentile(df[score], float((1 - thres4)*100))
    acc4 = len(df.loc[(df['label'] == 4) & (df[score] > thres4_val)]) / len(df.loc[df['label'] == 4])

    thres3 = len(df.loc[df['label'] == 3]) / len(df)
    thres3_val = np.percentile(df[score], float((1 - (thres4 + thres3))*100))
    acc3 = len(df.loc[(df['label'] == 3) & (df[score] > thres3_val) & (df[score] <= thres4_val)]) / len(df.loc[df['label'] == 3])


    thres2 = len(df.loc[df['label'] == 2]) / len(df)
    thres2_val = np.percentile(df[score], float((1 - (thres2 + thres3 + thres4))*100))
    acc2 = len(df.loc[(df['label'] == 2) & (df[score] > thres2_val) & (df[score] <= thres3_val)]) / len(df.loc[df['label'] == 2])


    thres1 = len(df.loc[df['label'] == 1]) / len(df)
    thres1_val = np.percentile(df[score], float( (1 - (thres1 + thres2 + thres3 + thres4))*100))
    acc1 = len(df.loc[(df['label'] == 1) & (df[score] > thres1_val) & (df[score] <= thres2_val)]) / len(df.loc[df['label'] == 1])

    thres0 = len(df.loc[df['label'] == 0]) / len(df)
    thres0_val = np.percentile(df[score],  float(thres0*100))
    acc0 = len(df.loc[(df['label'] == 0) & (df[score] <= thres0_val)]) / len(df.loc[df['label'] == 0])
    acc = (acc0 + acc1 + acc2 + acc3 + acc4) / 5

    acc_total = (len(df.loc[(df['label'] == 4) & (df[score] > thres4_val)]) + len(df.loc[(df['label'] == 3) & (df[score] > thres3_val) & (df[score] <= thres4_val)])  +
      len(df.loc[(df['label'] == 2) & (df[score] > thres2_val) & (df[score] <= thres3_val)]) + len(df.loc[(df['label'] == 1) & (df[score] > thres1_val) & (df[score] <= thres2_val)])  +
         len(df.loc[(df['label'] == 0) & (df[score] <= thres0_val)])) / len(df)



    df['binary_label'] = 0
    df.loc[df['label'] > 0, 'binary_label'] = 1
    fpr, tpr, thresholds = roc_curve(np.array(df['binary_label']),np.array(df[score]))
    auc = metrics.auc(fpr, tpr)
    ap = average_precision_score(np.array(df['binary_label']), np.array(df[score]))


    df['binary_label'] = 0
    df.loc[df['label'] > 1, 'binary_label'] = 1

    fpr, tpr, thresholds = roc_curve(np.array(df['binary_label']),np.array(df[score]))
    auc_mid = metrics.auc(fpr, tpr)
    ap_mid = average_precision_score(np.array(df['binary_label']), np.array(df[score]))


    df['binary_label'] = 0
    df.loc[df['label'] > 2, 'binary_label'] = 1
    fpr, tpr, thresholds = roc_curve(np.array(df['binary_label']),np.array(df[score]))
    auc_mid2 = metrics.auc(fpr, tpr)
    ap_mid2 = average_precision_score(np.array(df['binary_label']), np.array(df[score]))


    df['binary_label'] = 0
    df.loc[df['label'] == 4, 'binary_label'] = 1
    fpr, tpr, thresholds = roc_curve(np.array(df['binary_label']),np.array(df[score]))
    auc_sev = metrics.auc(fpr, tpr)
    ap_sev = average_precision_score(np.array(df['binary_label']), np.array(df[score]))



    return res[0], auc, auc_mid, auc_mid2, auc_sev,acc0,acc1,acc2,acc3,acc4,acc, acc_total, ap, ap_mid, ap_mid2, ap_sev
