import torch.nn.functional as F
from utils import *
import torch
import pandas as pd
import warnings


warnings.filterwarnings("ignore")


def evaluate_severity( patches, padding,patchsize, stride, seed,ref_dataset, val_dataset, model,  data_path, criterion,  dev, shots, meta_data_dir):

    model.eval()
    loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False) #create loader for dataset for test set

    if patches:
        mat = create_mat_patches(ref_dataset, model , padding,patchsize, stride, dev, shots)
    else:
        mat = create_mat(ref_dataset, model , padding,patchsize, stride, dev, shots)

    if shots > 0:
        mat, mat_anom = mat


    for i in range(mat.shape[0]):
        try:
            mat_dists = torch.cat ((mat_dists, 1 - F.cosine_similarity(mat[i,0,:], mat[:,0,:], dim=-1).unsqueeze(0)))
        except:
            mat_dists = 1 - F.cosine_similarity(mat[i,0,:], mat[:,0,:], dim=-1).unsqueeze(0) #30 x 1000 with #30 x1000

    mat_dists  = mat_dists.flatten()[1:].view(mat_dists.shape[0]-1, mat_dists.shape[0]+1)[:,:-1].reshape(mat_dists.shape[0], mat_dists.shape[0]-1) #mat_dists.to(dev) * (torch.FloatTensor([1]).to(dev) - torch.eye(mat_dists.shape[0]).to(dev))
    c= create_centre(mat) #1 x 256
    if shots > 0:
        c_anom = create_centre(mat_anom)

    ref_max = torch.max(mat_dists).cpu().numpy()
    ref_min = torch.min(mat_dists).cpu().numpy()
    ref_sum =torch.sum(mat_dists).cpu().numpy() / 2
    ref_mean = torch.mean(mat_dists).cpu().numpy()
    ref_var = torch.var(mat_dists).cpu().numpy()
    ref_std = pd.DataFrame(torch.std(mat[:,0,:], dim = 0).cpu().numpy())
    labels=[]
    labels_sev=[]
    names=[]
    scores_nominal = {}

    if shots > 0:
        ref_centre_dist = F.cosine_similarity(c, c_anom, dim=0).cpu().numpy()
        ref_info = [ref_max, ref_min, ref_sum, ref_mean, ref_var,ref_centre_dist]
        scores_anom = {}
        get_anom_scores = get_scores( c_anom, mat_anom, dev)
    else:
        ref_info = [ref_max, ref_min, ref_sum, ref_mean, ref_var]



    get_nominal_scores = get_scores( c, mat, dev)


    with torch.no_grad():

        #loop through images in the dataloader
        for i, data in enumerate(loader):



                image = data[0][0]
                names.append(data[-2])
                labels_sev.append(data[-1].item())


                out = model.forward(image.to(dev).float()).detach() #get feature vector for test image

                if patches:
                    out = F.adaptive_avg_pool2d( create_patches(out, padding,patchsize, stride) , (1,1) )[:,:,0,0].squeeze(1) #16 x 256

                dist_matrix= get_nominal_scores.get_dist_metrics(out)
                if shots > 0:
                    dist_matrix_anom= get_anom_scores.get_dist_metrics(out)



    if shots > 0:
        df, results,df_rmv, results_rmv = get_results(get_nominal_scores.scores, get_anom_scores.scores, ref_dataset.paths2, names, labels_sev)
    else:
        df, results,df_rmv, results_rmv = get_results(get_nominal_scores.scores, {}, ref_dataset.paths2, names, labels_sev)


    oarsi_results = oarsi(df, shots, meta_data_dir)
    oarsi_results_rmv = oarsi(df_rmv, shots, meta_data_dir)



    return df, results, pd.DataFrame(ref_info).T, ref_std, oarsi_results, df_rmv, results_rmv, oarsi_results_rmv
