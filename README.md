# Anomaly Detection for Continuous Disease Severity Grading with Self-Supervision and Few Labels: Application to Knee Osteoarthritis

This repository contains a Pytorch implementation of [Anomaly Detection for Continuous Disease Severity Grading with Self-Supervision and Few Labels: Application to Knee Osteoarthritis]().

## Abstract
The diagnostic accuracy and subjectivity of existing Knee Osteoarthritis (OA) ordinal grading systems has been a subject of on-going debate and concern. Existing automated solutions are trained to emulate these imperfect systems, whilst also being reliant on  large annotated databases for fully-supervised training. This work proposes a three stage approach for automated continuous grading of knee OA that is built upon the principles of Anomaly Detection (AD); learning a robust representation of healthy knee X-rays and grading disease severity based on its distance to the centre of normality. In the first stage, SS-FewSOME is proposed, a self-supervised AD technique that learns the 'normal' representation, requiring only examples of healthy subjects and <3% of the labels that existing methods require. In the second stage, this model is used to pseudo label a subset of unlabelled data as 'normal' or 'anomalous', followed by denoising of pseudo labels with CLIP. The final stage involves retraining on labelled and pseudo labelled data using the proposed Dual Centre Representation Learning (DCRL) which learns the centres of two representation spaces; normal and anomalous. Disease severity is then graded based on the distance to the learned centres. The proposed methodology outperforms existing techniques by margins of up to 24% in terms of OA detection and the disease severity scores correlate with the Kellgren-Lawrence grading system at the same level as human expert performance. 



## BibTeX Citation 

@inproceedings{
}

## Dataset
Reference for the dataset;
Chen, Pingjun (2018), “Knee Osteoarthritis Severity Grading Dataset”, Mendeley Data, V1, doi: 10.17632/56rmx5bjcr.1

Additional meta data is required to calculate OARSI detection performance. To obtain this data, apply for access and download the baseline images from the [OAI](https://nda.nih.gov/oai), copy the file '~/X-Ray Image Assessments_ASCII-2/kxr_sq_bu00.txt' to the 'meta' directory. Otherwise, remove the OARSI detection from the code. 


## Installation 
This code is written in Python 3.11 and requires the packages listed in requirements.txt.

Use the following command to clone the repository to your local machine:


```
git clone https://github.com/niamhbelton/SS-FewSOME_Disease_Severity_Knee_Osteoarthritis.git
```

To run the code, set up a virtual environment:

```
pip install virtualenv
cd <path-to-repository-directory>
virtualenv myenv
source myenv/bin/activate
pip install -r requirements.txt
```


## Running Experiments
The code below runs all stages and evaluates the model on the unlabelled and test dataset every 10 epochs.

```
python3 main.py --data_path <path_to_data>
```

The code below runs all stages evaluating only at the end.
```
python3 main.py --data_path <path_to_data> --eval_epoch 0
```

## Output

The results are stored in the 'outputs' directory of the current working directory. There is a subfolder for each of the following;
- logs: stores the AUC for each epoch, the train AUC, train loss etc.
- results: creates a new file at each epoch that stores the AUC for various different methods of calculating the anomaly scores (in this work, we use 'centre_mean' for stages 'ss' and 'w_centre' for all other stages.
- dfs: creates a new file at each epoch that stores the anomaly scores for each data instance in the test set and the unlabelled training set.
- oarsi: creates a new file at each epoch that stores the AUC for OARSI detection.
- models: stores the model after each epoch. Change argument 'save_models' to equal one to save the models.
- label_details: stores the pseudo label distribution and names for each stage after 'ss'.

Within each of the above subfolders, there is a subfolder for each stage; 'ss', 'stage2', 'stage3' and 'stage_severe_pred'.


