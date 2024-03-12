# Anomaly Detection for Continuous Disease Severity Grading with Self-Supervision and Few Labels: Application to Knee Osteoarthritis

This repository contains a Pytorch implementation of [Anomaly Detection for Continuous Disease Severity Grading with Self-Supervision and Few Labels: Application to Knee Osteoarthritis]().

## Abstract
The diagnostic accuracy and subjectivity of existing Knee Osteoarthritis (OA) ordinal grading systems has been a subject of on-going debate and concern. Existing automated solutions are trained to emulate these imperfect systems, whilst also being reliant on  large annotated databases for fully-supervised training. This work proposes a three stage approach for automated continuous grading of knee OA that is built upon the principles of Anomaly Detection (AD); learning a robust representation of healthy knee X-rays and grading disease severity based on its distance to the centre of normality. In the first stage, SS-FewSOME is proposed, a self-supervised AD technique that learns the 'normal' representation, requiring only examples of healthy subjects and <3% of the labels that existing methods require. In the second stage, this model is used to pseudo label a subset of unlabelled data as 'normal' or 'anomalous', followed by denoising of pseudo labels with CLIP. The final stage involves retraining on labelled and pseudo labelled data using the proposed Dual Centre Representation Learning (DCRL) which learns the centres of two representation spaces; normal and anomalous. Disease severity is then graded based on the distance to the learned centres. The proposed methodology outperforms existing techniques by margins of up to 24% in terms of OA detection and the disease severity scores correlate with the Kellgren-Lawrence grading system at the same level as human expert performance. 



## BibTeX Citation 

@inproceedings{
}



## Installation 
This code is written in Python 3.11 and requires the packages listed in requirements.txt.

Use the following command to clone the repository to your local machine:


```
git clone https://github.com/niamhbelton/FewSOME.git
```

To run the code, set up a virtual environment:

```
pip install virtualenv
cd <path-to-SS-FewSOME-directory>
virtualenv myenv
source myenv/bin/activate
pip install -r requirements.txt
```


## Running Experiments
