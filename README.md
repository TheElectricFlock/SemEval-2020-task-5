# SemEval-2020-task-5
A repository for a solution to SemEval 2021: Task 5
 This task was to create an intelligent sytem capable of identifying toxic span in text.
 The results were an intelligent toxioc span flagging model thats fully convolutional (FCN)
 and takes it's inspiration from classic image segmentation models such as [SegNet](https://arxiv.org/pdf/1511.00561.pdf)
 
## Usage
for this project you will need the following:
- python 3.6
- keras == 2.4.3
- tensorflow == 2.4.1
- chars2vec
- jupyter notebook

Download the repo and open the jupyter notebook. from within the repo. Select `Toxic_Text_Segmentation_Model` and run the training.
To run inference, perform the same actions but select `Evalutation` from the notebook inteface

## Video Demonstration
https://youtu.be/PBKGFsLLHwk

## Tests
These packages come with tests. just run `python  test_utils.py`
