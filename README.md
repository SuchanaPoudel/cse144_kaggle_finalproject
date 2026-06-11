CSE 144 Final project

Name : Suchana Poudel

Requirements: 

1. Python 3.9
2. PyTorch 2.8.0
3. torchvision
4. timm
5. pandas
6. pillow
7. numpy
8. Appple M4 MPS(used) or CUDA GPU recommended

Kaggle score: 87%
<img width="1058" height="186" alt="Leaderboard19" src="https://github.com/user-attachments/assets/f11d4de5-0c82-43c4-b8a6-a5085f13a3f4" />

Model used : Swin Transformer Base pretrained on ImageNet-22k, fine-tuned on 100-class dataset.
https://drive.google.com/file/d/14F4VKc-excqh0xyQSlIO4O6H-yRzSZOm/view?usp=share_link

How to run training:
Download: Train and test data set from kaggle
Then, run train.py

How to run inference:
1. Download pretrained.pth from the drive
2. Place it the same folder as before
3. The inference runs with train.py
4. Output: submission_swin.csv

Files in this repsository: 

1. train.py
2. README.md
3. Finalreport.pdf
4. Leaderboard19.png

