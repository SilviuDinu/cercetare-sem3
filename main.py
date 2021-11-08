import torch
from FaceDetectorUtils import FaceDetector
# from facenet_pytorch import MTCNN, InceptionResnetV1
from mtcnn import MTCNN
import numpy as np
import os
import matplotlib.pyplot as plt


curr = os.path.dirname(__file__)

mtcnn = MTCNN()
fcd = FaceDetector(mtcnn)
fcd.runPictureDetection_MTCNN_NORMAL(os.path.join(curr, r'fer2013/train/'), os.path.join(curr, r'processed/normal_MTCNN/train/'))
fcd.runPictureDetection_MTCNN_PYTORCH(os.path.join(curr, r'fer2013/train/'), os.path.join(curr, r'processed/pytorch_MTCNN/train/'))

os.system('afplay /System/Library/Sounds/Glass.aiff')
