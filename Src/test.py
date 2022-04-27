import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import cv2
import pandas as pd
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score
from metrics import dice_loss, dice_coef, iou
from train import load_data, create_dir

x = cv2.imread("testImg.jpg", cv2.IMREAD_COLOR)
x = cv2.resize(x, (256, 256))
cv2.imwrite("out.jpg", x)