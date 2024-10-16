import torch
import tiatoolbox
from tiatoolbox.models import PatchPredictor
import os
import random
import shutil
# Load pre-trained ResNet50 from TIAToolbox
predictor = PatchPredictor(pretrained_model='resnet50-kather100k', batch_size=32)
