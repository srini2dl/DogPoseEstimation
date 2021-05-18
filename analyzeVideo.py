import cv2
import numpy as np
import pandas as pd
import os
import tensorflow as tf

from pathlib import Path
import numpy as np
import ruamel.yaml
from deeplabcut.pose_estimation_tensorflow import AnalyzeVideo
from deeplabcut.pose_estimation_tensorflow.nnet import predict
from ruamel import yaml

from config import load_config

def read_config(configname):
    """
    Reads structured config file
    """
    config_file = ruamel.yaml.YAML()
    path = Path(configname)
    if os.path.exists(path):
        with open(path, 'r') as f:
            cfg = config_file.load(f)
    return cfg

def read_plainconfig(filename = "pose_cfg.yaml"):
    ''' read unstructured yaml'''
    with open(filename, 'r') as f:
        yaml_cfg = yaml.load(f,Loader=yaml.SafeLoader)
    return yaml_cfg

def GetScorerName(shuffle,trainingsiterations='unknown'):
    scorer = 'shuffle' + str(shuffle) + '_' + str(trainingsiterations)
    scorer_legacy = 'shuffle' + str(shuffle) + '_' + str(trainingsiterations)
    return scorer, scorer_legacy

tf.reset_default_graph()
start_path = os.getcwd()
config = 'config.yaml'
cfg = read_config(config)
path_test_config = 'models/train/pose_cfg.yaml'
dlc_cfg = load_config(str(path_test_config))
trainFraction = cfg["TrainingFraction"][0]
# gettting the latest snapshot
Snapshots = np.array(
            [
                fn.split(".")[0]
                for fn in os.listdir(os.path.join('models', "train"))
                if "index" in fn
            ]
        )

snapshotindex = cfg["snapshotindex"]
increasing_indices = np.argsort([int(m.split("-")[1]) for m in Snapshots])
Snapshots = Snapshots[increasing_indices]
print("Using %s" % Snapshots[snapshotindex], "for model", 'models')

dlc_cfg["init_weights"] = os.path.join(
        'models', "train", Snapshots[snapshotindex]
    )
trainingsiterations = (dlc_cfg["init_weights"].split(os.sep)[-1]).split("-")[-1]
dlc_cfg["num_outputs"] = cfg.get("num_outputs", dlc_cfg.get("num_outputs", 1))
dlc_cfg["batch_size"] = cfg["batch_size"]
DLCscorer, DLCscorerlegacy = GetScorerName(shuffle=1, trainingsiterations=trainingsiterations)
xyz_labs = ["x", "y", "likelihood"]
sess, inputs, outputs = predict.setup_GPUpose_prediction(dlc_cfg)
pdindex = pd.MultiIndex.from_product(
        [[DLCscorer], dlc_cfg["all_joints_names"], xyz_labs],
        names=["scorer", "bodyparts", "coords"],
    )
DLCscorer = AnalyzeVideo(
                    'videos/test.mp4',
                    DLCscorer,
                    DLCscorerlegacy,
                    trainFraction,
                    cfg,
                    dlc_cfg,
                    sess,
                    inputs,
                    outputs,
                    pdindex,
                    True,
                    'videos',
                )