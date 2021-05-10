

import os
import argparse

from train import train


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--displayiters", default=10, type=int,
                    help="Logs will be displayed after every displayiters steps")
    parser.add_argument("--saveiters", default=500, type=int, help="model will be saved after every saveiters steps")
    parser.add_argument("--maxiters", default=2000, type=int, help="Max iterations to run")
    parser.add_argument("--max_snapshots_to_keep", default=4, type=int, help="How many models to save")
    parser.add_argument("--keepdeconvweights", default=True, type=bool)
    args = parser.parse_args()
    return args

args = arg_parser()
from pathlib import Path
projectDirectory = Path().resolve()
labeledDataPath = projectDirectory / 'labeledData'
trainingDataPath = projectDirectory / 'trainingData'
modelPath = projectDirectory / 'models'
modelPathTrain = modelPath / 'train'
modelPathTest = modelPath / 'test'

modelfoldername = modelPath
poseconfigfile = modelPathTrain / "pose_cfg.yaml"

train(str(poseconfigfile),args.displayiters,args.saveiters,args.maxiters,max_to_keep=args.max_snapshots_to_keep)

