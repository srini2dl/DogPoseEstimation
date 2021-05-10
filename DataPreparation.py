from Utils import *
import scipy.io as sio
import pandas as pd

# Creating the folder structure to store labeled, training and model
Constants.trainingDataPath.mkdir(parents=True, exist_ok=True)
Constants.modelPath.mkdir(parents=True, exist_ok=True)

trainingsetfolder = Constants.trainingDataPath
Data = pd.read_hdf((str(Constants.labeledDataPath) + '/CollectedData_parts.h5'), 'df_with_missing')
Data = Data[Constants.scorer]
net_type = 'resnet_50'
augmenter_type = 'default'
defaultconfigfile = Constants.pose_cfg
num_shuffles = 1
resNetPath = str(Constants.projectDirectory) + '/resnet_v1_50.ckpt'

# Preparing the dataset
Shuffles = range(1, num_shuffles + 1)
splits = [(trainFraction, shuffle, SplitTrials(range(len(Data.index)), trainFraction))
                  for trainFraction in Constants.trainingFraction for shuffle in Shuffles]

def readyToTrain(splits):
    bodyparts = Constants.bodyparts
    nbodyparts = len(Constants.bodyparts)
    for trainFraction, shuffle, (trainIndexes, testIndexes) in splits:
        data, MatlabData = format_training_data(Data, trainIndexes, nbodyparts)
        # creating mat file for training
        sio.savemat(os.path.join(Constants.trainingDataPath, Constants.shuffledMat), {'dataset': MatlabData})

        # creating pickle file for training
        SaveMetadata(os.path.join(Constants.trainingDataPath, Constants.shuffledPickle), data, trainIndexes,
                     testIndexes, trainFraction)

        # creating model with test and train folders
        Constants.modelPath.mkdir(parents=True, exist_ok=True)
        Constants.modelPathTrain.mkdir(parents=True, exist_ok=True)
        Constants.modelPathTest.mkdir(parents=True, exist_ok=True)

        # creating config file for test & train
        path_train_config = str(Constants.modelPathTrain) + '/pose_cfg.yaml'
        path_test_config = str(Constants.modelPathTest) + '/pose_cfg.yaml'

        items2change = {
            "dataset": os.path.join(Constants.trainingDataPath, Constants.shuffledMat),
            "metadataset": os.path.join(Constants.trainingDataPath, Constants.shuffledPickle),
            "num_joints": len(bodyparts),
            "all_joints": [[i] for i in range(len(bodyparts))],
            "all_joints_names": [str(bpt) for bpt in bodyparts],
            "init_weights": str(Constants.projectDirectory) + '/resnet_v1_50.ckpt',
            "project_path": str(Constants.projectDirectory),
            "net_type": net_type,
            "dataset_type": augmenter_type,
        }
        trainingDataSet = MakeTrain_pose_yaml(items2change, path_train_config, str(Constants.pose_cfg))
        keys2save = [
            "dataset", "num_joints", "all_joints", "all_joints_names",
            "net_type", 'init_weights', 'global_scale', 'location_refinement',
            'locref_stdev'
        ]
        MakeTest_pose_yaml(trainingDataSet, keys2save, path_test_config)
    return splits

readyToTrain(splits)