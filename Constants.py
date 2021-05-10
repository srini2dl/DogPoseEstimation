from pathlib import Path

projectDirectory = Path().resolve()
labeledDataPath = projectDirectory / 'labeledData'
trainingDataPath = projectDirectory / 'trainingData'
modelPath = projectDirectory / 'models'
modelPathTrain = modelPath / 'train'
modelPathTest = modelPath / 'test'

shuffledMat = 'shuffledMat.mat'
shuffledPickle = 'shuffledPickle.pickle'

scorer ='parts'
trainingFraction = [0.8]  # ?
pose_cfg = projectDirectory / 'pose_cfg.yaml'
bodyparts = ['L_F_Paw',
              'L_F_Knee',
              'L_F_Elbow',
              'L_B_Paw',
              'L_B_Knee',
              'L_B_Elbow',
              'R_F_Paw',
              'R_F_Knee',
              'R_F_Elbow',
              'R_B_Paw',
              'R_B_Knee',
              'R_B_Elbow',
              'TailBase',
              'L_EarBase',
              'R_EarBase',
              'Nose',
              'L_Eye',
              'R_Eye',
              'Withers',
              'Throat']
