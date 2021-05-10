import Constants
import numpy as np
import os.path
import os, pickle, yaml
from skimage import io

def MakeTest_pose_yaml(dictionary, keys2save, saveasfile):
    dict_test = {}
    for key in keys2save:
        dict_test[key] = dictionary[key]

    dict_test['scoremap_dir'] = 'test'
    with open(saveasfile, "w") as f:
        yaml.dump(dict_test, f)

def MakeTrain_pose_yaml(itemstochange,saveasconfigfile,defaultconfigfile):
    raw = open(defaultconfigfile).read()
    docs = []
    for raw_doc in raw.split('\n---'):
        try:
            docs.append(yaml.load(raw_doc,Loader=yaml.SafeLoader))
        except SyntaxError:
            docs.append(raw_doc)

    for key in itemstochange.keys():
        docs[0][key] = itemstochange[key]

    with open(saveasconfigfile, "w") as f:
        yaml.dump(docs[0], f)
    return docs[0]


def SaveMetadata(metadatafilename, data, trainIndexes, testIndexes, trainFraction):
    with open(metadatafilename, 'wb') as f:
        # Pickle the 'labeled-data' dictionary using the highest protocol available.
        pickle.dump([data, trainIndexes, testIndexes, trainFraction], f, pickle.HIGHEST_PROTOCOL)

def _read_image_shape_fast(path):
    return io.imread(path).shape

def format_training_data(df, train_inds, nbodyparts):
    train_data = []
    matlab_data = []

    def to_matlab_cell(array):
        outer = np.array([[None]], dtype=object)
        outer[0, 0] = array.astype('int64')
        return outer

    for i in train_inds:
        data = dict()
        filename = df.index[i]
        data['image'] = filename
        img_shape = _read_image_shape_fast(os.path.join(Constants.projectDirectory, filename))
        try:
            data['size'] = img_shape[2], img_shape[0], img_shape[1]
        except IndexError:
            data['size'] = 1, img_shape[0], img_shape[1]
        temp = df.iloc[i].values.reshape(-1, 2)
        joints = np.c_[range(nbodyparts), temp]
        joints = joints[~np.isnan(joints).any(axis=1)].astype(int)
        # Check that points lie within the image
        inside = np.logical_and(np.logical_and(joints[:, 1] < img_shape[1], joints[:, 1] > 0),
                                np.logical_and(joints[:, 2] < img_shape[0], joints[:, 2] > 0))
        if not all(inside):
            joints = joints[inside]
        if joints.size:  # Exclude images without labels
            data['joints'] = joints
            train_data.append(data)
            matlab_data.append((np.array([data['image']], dtype='U'),
                                np.array([data['size']]),
                                to_matlab_cell(data['joints'])))
    matlab_data = np.asarray(matlab_data, dtype=[('image', 'O'), ('size', 'O'), ('joints', 'O')])
    return train_data, matlab_data

def SplitTrials(trialindex, trainFraction=0.8):
    ''' Split a trial index into train and test sets. Also checks that the trainFraction is a two digit number between 0 an 1. The reason
    is that the folders contain the trainfraction as int(100*trainFraction). '''
    if trainFraction>1 or trainFraction<0:
        print("The training fraction should be a two digit number between 0 and 1; i.e. 0.95. Please change accordingly.")
        return ([],[])

    if abs(trainFraction-round(trainFraction,2))>0:
        print("The training fraction should be a two digit number between 0 and 1; i.e. 0.95. Please change accordingly.")
        return ([],[])
    else:
        trainsetsize = int(len(trialindex) * round(trainFraction,2))
        shuffle = np.random.permutation(trialindex)
        testIndexes = shuffle[trainsetsize:]
        trainIndexes = shuffle[:trainsetsize]

        return (trainIndexes, testIndexes)