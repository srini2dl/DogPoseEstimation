"""
Adapted from DeeperCut by Eldar Insafutdinov
https://github.com/eldar/pose-tensorflow
"""

import logging
import pprint

import yaml


def _merge_a_into_b(a, b):
    """
    Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    for k, v in a.items():
        # a must specify keys that are in b
        # if k not in b:
        #    raise KeyError('{} is not a valid config key'.format(k))

        # recursively merge dicts
        if isinstance(v, dict):
            if not b.get(k, False):
                b[k] = v
            else:
                try:
                    _merge_a_into_b(a[k], b[k])
                except:
                    print("Error under config key: {}".format(k))
                    raise
        else:
            b[k] = v


def cfg_from_file(filename):
    """
    Load a config from file filename and merge it into the default options.
    """
    with open(filename, "r") as f:
        yaml_cfg = yaml.load(f, Loader=yaml.SafeLoader)

    # Update the snapshot path to the corresponding path!
    trainpath = str(filename).split("pose_cfg.yaml")[0]
    yaml_cfg["snapshot_prefix"] = trainpath + "snapshot"
    # the default is: "./snapshot"

    # reloading defaults, as they can bleed over from a previous run otherwise
    cfg = dict()

    cfg['stride'] = 8.0
    cfg['weigh_part_predictions'] = False
    cfg['weigh_negatives'] = False
    cfg['fg_fraction'] = 0.25

    # imagenet mean for resnet pretraining:
    cfg['mean_pixel'] = [123.68, 116.779, 103.939]
    cfg['shuffle'] = True
    cfg['snapshot_prefix'] = "./snapshot"
    cfg['log_dir'] = "log"
    cfg['global_scale'] = 1.0
    cfg['location_refinement'] = False
    cfg['locref_stdev'] = 7.2801
    cfg['locref_loss_weight'] = 1.0
    cfg['locref_huber_loss'] = True
    cfg['optimizer'] = "sgd"
    cfg['intermediate_supervision'] = False
    cfg['intermediate_supervision_layer'] = 12
    cfg['regularize'] = False
    cfg['weight_decay'] = 0.0001
    cfg['crop_pad'] = 0
    cfg['scoremap_dir'] = "test"

    cfg['batch_size'] = 1

    # types of datasets, see factory: deeplabcut/pose_estimation_tensorflow/dataset/factory.py
    cfg['dataset_type'] = "imgaug"  # >> imagaug default as of 2.2
    # you can also set this to deterministic, see https://github.com/AlexEMG/DeepLabCut/pull/324
    cfg['deterministic'] = False
    cfg['mirror'] = False

    # for DLC 2.2. (here all set False to not use PAFs/pairwise fields)
    cfg['pairwise_huber_loss'] = True
    cfg['weigh_only_present_joints'] = False
    cfg['partaffinityfield_predict'] = False
    cfg['pairwise_predict'] = False

    default_cfg = cfg
    _merge_a_into_b(yaml_cfg, default_cfg)

    logging.info("Config:\n" + pprint.pformat(default_cfg))
    return default_cfg  # updated


def load_config(filename="pose_cfg.yaml"):
    return cfg_from_file(filename)


if __name__ == "__main__":
    print(load_config())
