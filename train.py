import logging
import os
import threading
import tensorflow as tf
from pathlib import Path

from tensorflow_core.contrib import slim

from config import load_config
from pose_dataset import Batch
from pose_netmulti import PoseNet


class LearningRate(object):
    def __init__(self, cfg):
        self.steps = cfg['multi_step']
        self.current_step = 0

    def get_lr(self, iteration):
        lr = self.steps[self.current_step][0]
        if iteration == self.steps[self.current_step][1]:
            self.current_step += 1

        return lr

def setup_preloading(batch_spec):
    placeholders = {
        name: tf.placeholder(tf.float32, shape=spec)
        for (name, spec) in batch_spec.items()
    }
    names = placeholders.keys()
    placeholders_list = list(placeholders.values())

    QUEUE_SIZE = 20
    vers = (tf.__version__).split(".")
    if int(vers[0]) == 1 and int(vers[1]) > 12:
        q = tf.queue.FIFOQueue(QUEUE_SIZE, [tf.float32] * len(batch_spec))
    else:
        q = tf.FIFOQueue(QUEUE_SIZE, [tf.float32] * len(batch_spec))
    enqueue_op = q.enqueue(placeholders_list)
    batch_list = q.dequeue()

    batch = {}
    for idx, name in enumerate(names):
        batch[name] = batch_list[idx]
        batch[name].set_shape(batch_spec[name])
    return batch, enqueue_op, placeholders


def start_preloading(sess, enqueue_op, dataset, placeholders):
    def load_and_enqueue():
        while not coord.should_stop():
            batch_np = dataset.next_batch()
            feedDict = {pl: batch_np[name] for (name, pl) in placeholders.items()}
            sess.run(enqueue_op, feed_dict=feedDict)
    coord = tf.train.Coordinator()

    t = threading.Thread(target=load_and_enqueue)

    t.start()

    return coord, t

def train(
    config_yaml,
    displayiters,
    saveiters,
    maxiters,
    max_to_keep=5,
    allow_growth=False,
):
    start_path = os.getcwd()
    os.chdir(
        str(Path(config_yaml).parents[0])
    )
    cfg = load_config(config_yaml)
    from pose_dataset_imgaug import (
        PoseDataset,
    )

    dataset = PoseDataset(cfg)
    batch_spec = {
        Batch.inputs: [1, None, None, 3],
        Batch.part_score_targets: [1, None, None, 20],
        Batch.part_score_weights: [1, None, None, 20],
        Batch.locref_targets: [1, None, None, 20 * 2],
        Batch.locref_mask: [1, None, None, 20 * 2],
    }
    batch, enqueue_op, placeholders = setup_preloading(batch_spec)

    losses = PoseNet(cfg).train(batch)
    total_loss = losses["total_loss"]

    for k, t in losses.items():
        tf.summary.scalar(k, t)
    merged_summaries = tf.summary.merge_all()

    variables_to_restore = slim.get_variables_to_restore(include=["resnet_v1"])
    print("loading")

    restorer = tf.train.Saver(variables_to_restore)
    saver = tf.train.Saver(
        max_to_keep=max_to_keep
    )

    sess = tf.Session()

    coord, thread = start_preloading(sess, enqueue_op, dataset, placeholders)
    train_writer = tf.summary.FileWriter(cfg['log_dir'], sess.graph)

    # setting optimizer
    learning_rate = tf.placeholder(tf.float32, shape=[])
    optimizer = tf.train.MomentumOptimizer(
        learning_rate=learning_rate, momentum=0.9
    )
    train_op = slim.learning.create_train_op(total_loss, optimizer)

    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    # Restore variables from disk.
    restorer.restore(sess, cfg['init_weights'])
    max_iter = 2000
    print("Maximum iterations:", max_iter)
    display_iters = max(1, int(displayiters))
    print("Display iteration:", display_iters)

    if saveiters == None:
        save_iters = max(1, int(cfg['save_iters']))

    else:
        save_iters = max(1, int(saveiters))
        print("Save_iters overwritten as", save_iters)

    cum_loss = 0.0
    lr_gen = LearningRate(cfg)

    stats_path = Path(config_yaml).with_name("learning_stats.csv")
    lrf = open(str(stats_path), "w")

    print("Starting training....")
    for it in range(max_iter + 1):
        current_lr = lr_gen.get_lr(it)
        dict={learning_rate: current_lr}

        [_, loss_val, summary] = sess.run(
            [train_op, total_loss, merged_summaries],
            feed_dict=dict,
        )
        cum_loss += loss_val
        train_writer.add_summary(summary, it)

        average_loss = cum_loss / display_iters
        cum_loss = 0.0
        print("iteration: {} loss: {} lr: {}".format(
            it, "{0:.4f}".format(average_loss), current_lr))

        # Save snapshot
        if (it % save_iters == 0 and it != 0) or it == max_iter:
            model_name = cfg['snapshot_prefix']
            saver.save(sess, model_name, global_step=it)

    lrf.close()
    sess.close()
    coord.request_stop()
    coord.join([thread])
    # return to original path.
    os.chdir(str(start_path))