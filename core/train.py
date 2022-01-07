from utils import scheduler, gifCallback
import tensorflow as tf
import random
import json

from data_generators import get_train_dataset, get_val_dataset
from hourglass import StackedHourglassNetwork
from generate_params import load_filenames
from custom_loss import adaptive_wing_loss

from params.project_config import data_folder

# the callbacks used during training
# lr scheduler, model checkpoint to save the weights after
# each record low validation loss, and the custom gifCallback
def run_training():

    callbacks = [tf.keras.callbacks.LearningRateScheduler(scheduler),
                tf.keras.callbacks.ModelCheckpoint(
                    data_folder + 'models/weights.40k_{epoch:02d}.hdf5',
                    monitor="loss",
                    save_weights_only=True,
                    verbose=1,
                    save_best_only=True,
                    mode="auto",
                    save_freq="epoch",
                #), gifCallback()]
                )]

    # using RMSprop optimizer
    optimizer = tf.keras.optimizers.RMSprop(2.5e-4)

    # create the stacked hourglass model with the given number
    # of stacked hourglass modules
    # code source: https://github.com/ethanyanjiali/deep-vision/tree/master/Hourglass/tensorflow
    model = StackedHourglassNetwork(num_stack=2)
    model.compile(optimizer=optimizer, loss=adaptive_wing_loss)

    # from each of the generators create a pair of interleaved datasets
    # tensorflow can automatically multiprocess interleaved datasets
    # so that while batches can be loaded and processed ahead of time
    interleaved_train = tf.data.Dataset.range(2).interleave(
                        get_train_dataset,
                        num_parallel_calls=tf.data.experimental.AUTOTUNE
                    )
    interleaved_val = tf.data.Dataset.range(2).interleave(
                        get_val_dataset,
                        num_parallel_calls=tf.data.experimental.AUTOTUNE
                    )

    # load training and validation filenames
    train_ids, val_ids = load_filenames()
    # random.shuffle(train_ids)
    # random.shuffle(val_ids)

    # store generator parameters
    batch_size = 2
    image_shape = (256,256,3)
    #
    # train_params = {
    #     'ids': train_ids,
    #     'batch_size': batch_size,
    #     'image_shape': image_shape
    # }
    # val_params = {
    #     'ids': val_ids,
    #     'batch_size': batch_size,
    #     'image_shape': image_shape
    # }
    # json.dump(train_params, open(data_folder + 'train_params.json', 'w'))
    # json.dump(val_params, open(data_folder + 'val_params.json', 'w'))

    # train the model
    model.fit(interleaved_train,
                       steps_per_epoch = int(len(train_ids) // batch_size),
                       epochs = 2,
                        verbose=1,
                        validation_data=interleaved_val,
                        validation_steps=int(len(val_ids) // batch_size),
                       callbacks=callbacks)

    #model.save_weights('./data/models/10k_weights_55_epochs_noflip')
if __name__ == "__main__":
    run_training()
