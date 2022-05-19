import argparse
import os
import numpy as np
from tqdm import tqdm
import logging
from src.utils.common import read_yaml, create_directories
import random
import tensorflow as tf


STAGE = "creating base model" # <<< change stage name 

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )


def main(config_path):
    # read config files
    config = read_yaml(config_path)
    # get the data
    mnist = tf.keras.datasets.mnist
    (X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()
    X_valid, X_train = X_train_full[:5000] / 255., X_train_full[5000:] / 255.
    y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

    # scale the test set as well
    X_test = X_test / 255.
    return (X_train,y_train),(X_valid,y_valid),(X_test,y_test)
    seed = 2021
    tf.random.set_seed(seed)
    np.random.seed(seed)

   # define layers
    LAYERS = [
        tf.keras.layers.Flatten(input_shape = [28,28],name = "inputlayer1"),
        tf.keras.layers.Dense(300,name = "hiddenlayer2"),
        tf.keras.layers.LekyRelu(),
        tf.keras.layers.Dense(100,name = "hiddenlayer3"),
        tf.keras.layers.LekyRelu(),
        tf.keras.layers.Dense(10,name = "outputlayer")  
        
    ]

  #define the model and compile it
    LOSS = "categorical_cross_entropy"
    OPTIMIZER = tf.keras.optimizers.SGD(learning_rate=1e-3)
    METRICS = ["accuracy"]

    model = tf.keras.models.Sequential(LAYERS)
    model.compile(loss = LOSS,optimizer = OPTIMIZER,metrics = METRICS)
    model.summary()

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info("\n********************")
        logging.info(f">>>>> stage {STAGE} started <<<<<")
        main(config_path=parsed_args.config)
        logging.info(f">>>>> stage {STAGE} completed!<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e