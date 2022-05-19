import argparse
from email.mime import base
import os
from urllib.parse import DefragResult
import tensorflow as tf
from tqdm import tqdm
import logging
from src.utils.common import read_yaml, create_directories
import random
import numpy as np
import io


STAGE = "transfer learning" ## <<< change stage name 

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )

def update_even_odd_labels(list_of_labels):
    for idx, label in enumerate(list_of_labels):
        even_condition = label%2 == 0
        list_of_labels[idx] = np.where(even_condition, 1, 0)
    return list_of_labels




def main(config_path):
    ## read config files
    config = read_yaml(config_path)
    
    (X_train_full, y_train_full), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    X_train_full = X_train_full / 255.0
    X_test = X_test / 255.0
    X_valid, X_train = X_train_full[:5000], X_train_full[5000:]
    y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
    y_train_bin, y_test_bin, y_valid_bin = update_even_odd_labels([y_train, y_test, y_valid])
    seed = 2021
    tf.random.set_seed(seed)
    np.random.seed(seed)

    def _log_model_summary(model):
        with io.StringIO() as stream:
            model.summary(print_fn= lambda x: stream.write(f"{x}\n"))
            summary_str = stream.getvalue()
        return summary_str

    # load the base model
    base_model_path = os.path.join("artifacts","models","base_model.h5")
    base_model = tf.keras.models.load_model(base_model_path)
    logging.info(f"transferlearning model summary: \n{_log_model_summary(base_model)}")



    for layer in base_model.layers[: -1]:
        
        print(f"trainable status of {layer.name}:{layer.trainable}")
        layer.trainable = False
        print(f"trainable status of {layer.name}:{layer.trainable}")

        base_layer =base_model.layers[:-1]





   # define layers
    

    new_model = tf.keras.models.Sequential(base_layer)
    new_model.add(tf.keras.layers.Dense(2,activation = "softmax",name = "outputlayer"))
    logging.info(f"transferlearning model summary: \n{_log_model_summary(new_model)}")



    LOSS = "sparse_categorical_crossentropy"
    OPTIMIZER = tf.keras.optimizers.SGD(learning_rate=1e-3)
    METRICS = ["accuracy"]

    new_model.compile(loss=LOSS, optimizer=OPTIMIZER, metrics=METRICS) 
    
   

    ## Train the model
    history = new_model.fit(
        X_train, y_train_bin, 
        epochs=30, 
        validation_data=(X_valid, y_valid_bin),
        verbose=2)

    ## save the base model - 
    model_dir_path = os.path.join("artifacts","models")
    create_directories([model_dir_path])

    model_file_path = os.path.join(model_dir_path, "even_odd_model.h5")
    new_model.save(model_file_path)
    logging.info(f"base model is saved at {model_file_path}")
    logging.info(f"evaluation metrics {new_model.evaluate(X_test,y_test_bin)}")



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