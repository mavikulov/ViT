import os 
import pickle
import tarfile
import argparse

import numpy as np
from urllib.request import urlretrieve
from sklearn.model_selection import train_test_split


def unpickle(pickle_file):
    with open(pickle_file, 'rb') as file:
        data = pickle.load(file, encoding='latin1')
    return data


def download_cifar10(
        path, 
        url="https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
        tarname='cifar-10-python.tar.gz'
):
    if not os.path.exists(path):
        os.makedirs(path)
    
    if os.path.exists(os.path.join(path, "cifar-10-batches-py")):
        print(f"Data folder cifar-10-batches-py is already downloaded")
        print(f"Stop donwloading...")
        return 

    urlretrieve(url, os.path.join(path, tarname))
    tar_file = tarfile.open(os.path.join(path, tarname))
    tar_file.extractall(path=path)


def load_cifar10(data_path=".", channels_last=False, save_data=False, test_size=0.2, random_state=42):
    path_to_batches = os.path.join(data_path, "cifar-10-batches-py")
    test_path = os.path.join(path_to_batches, "test_batch")
    train_paths = sorted([
        os.path.join(path_to_batches, batch) for batch in os.listdir(path_to_batches)
        if batch.startswith("data_batch")
    ])
    
    if not os.path.exists(test_path) or not all(list(map(os.path.exists, train_paths))):
        print ("Dataset not found. Downloading...")
        download_cifar10(data_path)

    train_batches = list(map(unpickle, train_paths))
    test_batch = unpickle(test_path)

    X = np.concatenate([batch["data"] for batch in train_batches]).reshape(-1, 3, 32, 32).astype('float32') / 255
    Y = np.concatenate([batch["labels"] for batch in train_batches]).astype('int32')

    X_train, X_val, y_train, y_val = train_test_split(
        X, Y,
        test_size=test_size,
        random_state=random_state
    )

    X_test = test_batch["data"].reshape(-1, 3, 32, 32).astype('float32') / 255
    y_test = np.array(test_batch["labels"]).astype('int32')

    if channels_last:
        X_train = X_train.transpose([0, 2, 3, 1])
        X_test = X_test.transpose([0, 2, 3, 1])
        X_val = X_val.transpose([0, 2, 3, 1])

    if save_data:
        save_folder = "data_npz"
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        
        np.savez_compressed(
            file=os.path.join(save_folder, "dataset.npz"),
            X_train=X_train,
            X_val=X_val,
            X_test=X_test,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test
        )

    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root_path",
        type=str,
        default=".",
        help="Path to root folder"
    )

    args = parser.parse_args()
    path = args.root_path
    download_cifar10(path)
