import os
import pathlib

from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import mnist, cifar10
from keras import utils


class DataLoader:
    # dataset: str | "mnist" or "cifar" or str path or list path [path_train, path_test]
    def __init__(
        self,
        image_size: int = 28,
        batch_size: int = 8,
        channel_num: int = 3,
        dataset="mnist",
    ):
        """
        Class Loader Dataset

        support dataset:
        - Keras - MNIST
        - Keras - CIFAR10
        - Csutom Dataset

        Parameters:
            image_size: `int` - size of image will be proccess
                default: 28
            batch_size: `int` - batch for step training, it will used if register custom dataset
                default: 8
            dataset: `string` | `list` - dataset image,
                default: "mnist"
                custom dataset: "path_to_root_dataset" or ["path_to_train_dataset", "path_to_valid_dataset"]
                if use custom dataset, please provide your dataset to implement this folder structures:
                    | root_dataset
                        | train
                            | class 1
                                | image 1 \n
                                | image 2 \n
                                | image ... \n
                            | class 2
                                | image 1 \n
                                | image 2 \n
                                | image ... \n
                            | class ...
                        | valid
                            | class 1
                                | image 1 \n
                                | image 2 \n
                                | image ... \n
                            | class 2
                                | image 1 \n
                                | image 2 \n
                                | image ... \n
                            | class ...

        """
        self._use_path = False
        self._image_size = image_size
        self._batch_size = batch_size
        self._channel_num = channel_num

        if dataset == "mnist":
            self.__keras_dataset = "mnist"
            self._dataset = mnist.load_data()
        elif dataset == "cifar10" or dataset == "cifar":
            self.__keras_dataset = "cifar10"
            self._dataset = cifar10.load_data()
        else:
            self._dataset = dataset
            self._use_path = True

        self._data_ready = {
            "train": {
                "generator": None,
                "x": None,
                "y": None,
                "batch": 0,
            },
            "valid": {
                "generator": None,
                "x": None,
                "y": None,
                "batch": 0,
            },
        }

        self._preproccess()

    def __keras_dataset_load(self):
        image_size = self._image_size
        (x_train, y_train), (x_test, y_test) = self._dataset

        if self.__keras_dataset == "mnist":
            self._class_num = len(set(y_train))
        elif self.__keras_dataset == "cifar10":
            self._class_num = 10

        x_train = (
            x_train.reshape(-1, image_size, image_size, 1).astype("float32") / 255.0
        )
        x_test = x_test.reshape(-1, image_size, image_size, 1).astype("float32") / 255.0
        y_train = utils.to_categorical(y_train)
        y_test = utils.to_categorical(y_test)

        self._data_ready["train"]["x"] = x_train
        self._data_ready["train"]["y"] = y_train

        self._data_ready["valid"]["x"] = x_test
        self._data_ready["valid"]["y"] = y_test

    def __custom_dataset(self, train_data_dir, valid_data_dir):
        image_size = self._image_size
        batch_size = self._batch_size
        dirs_train = os.listdir(train_data_dir)
        dirs_valid = os.listdir(valid_data_dir)

        self._class_num = len(dirs_train)

        img_width, img_height = image_size, image_size

        nb_train_sample = 0
        nb_valid_sample = 0

        for clz in dirs_train:
            path_class = train_data_dir + "/" + clz
            n_img = len(os.listdir(path_class))
            nb_train_sample += n_img
        for clz in dirs_valid:
            path_class = valid_data_dir + "/" + clz
            n_img = len(os.listdir(path_class))
            nb_valid_sample += n_img

        train_datagen = ImageDataGenerator(rescale=1.0 / 255.0, horizontal_flip=True)
        valid_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

        if self._channel_num == 1:
            color_mode = "grayscale"
        else:
            color_mode = "rgb"

        train_generator = train_datagen.flow_from_directory(
            train_data_dir,
            color_mode=color_mode,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode="categorical",
        )
        valid_generator = valid_datagen.flow_from_directory(
            valid_data_dir,
            color_mode=color_mode,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode="categorical",
        )

        train_batch = nb_train_sample / batch_size
        valid_batch = nb_valid_sample / batch_size

        self._data_ready["train"]["generator"] = train_generator
        self._data_ready["train"]["batch"] = train_batch

        self._data_ready["valid"]["generator"] = valid_generator
        self._data_ready["valid"]["batch"] = valid_batch

    def _preproccess(self):
        if self._use_path:
            if isinstance(self._dataset, list):
                train_data_dir = self._dataset[0]
                valid_data_dir = self._dataset[1]
            else:
                path = self._dataset
                if path[-1] == "/":
                    path = self._dataset[:-1]
                train_data_dir = path + "/train"
                valid_data_dir = path + "/valid"

            self.__custom_dataset(train_data_dir, valid_data_dir)
        else:
            self.__keras_dataset_load()

    def load_data(self):
        return self._data_ready, self._use_path
