import os
import math

import tensorflow as tf

import numpy as np

from keras import backend as K
from keras import utils

from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.models import Sequential
from keras.utils import plot_model
from keras.utils.layer_utils import count_params

from ..utils.collection import Collection


class NeuralNetork:
    def __init__(self, nn_params, data_loader):
        self._data_loader = data_loader
        self._nn_params = nn_params

        self._input_size = Collection.get(nn_params, "input_size", (64, 64))
        self._input_channel = Collection.get(nn_params, "input_channel", 3)
        self._output_class_number = Collection.get(nn_params, "output_class_number")

        self._input_filters = Collection.get(nn_params, "input_filters", (3, 3))
        self._input_kernel_size = Collection.get(nn_params, "input_kernel_size", 32)
        self._use_activation = Collection.get(nn_params, "use_activation", False)
        self._input_padding = Collection.get(nn_params, "input_padding", "valid")
        self._input_strides = Collection.get(nn_params, "input_strides", (1, 1))

        if K.image_data_format() == "channels_first":
            self._input_shape = (
                self._input_channel,
                self._input_size,
                self._input_size,
            )
        else:
            self._input_shape = (
                self._input_size,
                self._input_size,
                self._input_channel,
            )

        if self._use_activation:
            self._use_activation = "relu"
        else:
            self._use_activation = None

    def __reset_model(self):
        K.clear_session()

    def root_model(self):
        input_layer = Conv2D(
            self._input_filters,
            self._input_kernel_size,
            padding=self._input_padding,
            strides=self._input_strides,
            activation=self._use_activation,
            input_shape=self._input_shape,
        )

        output_layer = Dense(self._output_class_number, activation="softmax")

        return input_layer, output_layer

    def count_params(self, model):
        trainable_count = count_params(model.trainable_weights)
        non_trainable_count = count_params(model.non_trainable_weights)

        params = str(trainable_count + non_trainable_count)

        return format(int(params), ",")

    def create_model(self, cells):
        # Reset Model Keras
        self.__reset_model()

        input_layer, output_layer = self.root_model()

        # Initial Model
        model = Sequential()

        model.add(input_layer)
        flatten = True

        for cell in cells:
            if cell._is_fc and flatten:
                flatten = False
                model.add(Flatten())

            node_architecures = cell.decode_architecture()

            for layer in node_architecures:
                model.add(layer)

        if flatten:
            model.add(Flatten())

        model.add(output_layer)

        return self.count_params(model), model

    def train(self, cells):
        K.clear_session()
        data_loader = self._data_loader
        data_ready, custom_dataset = data_loader.load_data()

        optimizer = Collection.get(self._nn_params, "optimizer", "adam")
        metrics = Collection.get(self._nn_params, "metrics", ["accuracy"])

        params, model = self.create_model(cells=cells)

        print("Parameters Train:", params)

        model.compile(
            optimizer=optimizer,
            loss="categorical_crossentropy",
            metrics=metrics,
        )

        if custom_dataset:
            model.fit(
                data_ready["train"]["generator"],
                steps_per_epoch=data_ready["train"]["batch"],
                epochs=Collection.get(self._nn_params, "epochs_train", 10),
                # workers=4,
            )
            result = model.evaluate(
                data_ready["valid"]["generator"],
                verbose=0,
            )
        else:
            model.fit(
                data_ready["train"]["x"],
                data_ready["train"]["y"],
                epochs=Collection.get(self._nn_params, "epochs_train", 10),
                batch_size=Collection.get(self._nn_params, "batch_size", None),
            )
            result = model.evaluate(
                data_ready["valid"]["x"],
                data_ready["valid"]["y"],
                verbose=0,
            )

        fitness, metrics_ret = self.__calculate_fitness(result)

        return params, fitness, metrics_ret

    def __calculate_fitness(self, result):
        # Normalisasi nilai Loss
        loss = result[0]
        loss_norm = math.log10(loss)
        # loss_norm = loss
        accuracy = result[1]

        w_acc = 0.2
        w_loss = 0.8

        fitness = (w_acc * accuracy) + (w_loss * (1 - loss_norm))

        metrics_ret = {"loss": loss, "accuracy": accuracy}

        return fitness, metrics_ret

    def try_build(self, cells):
        try:
            optimizer = Collection.get(self._nn_params, "optimizer", "adam")
            metrics = Collection.get(self._nn_params, "metrics", ["accuracy"])

            _, model = self.create_model(cells=cells)

            model.compile(
                optimizer=optimizer,
                loss="categorical_crossentropy",
                metrics=metrics,
            )

            return True
        except:
            return False

    @staticmethod
    def viz_model(model, to_file="model.png"):
        plot_model(
            model,
            to_file=to_file,
            show_shapes=True,
            show_layer_activations=True,
        )
