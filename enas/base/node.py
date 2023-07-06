import random

from ..core.constant import layers_search
from ..utils.collection import Collection, Cache
from ..utils.threshold import Threshold
from ..utils.decoder import Decoder

from keras.layers import (
    Conv2D,
    MaxPooling2D,
    AveragePooling2D,
    Dense,
    Dropout,
    BatchNormalization,
)


class Node:
    def __init__(self, layer_type, index_layer, use_bn, id=None):
        self.id = id

        self._layer_type = layer_type
        self._index_layer = index_layer
        self._use_bn = use_bn

        self._layer = None

        self._compiled_layer = []

    def compile(self):
        self.encode()

    def encode(self):
        layer = Collection.get_in_flatten_dict(
            None,
            self._index_layer,
            key=self._layer_type,
            use_cache=True,
        )
        self._layer = layer

    def build_layer(self):
        self.decode()

    def decode(self):
        layer = self._layer
        self._compiled_layer = []

        if self._use_bn:
            bn_layer = BatchNormalization()
            self._compiled_layer.append(bn_layer)

        if layer["ztype"] == "conv":
            filters = layer["filters"]
            kernel_size = layer["kernel_size"]

            # Get Optional Parameter
            stride = Decoder.decode_stride(layer)
            padding = Decoder.decode_padding(layer)
            activation = Decoder.decode_activation(layer)
            conv_layer = Conv2D(
                filters=filters,
                kernel_size=kernel_size,
                strides=stride,
                padding=padding,
                activation=activation,
            )
            self._compiled_layer.append(conv_layer)

        elif layer["ztype"] == "pool":
            pool_size = layer["pool_size"]
            if layer["type"] == "max":
                pool_layer = MaxPooling2D(pool_size=pool_size)
            elif layer["type"] == "avg":
                pool_layer = AveragePooling2D(pool_size=pool_size)

            self._compiled_layer.append(pool_layer)

        elif layer["ztype"] == "fc":
            units = layer["units"]

            activation = Decoder.decode_activation(layer)
            dropout = Decoder.decode_dropout(layer)

            fc_layer = Dense(units=units)

            self._compiled_layer.append(fc_layer)

            if dropout is not None:
                rate = 0.5 if isinstance(dropout, bool) else dropout
                do_layer = Dropout(rate=rate)
                self._compiled_layer.append(do_layer)

    def desc(self):
        desc = self._layer.copy()
        if self._use_bn:
            desc["batch_normalization"] = True
        desc.pop("ztype")
        return desc

    def update_layer(self, index_layer):
        n_layer = len(Collection.dict_to_list(layers_search, key=self._layer_type))

        index_layer = Threshold.handle(
            index_layer,
            min_threshold=0,
            max_threshold=n_layer - 1,
            behaviour="none",
        )

        self._index_layer = index_layer
