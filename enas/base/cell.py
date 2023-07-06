import random

from ..utils.collection import Collection

from .node import Node


class Cell:
    def __init__(self, block_layers, is_fc=False, bn_rate=0.0, id=None):
        self.id = id
        self._block_layers = block_layers
        self._is_fc = is_fc

        self._bn_rate = bn_rate

        self._nodes = []

    def build_nodes(self):
        n_conv = len(Collection.get(key="conv"))
        n_pool = len(Collection.get(key="pool"))
        n_fc = len(Collection.get(key="fc"))

        for i in range(len(self._block_layers)):
            layer = self._block_layers[i]

            if layer == "conv":
                max_index = n_conv - 1
            elif layer == "pool":
                max_index = n_pool - 1
            elif layer == "fc":
                max_index = n_fc - 1
                
            index_layer = random.randint(0, max_index)

            use_bn = False
            if layer != "fc":
                prob = [self._bn_rate, 1 - self._bn_rate]
                use_bn = random.choices([True, False], prob)[0]

            id = "node-" + str(i + 1)

            node = Node(
                layer,
                index_layer,
                use_bn,
                id=id,
            )
            node.compile()

            node.build_layer()

            self._nodes.append(node)

    def get_nodes_param(self):
        ix = []
        for node in self._nodes:
            ix.append(node._index_layer)
        return ix

    def update_node(self, indexes_layer):
        for i in range(len(self._nodes)):
            ix_layer = indexes_layer[i]

            self._nodes[i].update_layer(ix_layer)
            self._nodes[i].compile()

    def decode_architecture(self):
        node_layers = []

        for node in self._nodes:
            node.decode()
            layers = node._compiled_layer

            for layer in layers:
                node_layers.append(layer)

        return node_layers
