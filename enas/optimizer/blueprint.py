import random

from ..utils.collection import Collection


class Blueprint:
    def __init__(self, num_cells, num_nodes, include_fc=False):
        self._num_cells = num_cells
        self._num_nodes = num_nodes
        self._include_fc = include_fc
        self._architectures = self.__build()

    def __build(self):
        architectures = []

        fc = False
        for cell in range(self._num_cells):
            temp_node = []
            if cell == self._num_cells - 1 and self._include_fc is True:
                num_nodes = self._num_nodes
                fc = True
            else:
                num_nodes = self._num_nodes

            for _ in range(num_nodes):
                if fc:
                    layer = "fc"
                else:
                    for _ in range(5):
                        layer = random.choices(["conv", "pool"], [0.7, 0.4])[0]
                temp_node.append(layer)
            architectures.append(temp_node)

        return architectures

    def rebuild(self):
        self._architectures = self.__build()
