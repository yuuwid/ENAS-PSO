import random
import math
import numpy as np
import copy

from ..base.cell import Cell
from ..model.nn import NeuralNetork


class Particle:
    def __init__(
        self,
        blueprint,
        model_builder: NeuralNetork = None,
        id=None,
    ):
        self.id = id
        self._model_builder = model_builder
        self._blueprint = blueprint

        self._cells = []

        self._model = None
        self._parameter_train = 0
        self._metrics_train = {}
        self._best_model = None

        self._position = []
        self._best_position = []
        self._velocity = []
        self._fitness = None

        self._tolerance_fitness = 3

    def build_cells(self, batch_normalization_rate=0.0):
        self._cells = []
        self._bnorm_rate = batch_normalization_rate

        n_cells = len(self._blueprint._architectures)

        fc = False
        for i in range(n_cells):
            id = "cell-" + str(i + 1)
            architecture = self._blueprint._architectures[i]
            if i + 1 == n_cells and self._blueprint._include_fc is True:
                fc = True

            cell = Cell(
                architecture,
                is_fc=fc,
                bn_rate=self._bnorm_rate,
                id=id,
            )

            cell.build_nodes()

            self._cells.append(cell)

        self.encode_position()
        self._best_position = self._position

    def encode_position(self):
        position = []
        velocity = []

        for cell in self._cells:
            position.append(cell.get_nodes_param())
            velocity.append(np.zeros(2).tolist())

        self._position = position
        self._velocity = velocity

    def update_velocity(self, gbest_pos, w, c1, c2):
        for i in range(len(self._velocity)):
            for j in range(len(self._velocity[i])):
                r1 = random.random()
                r2 = random.random()

                velo = self._velocity[i][j]

                cogn_comp = c1 * r1 * (self._best_position[i][j] - self._position[i][j])
                social_comp = c2 * r2 * (gbest_pos[i][j] - self._position[i][j])
                velo_new = w * velo + cogn_comp + social_comp

                self._velocity[i][j] = velo_new

    def update_position(self):
        for i in range(len(self._position)):
            for j in range(len(self._position[i])):
                position = self._position[i][j]

                position_new = position + self._velocity[i][j]

                if position_new - int(position_new) >= 0.5:
                    position_new = math.ceil(position_new)
                else:
                    position_new = round(position_new)

                self._position[i][j] = position_new

        self.__update_cell()

    def __update_cell(self):
        pos = self._position

        for i in range(len(self._cells)):
            self._cells[i].update_node(pos[i])

        self.encode_position()

    def evaluate_fitness(self):
        model_builder = self._model_builder

        if self._tolerance_fitness == 0:
            print("Overfitness: Try to Create New Model")
            self.build_cells(self._bnorm_rate)
            self._tolerance_fitness = 3
            self._fitness = None

        try:
            print("[START] Evaluate Fitness: " + str(self.id))

            self._parameter_train, fitness, self._metrics_train = model_builder.train(
                self._cells
            )

            if fitness == self._fitness:
                self._tolerance_fitness -= 1

            if self._fitness is None:
                self._fitness = fitness
                self._best_position = copy.deepcopy(self._position)
            else:
                if fitness > self._fitness:
                    self._fitness = fitness
                    self._best_position = copy.deepcopy(self._position)

            print("[ OUT ] Fitness         : " + str(self._fitness))
            print("[ END ] Evaluate Fitness: " + str(self.id))
            print()
        except Exception as err:
            # print(err)
            # input()
            print("Model Not Build !")
            if self._fitness is None:
                print("Trying to Create New Model...")
                self.build_cells(self._bnorm_rate)
                self._tolerance_fitness -= 1
                print("[ OUT ] Fitness         : " + str(self._fitness))
                print("[ END ] Evaluate Fitness: " + str(self.id))
                self.evaluate_fitness()
            else:
                self.evaluate_fitness()

    def desc_layers(self):
        layers = []

        for cell in self._cells:
            for node in cell._nodes:
                desc = node.desc()
                layers.append({node._layer_type: desc})

        return layers
