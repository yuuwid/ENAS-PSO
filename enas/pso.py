import os
import shutil
import time
import random
import matplotlib.pyplot as plt

from .optimizer.particle import Particle
from .optimizer.blueprint import Blueprint

from .model.nn import NeuralNetork

from .utils.data_loader import DataLoader
from .utils.collection import Collection
from .utils.json import Json


class PSO:
    def __init__(
        self,
        n_pop: int,
        c1: float,
        c2: float,
        w: float,
        exp=None,
        replace=False,
    ):
        # Start Session
        Collection.start_session()
        self._json_dumper = Json(exp=exp, replace=replace)

        self._n_pop = n_pop
        self._c1 = c1
        self._c2 = c2
        self._w = w

        self._gbest = [None, None]
        self._populations = []
        self._histories = []

        self._nn_params = {}

        self.__status = {
            "neural": False,
            "data_loader": False,
            "search": False,
        }

    def set_data_loader(self, data_loader: DataLoader):
        self._data_loader = data_loader
        self.__status["data_loader"] = True

    def init_neural(
        self,
        epochs_train=1,
        input_filters=32,
        input_kernel_size=(3, 3),
        input_padding="valid",
        input_strides=(1, 1),
        input_use_activation: bool = False,
        batch_normalization_rate=0.0,
    ):
        if self.__status["data_loader"] is False:
            msg = "Before call init_neural(), Please set data_loader with set_data_loader()"
            exit(msg)

        self._nn_params = {
            "input_size": self._data_loader._image_size,
            "input_channel": self._data_loader._channel_num,
            "output_class_number": self._data_loader._class_num,
            "batch_size": self._data_loader._batch_size,
            "epochs_train": epochs_train,
            "input_filters": input_filters,
            "input_kernel_size": input_kernel_size,
            "use_activation": input_use_activation,
            "input_padding": input_padding,
            "input_strides": input_strides,
        }

        self._json_dumper.put(
            "parameters", "cnn", self._nn_params, save=True, replace=True
        )

        if batch_normalization_rate > 1:
            batch_normalization_rate = 1.0
        elif batch_normalization_rate < 0:
            batch_normalization_rate = 0.0

        self._bnorm_rate = batch_normalization_rate
        self._model_builder = NeuralNetork(self._nn_params, self._data_loader)
        self.__status["neural"] = True

    def init_search(
        self,
        num_cells: int,
        num_nodes: int,
        include_fc: bool = False,
    ):
        if self.__status["neural"] is False:
            msg = "Before call init_search(), Please set neural parameters with init_neural()"
            exit(msg)

        if self.__status["data_loader"] is False:
            msg = "Before call init_search(), Please set data_loader with set_data_loader()"
            exit(msg)

        self._num_cells = num_cells
        self._num_nodes = num_nodes
        self._include_fc = include_fc

        nas = {
            "num_nodes": num_nodes,
            "num_cells": num_cells,
            "include_extractor": include_fc,
        }
        self._json_dumper.put("parameters", "nas", nas, save=True, replace=True)

        self.__init_blueprint()
        self.__initial_population()

    def __init_blueprint(self):
        blueprint = Blueprint(self._num_cells, self._num_nodes, self._include_fc)
        self._blueprint = blueprint

        self._json_dumper.put(
            "parameters",
            "blueprint",
            self._blueprint._architectures,
            save=True,
            replace=True,
        )

        input_layer = {
            "conv": {
                "filters": self._model_builder._input_filters,
                "kernel_size": self._model_builder._input_kernel_size,
                "padding": self._model_builder._input_padding,
                "activation": self._model_builder._use_activation,
                "input_shape": self._model_builder._input_shape,
            }
        }
        output_layer = {
            "fc": {
                "units": self._model_builder._output_class_number,
                "activation": "softmax",
            }
        }

        self._json_dumper.put("architecture", "input", input_layer, replace=True)
        self._json_dumper.put("architecture", "layers", None, replace=True)
        self._json_dumper.put("architecture", "output", output_layer, replace=True)
        self._json_dumper.save()

    def __initial_population(self):
        tolerances = 20
        passed = False

        start_time = time.time()

        for i in range(self._n_pop):
            id = "particle-" + str(i + 1)
            p = Particle(
                id=id,
                blueprint=self._blueprint,
                model_builder=self._model_builder,
            )
            p.build_cells(batch_normalization_rate=self._bnorm_rate)

            try_build_status = False
            for _ in range(tolerances):
                try_build_status = self._model_builder.try_build(p._cells)
                if try_build_status:
                    passed = True
                    break
                else:
                    if passed:
                        p.build_cells(batch_normalization_rate=self._bnorm_rate)

            if try_build_status is False and passed is False:
                exit("Model cannot Build")

            p.evaluate_fitness()

            self._populations.append(p)

            self._json_dumper.new(id, {"particle": []}, root_folder="particles")
            dump = {
                "iter": 0,
                "particle_id": p.id,
                "parameters": p._parameter_train,
                "metrics": p._metrics_train,
                "fitness": p._fitness,
                "architecture": p.desc_layers(),
            }
            self._json_dumper.put(id, "particle", dump)
            self._json_dumper.save()

        self.get_global_best()
        end_time = time.time()
        est_time = end_time - start_time

        result = {
            "epoch": 0,
            "best_particle": self._gbest[0].id,
            "best_fitness": self._gbest[1],
            "time_consume": est_time,
        }

        self._json_dumper.put("results", "searching", result, save=True)

    def get_global_best(self):
        best_particle = self._gbest[0]
        best_fitness = self._gbest[1]

        for p in self._populations:
            if best_fitness is None:
                best_particle = p
                best_fitness = p._fitness
            else:
                if p._fitness > best_fitness:
                    best_particle = p
                    best_fitness = p._fitness
                elif p._fitness == best_fitness:
                    if random.random() > 0.5:
                        best_particle = p
                        best_fitness = p._fitness

        self._gbest[0] = best_particle
        self._gbest[1] = best_fitness

        self._save_model("checkpoint")

        self._json_dumper.put(
            "architecture",
            "layers",
            best_particle.desc_layers(),
            save=True,
            replace=True,
        )

        self._histories.append(best_fitness)

    def get_best_fitness(self):
        return self._gbest[1]

    def get_best_architecture(self):
        model_builder = self._model_builder
        _, model = model_builder.create_model(cells=self._gbest[0]._cells)
        return model

    def search(self, epochs=10):
        self._epochs = epochs
        pso = {
            "n_pop": self._n_pop,
            "c1": self._c1,
            "c2": self._c2,
            "w": self._w,
            "epochs": self._epochs,
        }
        self._json_dumper.put("parameters", "pso", pso, save=True, replace=True)

        for epoch in range(epochs):
            start_time_epoch = time.time()

            print()
            msg = "EPOCH -- {epoch}/{epochs} START".format(
                epoch=epoch + 1,
                epochs=epochs,
            )
            print(msg)

            particle_fitness = []

            for p in self._populations:
                start_time_particle = time.time()
                p.evaluate_fitness()
                dump = {
                    "iter": epoch + 1,
                    "particle_id": p.id,
                    "parameters": p._parameter_train,
                    "metrics": p._metrics_train,
                    "fitness": p._fitness,
                    "architecture": p.desc_layers(),
                }
                self._json_dumper.put(p.id, "particle", dump, save=True)
                end_time_particle = time.time()
                est_time_particle = end_time_particle - start_time_particle
                particle_fitness.append(
                    {p.id: p._fitness, "time_consume": est_time_particle}
                )

            self.get_global_best()

            print("Gbest: ", self._gbest[0].id)

            for p in self._populations:
                p.update_velocity(
                    self._gbest[0]._position,
                    w=self._w,
                    c1=self._c1,
                    c2=self._c2,
                )
                p.update_position()

            end_time_epoch = time.time()
            est_time_epoch = end_time_epoch - start_time_epoch

            result = {
                "epoch": epoch + 1,
                "best_particle": self._gbest[0].id,
                "best_fitness": self._gbest[1],
                "time_consume": est_time_epoch,
                "particle_fitness": particle_fitness,
            }

            self._json_dumper.put("results", "searching", result, save=True)

            msg = "EPOCH -- {epoch}/{epochs} END".format(epoch=epoch + 1, epochs=epochs)
            print(msg)
            print()

        self._save_model("best")

    def _save_model(self, model_name="checkpoint"):
        try:
            model = self.get_best_architecture()
            optimizer = Collection.get(self._nn_params, "optimizer", "adam")
            metrics = Collection.get(self._nn_params, "metrics", ["accuracy"])

            model.compile(
                optimizer=optimizer,
                loss="categorical_crossentropy",
                metrics=metrics,
            )

            path = self._json_dumper._path_exp + "/model"

            if os.path.exists(path) is False:
                os.mkdir(path)

            model_path = path + "/" + model_name + ".h5"
            viz_name = path + "/viz-model-" + model_name + ".png"

            model.save(model_path)
            NeuralNetork.viz_model(model, to_file=viz_name)
            self._save_analytics()

        except Exception as ex:
            print(ex)
            print("Best Architecture Not Found | please re-search with larger epochs")
            exit()

    def _save_analytics(self):
        best_fitnesses = self._histories
        x_axis = [i for i in range(1, len(best_fitnesses) + 1)]

        fitness_epoch = self._json_dumper._path_exp + "/results-fitness-epochs.png"
        plt.plot(x_axis, best_fitnesses)
        plt.xlabel("Epochs")
        plt.ylabel("Fitness")
        plt.suptitle("RESULTS ENAS-PSO")
        plt.savefig(fitness_epoch)
        plt.close()
