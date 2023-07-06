import json
import os
import shutil


class JsonDummer:
    def __init__(self, filename: str, params: dict = {}, root_folder=None):
        if filename.split(".")[-1] != "json":
            filename = filename + ".json"
        self._filename = filename

        self._root_folder = root_folder

        self._json = params

    def put(self, key, values, replace=False):
        try:
            if isinstance(self._json[key], list):
                if replace:
                    self._json[key] = values
                else:
                    self._json[key].append(values)
            else:
                ori_val = [self._json[key]]
                if replace:
                    self._json[key] = values
                else:
                    ori_val.append(values)
                    self._json[key] = ori_val
        except Exception as ex:
            pass

    def build_path(self):
        if self._root_folder is not None:
            filepath = self._root_folder + "/" + self._filename
            return self._root_folder, filepath
        else:
            return self._root_folder, self._filename


class Json:
    def __init__(self, exp=None, replace=False):
        self._exp = exp
        self._replace = replace

        self._root_path = "enas/runs/"
        self._dummer = {}

        self.__check_exp()
        self.__create_dummer()

    def __check_exp(self):
        if self._exp is None:
            path_exp = self._root_path + "exp-1"
        else:
            path_exp = self._root_path + self._exp

        i = 2
        while True:
            if self._replace is False:
                if os.path.exists(path_exp):
                    if self._exp is None:
                        exp = "exp-" + str(i)
                        path_exp = self._root_path + exp
                    else:
                        path_exp = self._root_path + self._exp + "-" + str(i)
                else:
                    break
            else:
                break

            i = i + 1

        self._path_exp = path_exp
        self.__create_folder(self._replace)

    def __create_dummer(self):
        parameters_dump = JsonDummer(
            "parameters.json",
            {"pso": {}, "nas": {}, "cnn": {}, "blueprint": None},
        )
        architecture_dump = JsonDummer(
            "architecture.json",
            {"input": None, "layers": None, "output": None},
        )
        res_dump = JsonDummer("results.json", {"searching": []})

        self._dummer["parameters"] = parameters_dump
        self._dummer["architecture"] = architecture_dump
        self._dummer["results"] = res_dump

    def new(self, name: str, params: dict, root_folder=None):
        if name.split(".")[-1] == "json":
            name = name[:-1]

        new_dump = JsonDummer(name, params, root_folder)

        self._dummer[name] = new_dump

    def put(self, to_name, key, values, save=False, replace=False):
        """
        to_name: parameters, architecture, results, or name when call new() method
        """
        if self._dummer.get(to_name):
            self._dummer[to_name].put(key, values, replace)

        if save:
            self.save()

    def __create_folder(self, replace=False):
        if replace:
            if os.path.exists(self._path_exp):
                shutil.rmtree(self._path_exp)
            os.mkdir(self._path_exp)

        if os.path.exists(self._path_exp) is False:
            os.mkdir(self._path_exp)

    def __save(self):
        try:
            for dump in self._dummer.values():
                root, filepath = dump.build_path()
                if root is not None:
                    root = self._path_exp + "/" + root
                    if os.path.exists(root) is False:
                        os.mkdir(root)

                path = self._path_exp + "/" + filepath

                with open(path, "w") as f:
                    json.dump(dump._json, f)

        except Exception as ex:
            pass

    def save(self):
        self.__create_folder()
        self.__save()
