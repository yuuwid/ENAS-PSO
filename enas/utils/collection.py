import random
from ..core.constant import layers_search
from .cache import Cache


class Collection:
    @staticmethod
    def start_session():
        Cache.start_session()

        Cache.add_cache("layers")

        Cache.insert_cache("layers", "keys", ["conv", "pool", "fc"])

        conv_cached = list(Collection.get(layers_search, "conv").copy())
        random.shuffle(conv_cached)
        Cache.insert_cache("layers", "conv", conv_cached)

        pool_cached = list(Collection.get(layers_search, "pool").copy())
        random.shuffle(pool_cached)
        Cache.insert_cache("layers", "pool", pool_cached)

        fc_cached = list(Collection.get(layers_search, "fc").copy())
        random.shuffle(fc_cached)
        Cache.insert_cache("layers", "fc", fc_cached)

    @staticmethod
    def all_layers_cached():
        return Cache.get_cache('layers')

    @staticmethod
    def get(data: dict = None, key=None, default=None):
        if data is None:
            data = Cache.get_cache("layers")

        if data.get(key):
            return data[key]
        return default

    @staticmethod
    def dict_to_list(dict_in=None, key="all"):
        if dict_in is None:
            dict_in = Collection.all_layers_cached()

        flatten_dict = []

        if key == "all":
            for key, layers in dict_in.items():
                for l in layers:
                    l["ztype"] = key
                    flatten_dict.append(l)
        else:
            for l in Collection.get(dict_in, key):
                l["ztype"] = key
                flatten_dict.append(l)

        return flatten_dict

    @staticmethod
    def get_in_flatten_dict(dict_in, index, key="all", use_cache=False):
        if use_cache:
            cached = Cache.get_cache("layers")
            res = Collection.dict_to_list(cached, key)
        else:
            res = Collection.dict_to_list(dict_in, key)
        return res[index]
