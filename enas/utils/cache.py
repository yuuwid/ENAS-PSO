class Cache:
    cached = {}

    @staticmethod
    def start_session():
        Cache.cached = {}

    @staticmethod
    def add_cache(key, val={}):
        Cache.cached[key] = val

    @staticmethod
    def insert_cache(cache, key, val):
        Cache.cached[cache][key] = val

    @staticmethod
    def get_cache(cache):
        return Cache.cached[cache]
