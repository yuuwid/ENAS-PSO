# DEFINE LAYER FOR SEARCH

layers_search = {
    # =========================================================
    # Layer Convolution (will be shuffle inside proccess)
    # Required:
    #   kernel_size: int, tuple
    #   filters: int
    #
    # Optional:
    #   strides: int, tuple | default: (1, 1)
    #   padding: str, int, tuple | default: "valid"
    #   activation: boolean, None | default: None
    "conv": [
        {"filters": 16, "kernel_size": (3, 3)},
        {"filters": 32, "kernel_size": (3, 3)},
        {"filters": 64, "kernel_size": (3, 3)},
        {"filters": 96, "kernel_size": (3, 3)},
        #
        {"filters": 16, "kernel_size": (3, 3), "activation": True},
        {"filters": 32, "kernel_size": (3, 3), "activation": True},
        {"filters": 64, "kernel_size": (3, 3), "activation": True},
        {"filters": 96, "kernel_size": (3, 3), "activation": True},
        #
        {"filters": 16, "kernel_size": (5, 5)},
        {"filters": 32, "kernel_size": (5, 5)},
        {"filters": 64, "kernel_size": (5, 5)},
        {"filters": 96, "kernel_size": (5, 5)},
        #
        {"filters": 16, "kernel_size": (5, 5), "activation": True},
        {"filters": 32, "kernel_size": (5, 5), "activation": True},
        {"filters": 64, "kernel_size": (5, 5), "activation": True},
        {"filters": 96, "kernel_size": (5, 5), "activation": True},
    ],
    # Layer Pooling
    # Duplicate it to increase rate
    #
    # Required:
    #   type: str | type = "max" or "avg"
    #   pool_size: int, tuple
    #
    # Optional:
    #   strides: int, tuple, None | default: None
    #   padding: str, int, tuple | default: "valid"
    # =========================================================
    # =========================================================
    "pool": [
        {"type": "max", "pool_size": (3, 3)},
        {"type": "max", "pool_size": (3, 3), "strides": (2, 2)},
        {"type": "max", "pool_size": (4, 4)},
        {"type": "max", "pool_size": (4, 4), "strides": (2, 2)},
        #
        {"type": "max", "pool_size": (3, 3)},
        {"type": "max", "pool_size": (3, 3), "strides": (3, 3)},
        {"type": "max", "pool_size": (4, 4)},
        {"type": "max", "pool_size": (4, 4), "strides": (3, 3)},
    ],
    # =========================================================
    # =========================================================
    # Feature Extraction Layer
    # Layer Fully Connected Layer
    #
    # Required:
    #   units: int
    #
    # Optional:
    #   dropout: float, boolean | default False
    #   activation: boolean, None | default None
    "fc": [
        {"units": 84, "activation": True},
        #
        {"units": 256, "dropout": True},
        {"units": 128, "dropout": True},
        {"units": 120, "dropout": True},
        {"units": 96, "dropout": True},
        {"units": 64, "dropout": True},
        #
        {"units": 256, "dropout": False, "activation": True},
        {"units": 128, "dropout": False, "activation": True},
        {"units": 120, "dropout": False, "activation": True},
        {"units": 96, "dropout": False, "activation": True},
        {"units": 64, "dropout": False, "activation": True},
        #
        {"units": 256, "dropout": True, "activation": True},
        {"units": 128, "dropout": True, "activation": True},
        {"units": 120, "dropout": True, "activation": True},
        {"units": 96, "dropout": True, "activation": True},
        {"units": 64, "dropout": True, "activation": True},
    ],
    # =========================================================
}
