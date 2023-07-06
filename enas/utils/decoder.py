class Decoder:
    @staticmethod
    def decode_stride(layer):
        if layer.get("strides"):
            strides = layer["strides"]
            if isinstance(strides, tuple) == False:
                if isinstance(strides, str):
                    strides = int(strides)
                strides = (strides, strides)
            else:
                strides = (strides[0], strides[0]) if len(strides) == 1 else strides
            return strides
        return (1, 1)

    @staticmethod
    def decode_padding(layer):
        if layer.get("padding"):
            padding = layer["padding"]
            if isinstance(padding, tuple) == False:
                if isinstance(padding, str):
                    padding = padding if padding in ["same", "valid"] else "same"
                else:
                    padding = int(padding)
                    padding = (padding, padding)
            else:
                padding = (padding[0], padding[0]) if len(padding) == 1 else padding
            return padding
        return "valid"

    @staticmethod
    def decode_activation(layer):
        if layer.get("act"):
            return "relu" if layer["act"] is True else layer["act"]
        if layer.get("activation"):
            return "relu" if layer["activation"] is True else layer["activation"]
        return None

    @staticmethod
    def decode_dropout(layer):
        if layer.get("dropout"):
            return layer["dropout"]
        return None
