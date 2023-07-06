import random


class Threshold:
    @staticmethod
    def handle(value, min_threshold, max_threshold, behaviour="none"):
        """
        behavior is the conditional change in value when it exits a specified threshold

        'none', there is no behavior.
        when the value is less than min_threshold,
        the value will be changed to min_threshold,
        when the value is greater than max_threshold,
        then the value will be changed to max_threshold

        'absolute' or 'abs', (the conditional min_threshold equals 0).
        forces values below the threshold to be unsigned.
        when the value is less than 0,
        then the value is changed to an absolute value.
        for example value = -1 then new_value = 1

        'turn', when the value is below or above the threshold,
        the value will return to the scope of the threshold based on the difference.
        eg value = 10, threshold = 8, then new_value = 8 - (10-8) = 6

        'flip', when the value is below min_threshold,
        the value will be changed to max_threshold minus the difference.
        for example value = 0, min_threshold = 1, max_threshold = 5,
        then new_value = 5 - (1-0) = 4.

        when the value is above max_threshold,
        then the value is changed to min_threshold plus the difference.
        for example value = 6, min_threshold = 1, max_threshold = 5,
        then new value = 1 + (6-5) = 2
        """
        new_value = value

        if behaviour == "none":
            new_value = Threshold.__none_behaviour(
                new_value,
                min_threshold,
                max_threshold,
            )
        elif behaviour == "absolute" or behaviour == "abs":
            if min_threshold >= 0:
                new_value = Threshold.__abs_behaviour(
                    new_value,
                    min_threshold,
                    max_threshold,
                )
            else:
                new_value = Threshold.__none_behaviour(
                    new_value,
                    min_threshold,
                    max_threshold,
                )
        elif behaviour == "turn":
            new_value = Threshold.__turn_behaviour(
                new_value,
                min_threshold,
                max_threshold,
            )
        elif behaviour == "flip":
            new_value = Threshold.__flip_behaviour(
                new_value,
                min_threshold,
                max_threshold,
            )
        elif behaviour == "rand" or behaviour == "random":
            new_value = Threshold.__rand_behaviour(
                new_value,
                min_threshold,
                max_threshold,
            )
        else:
            new_value = Threshold.__none_behaviour(
                new_value,
                min_threshold,
                max_threshold,
            )

        return new_value

    @staticmethod
    def __none_behaviour(val, min_th, max_th):
        if val < min_th:
            val = min_th
        elif val > max_th:
            val = max_th
        return val

    @staticmethod
    def __abs_behaviour(val, min_th, max_th):
        if val < 0:
            val = abs(val)
            val = Threshold.__none_behaviour(val, min_th, max_th)
        elif val > max_th:
            val = max_th
        return val

    @staticmethod
    def __rand_behaviour(val, min_th, max_th):
        if val > max_th or val < min_th:
            return random.randint(min_th, max_th)
        else:
            return val

    @staticmethod
    def __turn_behaviour(val, min_th, max_th):
        # TODO: Develop
        return Threshold.__none_behaviour(val, min_th, max_th)

    @staticmethod
    def __flip_behaviour(val, min_th, max_th):
        # TODO: Develop
        return Threshold.__none_behaviour(val, min_th, max_th)
