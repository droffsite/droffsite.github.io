from enum import Enum as __Enum
from threading import Thread as _Thread
from threading import RLock as _RLock
from multiprocessing import cpu_count as __cpu_count
import numpy as _np


class Normalization(__Enum):
    Intensity = 0
    Length = 1
    Log = 2


def smooth_hsv(complex_array, norm=Normalization.Length, max_cutoff=None, high_saturation=False,
               conserve_memory=False, intensity_scale=1):
    """
    This function takes a complex input array and return a list of [red, green, blue] arrays
    that correspond to the complex array via color -> complex angle, intensity -> complex magnitude
    :param complex_array: input array to colorize
    :param norm: The normalization to use
    :param max_cutoff: if set, this will truncate the intensity to max out at this value
    :param high_saturation: if set to True, will cause Sin()^4 to be used instead Sin()^2
    :param conserve_memory: if True, will use float16 (this is slow), else use float32
    :return:
    """
    if conserve_memory:
        float_type = _np.float16
    else:
        float_type = _np.float32

    width = complex_array.shape[0]
    slice_width = 16
    while width % slice_width != 0:
        slice_width -= 1

    lock = _RLock()
    shape = list(_np.shape(complex_array))
    colors = _np.zeros(shape + [3], dtype=float_type)
    magnitude = _np.zeros_like(complex_array, dtype=float_type)
    apply_list = list(range(int(width / slice_width)))

    def absolute():
        while True:
            with lock:
                if apply_list:
                    on = apply_list.pop(0)
                else:
                    return
            j = on * slice_width
            k = j + slice_width
            magnitude[j:k, ::] = _np.absolute(complex_array[j:k, ::]).astype(dtype=float_type)

    threads = list()
    for _ in list(range(__cpu_count())):
        threads.append(_Thread(target=absolute))
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    n = 2
    if high_saturation:
        n = 4

    max_magnitude = _np.max(magnitude)
    if max_cutoff is not None:
        max_cutoff /= max_magnitude

    apply_list = list(range(int(width / slice_width)))

    def parfunc():
        while True:
            with lock:
                if apply_list:
                    on = apply_list.pop(0)
                else:
                    return
            j = on * slice_width
            k = j + slice_width

            hue = (_np.angle(complex_array[j:k, ::], deg=True) + 90.) / 60.
            val = _np.array(magnitude[j:k, ::])

            if norm is Normalization.Intensity:
                val **= 2
                val /= max_magnitude ** 2
            elif norm is Normalization.Log:
                val = _np.log(val)
                _np.copyto(val, 0, where=val < 0)
                val /= _np.log(max_magnitude)
            else:
                val /= max_magnitude

            if max_cutoff is not None:
                _np.copyto(val, 1, where=val > max_cutoff)
                val /= max_cutoff

            pi6 = _np.pi / 6.

            colors[j:k, ::, 0] = val * _np.abs(_np.sin((hue - 0) * pi6)) ** n
            colors[j:k, ::, 1] = 0.6 * val * _np.abs(_np.sin((hue - 2) * pi6)) ** n
            colors[j:k, ::, 2] = val * _np.abs(_np.sin((hue - 4) * pi6)) ** n

            colors[j:k, ::, 1] += colors[j:k, ::, 2] * 0.35
            colors[j:k, ::, 1] += colors[j:k, ::, 0] * 0.1

    threads = list()
    for _ in list(range(__cpu_count())):
        threads.append(_Thread(target=parfunc))
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    return colors*intensity_scale
