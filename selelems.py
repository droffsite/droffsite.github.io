import numpy as np

# Return a binary circle array with radius r

def circle_array(r=5, empty=False):
    """Return a binary circle array with radius r"""
    r2 = r**2
    dim = 2 * r + 1

    arr = np.asarray([[(x - r)**2 + (y - r)**2 <= r2 for y in range(r + 1)] for x in range(r + 1)], dtype=np.uint8)

    if empty:
        tmp = np.zeros_like(arr)
        tmp[0, :] = arr[0, :]
        tmp[:, 0] = arr[:, 0]

        tmp[1:, 1:] = [[arr[y, x] if arr[y, x - 1] != 1 else 0 for x in range(1, r + 1)] for y in range(1, r + 1)]

        arr = tmp

    arr = np.concatenate((arr, np.flip(arr[:, :-1], axis=1)), axis=1)

    return np.concatenate((arr, np.flip(arr[:-1, :])))


# Return a binary diamond array with dims (n, n)

def diamond_array(n, empty=False):
    """Return a binary diamond array with dims (n, n)"""
    a = np.arange(n)

    # Minimum of array a and reversed a
    b = np.minimum(a, a[::-1])

    # Broadcast b into column array, add to original array, convert boolean to int
    c = b[:, np.newaxis] + b >= n//2 if not empty else b[:, np.newaxis] + b == n//2

    return c.astype(np.uint8)


# Return an X-shaped array. n must be odd.

def x_array(n):
    """Return an X-shaped array. n must be odd."""
    mid = n//2
    a = diamond_array(n, True)

    b = np.zeros_like(a)
    b[:, 0:mid + 1] = diamond_array(n, True)[:, mid:]
    b[:, mid + 1:] = diamond_array(n, True)[:, 1:mid + 1]

    return b


# Return a plus-shaped array. n must be odd.

def plus_array(n):
    """Return a plus-shaped array. n must be odd."""
    mid = n//2
    a = np.zeros((n, n))
    a[:, mid] = 1
    a[mid, :] = 1

    return a.astype(np.uint8)