#!/usr/bin/env python3


import numpy as np
import argparse
import random


def circ_2_points(p1, p2):
    """Find the center and the radius of the circle defined by 2 points."""
    center = (p1 + p2) / 2
    radius = np.linalg.norm(p1 - p2) / 2
    return center, radius


def circ_3_points(p1, p2, p3):
    """Find the center and the radius of the circle defined by 3 points."""
    A = np.array([p2 - p1, p3 - p1])
    b = np.array([np.dot(p2, p2) - np.dot(p1, p1), np.dot(p3, p3) - np.dot(p1, p1)]) / 2
    center = np.linalg.solve(A, b)
    radius = np.linalg.norm(center - p1)
    return center, radius


def b_md(arr, p, b):
    """Subroutine of Welzl algorithm."""
    if len(p) == 0:
        if len(b) == 0:
            return None, None
        elif len(b) == 1:
            return arr[b[0], :], 0
        elif len(b) == 2:
            return circ_2_points(arr[b[0], :], arr[b[1], :])
        else:
            return circ_3_points(arr[b[0], :], arr[b[1], :], arr[b[2], :])
    else:
        c, r = b_md(arr, p[:-1], b)
        if c is None or np.linalg.norm(arr[p[-1], :] - c) >= r:
            c, r = b_md(arr, p[:-1], b + [p[-1]])
    return c, r


def one_center_plane(points: np.array, shuffle: bool = False):
    """Solve the 1-center (minmax) facility location problem on the plane using Welzl algorithm.
    The solution is the smallest enclosing circle of the points.

    Args:
        points (np.array): array of points with shape (n,2)
        shuffle (bool, optional): shuffle the points first. Defaults to False.

    Returns:
        (center, radius) - the smallest enclosing cirlce
    """
    n = points.shape[0]
    p = list(range(n))
    if shuffle:
        random.shuffle(p)
    return b_md(points, p, [])


def geometric_median_plane(
    points: np.array,
    w: np.array = None,
    eps: float = 1e-6,
    max_iters: int = 100000,
    y0: np.array =None
):
    """Find the (weighted) geometric median on the plane using Weiszfield algorithm.

    Args:
        points (np.array): points
        w (np.array, optional): weights. Defaults to None.
        eps (float, optional): tolerance. Defaults to 1e-6.
        max_iters (int, optional): maximum number of iterations. Defaults to 100000.
        y0 (np.array, optional): initial guess (if None, the mean point is used). Defaults to None.

    Returns:
        np.array: geometric median
    """
    if w is None:
        w = np.ones(points.shape[0])

    if y0 is None:
        y = np.average(points, axis=0, weights=w)
    else:
        y = y0.copy()

    converged = False
    for i in range(max_iters):
        y_prev = y
        invd = w / np.linalg.norm(points - y, axis=1)
        y = invd @ points / np.sum(invd)

        # clamp the current guess
        y[0] = np.fmod(y[0] + 90, 180) - 90
        y[1] = np.fmod(y[1] + 180, 360) - 180

        if np.linalg.norm(y - y_prev) < eps:
            converged = True
            break

    return y, converged


def great_circle_dists(x: np.array, ys: np.array):
    """Compute great circle distance between x and each of the points ys.

    Args:
        x (np.array): the "source", shape (2,)
        ys (np.array): the "destinations", shape (n,2)

    Returns:
        np.array: distances, shape (n,)
    """
    # x[0] = lat = phi
    # x[1] = lon = lambda
    theta = np.cos(x[0]) * np.cos(ys[:, 0]) * np.cos(x[1] - ys[:, 1]) + np.sin(x[0]) * np.sin(ys[:, 0])
    return np.arccos(theta)


def geometric_median_sphere(
    points: np.array,
    w: float = None,
    eps: float = 1e-6,
    max_iters: int = 100,
    y0: np.array = None,
    units: str = 'degrees'
):
    """Find the (weighted) geometric median on the sphere using algorithm from
    Robert F Love, James G Morris, and George O Wesolowsky. Facilities location. 1988.

    Args:
        points (np.array): points
        w (float, optional): weights. Defaults to None.
        eps (float, optional): tolerance. Defaults to 1e-6.
        max_iters (int, optional): maximum number of iterations. Defaults to 100.
        y0 (np.array, optional): initial guess (if None, the mean point is used). Defaults to None.
        units (str, optional): units of inputs and outputs. Defaults to 'degrees'.

    Returns:
        np.array: geometric median
    """

    if w is None:
        w = np.ones(points.shape[0])

    if units == 'degrees':
        a = np.pi * points / 180
    elif units == 'radians':
        a = points
    else:
        raise ValueError(units)

    if y0 is None:
        y = np.mean(points, axis=0)
    else:
        if units == 'degrees':
            y = np.pi * y0.copy() / 180
        elif units == 'radians':
            y = y0.copy()
        else:
            raise ValueError(units)

    # clamp the inital guess
    y[0] = np.fmod(y[0] + np.pi / 2, np.pi) - np.pi / 2
    y[1] = np.fmod(y[1] + np.pi, 2 * np.pi) - np.pi

    converged = False
    for i in range(max_iters):
        y_prev = y

        sin_dists = np.sin(great_circle_dists(y, a))
        sin_a_lat = np.sin(a[:, 0])
        cos_a_lat = np.cos(a[:, 0])
        sin_a_lon = np.sin(a[:, 1])
        cos_a_lon = np.cos(a[:, 1])

        s1 = np.sum(w * cos_a_lat * sin_a_lon / sin_dists)
        s2 = np.sum(w * cos_a_lat * cos_a_lon / sin_dists)
        s3 = np.sum(w * sin_a_lat / sin_dists)

        y_lon = np.arctan(s1 / s2)
        y_lat = np.arctan(s3 * np.sin(y_lon) / s1)
        y_lat = np.fmod(y_lat + np.pi / 2, np.pi) - np.pi / 2
        y_lon = np.fmod(y_lon + np.pi, 2 * np.pi) - np.pi
        y = np.array([y_lat, y_lon])

        if np.linalg.norm(y - y_prev) < eps:
            converged = True
            break

    # check the guess and its antipode
    ya = np.array([-y[0], np.pi + y[1]])
    w_y = np.average(np.linalg.norm(a - y, axis=1), axis=0, weights=w)
    w_ya = np.average(np.linalg.norm(a - ya, axis=1), axis=0, weights=w)
    if w_ya < w_y:
        y = ya

    if units == 'degrees':
        y = y * 180 / np.pi

    return y, converged


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Solve the facility location problem.'
    )
    parser.add_argument(
        'method',
        type=str,
        help='method',
        choices='median_plane,median_plane_weighted,median_sphere,median_sphere_weighted,one_center_plane'.split(',')
    )
    parser.add_argument('filename', help='csv input file describing the facilities with format lat,lon[,weight]')
    args = parser.parse_args()

    try:
        data = np.loadtxt(args.filename, delimiter=',')
    except Exception as e:
        print(f'Error while reading input file: {e}')
        exit(1)

    # remove duplicates
    # TODO: merge close points?
    data = np.unique(data, axis=0)

    if args.method == 'median_plane':
        points = data[:, :2]
        y, converged = geometric_median_plane(points)
    elif args.method == 'median_sphere':
        points = data[:, :2]
        y, converged = geometric_median_sphere(points)
    elif args.method == 'median_plane_weighted':
        points = data[:, :2]
        w = data[:, 2]
        y, converged = geometric_median_plane(points, w)
    elif args.method == 'median_sphere_weighted':
        points = data[:, :2]
        w = data[:, 2]
        y, converged = geometric_median_sphere(points, w)
    elif args.method == 'one_center_plane':
        points = data[:, :2]
        y, _ = one_center_plane(points, shuffle=True)
    else:
        print(f'Unknown method: {args.method}.')
        exit(1)

    print(f'{y[0]},{y[1]}')