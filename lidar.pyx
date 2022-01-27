import numpy as np

#cython syntax used to assign types to variables


#basic function for lidar. given a point (x0,y0) on the track and a direction dir_ang computes the
#distance of the border along the given direction
cpdef dist_grid(double x0, double y0, double dir_ang, map, float step=1. / 100, verbose=False):
    cdef double step_x = step * np.cos(dir_ang)
    cdef double step_y = step * np.sin(dir_ang)
    cdef double x = x0
    cdef double y = y0
    cdef int xg = int(x * 500) + 650
    cdef int yg = int(y * 500) + 650
    if not map[xg, yg]:
        print(x, y, xg, yg)
        print(map[xg, yg])
        assert (map[xg, yg])
    while map[xg, yg]:
        x += step_x
        y += step_y
        xg = int(x * 500) + 650
        yg = int(y * 500) + 650
    x -= step_x
    y -= step_y
    if step == 1. / 100:
        #print("reducing step")
        x, y = dist_grid(x, y, dir_ang, map, step=1. / 500, verbose=False)
    if verbose:
        print("start at = {}, cross border at {}".format((x0, y0), (x, y)))
    return x, y

cpdef lidar_grid(double x, double y, double vx, double vy, map, float angle=np.pi / 3, int pins=19):
    cdef double dir_ang = np.arctan2(vy, vx)  #car direction
    obs = np.zeros(pins)
    cdef int i = 0
    cdef double a = dir_ang - angle / 2
    cdef float a_step = angle / (pins - 1)
    cdef double cx
    cdef double cy
    for i in range(pins):
        cx, cy = dist_grid(x, y, a, map, verbose=False)
        obs[i] = ((cx - x) ** 2 + (cy - y) ** 2) ** .5
        a += a_step
    return obs
