import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pathlib

# generate the compiled and converted files for lidar.pyx using cython in the directory .pyxbld
# auto recompile them at every edit on lidar.pyx

import pyximport

pyxbld_dir = pathlib.PurePath.joinpath(pathlib.Path().resolve(), '.pyxbld')
pyximport.install(build_dir=pyxbld_dir, reload_support=True, language_level=3)

import lidar


# find border positions at theta, given the midline cs
# need to move along the perpendicular of the midline direction
def borders(cs, theta, track_semi_width=.02):
    d = cs(theta, 1)
    dx = d[:, 0]
    dy = d[:, 1]
    # print(dx, dy)
    pdx, pdy = -dy, dx
    corr = track_semi_width / np.sqrt(pdx ** 2 + pdy ** 2)
    pos = cs(theta)
    x = pos[:, 0]
    y = pos[:, 1]
    return x + pdx * corr, y + pdy * corr, x - pdx * corr, y - pdy * corr


# adjust range in increasing order between 0 and 2*np.pi as required by CubicSpline
def adjust_range(angles, n):
    incr = 0
    last = 0
    fixed = []
    for i, theta in enumerate(angles):
        theta = theta + incr
        if theta < last:
            incr += 2 * np.pi
            theta += 2 * np.pi
        fixed.append(theta)
        last = theta
    return fixed


# takes in input a midline, that is a CubicSpline at angles theta, and
# returns border lines as similar Splines
def border_poly(cs, theta):
    n = len(theta) / 2
    x_in, y_in, x_out, y_out = borders(cs, theta)
    c_in = list(zip(x_in, y_in))
    c_out = list(zip(x_out, y_out))
    theta_in = [np.arctan2(y, x) for (x, y) in c_in]
    theta_in = adjust_range(theta_in, n)
    theta_out = [np.arctan2(y, x) for (x, y) in c_out]
    theta_out = adjust_range(theta_out, n)
    cs_in = CubicSpline(theta_in, c_in, bc_type='periodic')
    cs_out = CubicSpline(theta_out, c_out, bc_type='periodic')
    return cs_in, cs_out, theta_in, theta_out


# we try to avoid too sharp turns in tracks
def smooth(var):
    n = var.shape[0]
    if 2 * var[0] - (var[n - 1] + var[1]) > 1:
        var[0] = (1 + var[n - 1] + var[1]) / 2
    elif 2 * var[0] - (var[n - 1] + var[1]) < -1:
        var[0] = (var[n - 1] + var[1] - 1) / 2
    for i in range(1, n - 1):
        if 2 * var[i] - (var[i - 1] + var[i + 1]) > 1:
            var[i] = (1 + var[i - 1] + var[i + 1]) / 2
        elif 2 * var[i] - (var[i - 1] + var[i + 1]) < -1:
            var[i] = (var[i - 1] + var[i + 1] - 1) / 2
    if 2 * var[n - 1] - (var[n - 2] + var[0]) > 1:
        var[n - 1] = (1 + var[n - 2] + var[0]) / 2
    elif 2 * var[n - 1] - (var[n - 2] + var[0]) < -1:
        var[n - 1] = (var[n - 2] + var[0] - 1) / 2
    return var


def create_random_track(curves=20):
    theta = 2 * np.pi * np.linspace(0, 1, curves)
    var = np.random.rand(curves)
    var = smooth(var)
    var = var * .5 + .7
    var[curves - 1] = var[0]
    # midline
    y = np.c_[np.cos(theta) * var, np.sin(theta) * var]
    cs = CubicSpline(theta, y, bc_type='periodic')
    theta2 = 2 * np.pi * np.linspace(0, 1, 2 * curves)
    cs_in, cs_out, _, _ = border_poly(cs, theta2)
    return cs, cs_in, cs_out


def no_inversion(theta_new, theta_old):
    if theta_old < -np.pi * .9 and theta_new > np.pi * .9:
        theta_new = theta_new - np.pi * 2
    return theta_new < theta_old


def complete(theta_new, theta_old):
    return theta_old > 0 and theta_new <= 0


# starting from borders we create a dense grid of points corresponding to legal
# positions on the track. This map is what defines the actual track.

# filling all points between (x0,y0) and (x1,y1) on the map. For each point
# in the line we fill a small region 3x3 around it.
def fill(x0, y0, x1, y1, map):
    # print(x0,y0,x1,y1)
    i = 0
    j = 0
    dx = x1 - x0
    dy = y1 - y0
    if abs(dx) >= abs(dy):
        if x0 < x1:
            xstep = 1
        else:
            xstep = -1
        ystep = dy / dx
        for i in range(0, dx + xstep, xstep):
            j = int(ystep * i)
            map[x0 + i - 1:x0 + i + 2, y0 + j - 1:y0 + j + 2] = 1
        # print(i,j)
    else:
        if y0 < y1:
            y_step = 1
        else:
            y_step = -1
        x_step = dx / dy
        for j in range(0, dy + y_step, y_step):
            i = int(x_step * j)
            map[x0 + i - 1:x0 + i + 2, y0 + j - 1:y0 + j + 2] = 1
    if not (map[x1, y1] == 1):
        print(x0 + i, y0 + j)
    return map.astype('bool')


def create_route_map(inner, outer, discr=2000, show_map=False):
    map = np.zeros((1300, 1300)).astype('bool')
    rad = 2 * np.pi / discr
    for theta in range(discr):
        # print(theta)
        x_in, y_in = inner(theta * rad)
        x_out, y_out = outer(theta * rad)
        x_in = int(x_in * 500) + 650
        y_in = int(y_in * 500) + 650
        x_out = int(x_out * 500) + 650
        y_out = int(y_out * 500) + 650
        limit_check = 0 <= x_out < 1300 and 0 <= y_out < 1300
        if limit_check:
            fill(x_in, y_in, x_out, y_out, map)
        else:
            return map, False
    if show_map:
        plt.figure(figsize=(12, 6))
        plt.subplot(122)
        # plt.axis('off')
        plt.imshow(np.rot90(map))
        plt.subplot(121)
        axes = plt.gca()
        axes.set_xlim([-1.3, 1.3])
        axes.set_ylim([-1.3, 1.3])
        axes.set_aspect('equal')
        # plt.axis('off')
        xs = 2 * np.pi * np.linspace(0, 1, 200)
        plt.plot(inner(xs)[:, 0], inner(xs)[:, 1])
        plt.plot(outer(xs)[:, 0], outer(xs)[:, 1])
        # plt.axes.set_aspect('equal')
        plt.show()
    return map, True


def lidar_grid(x, y, vx, vy, map, angle=np.pi / 3, pins=19):
    return lidar.lidar_grid(x, y, vx, vy, map, angle, pins)


def get_new_angle(car_x, car_y, new_car_x, new_car_y):
    old_pos = [car_x, car_y]
    actual_pos = [new_car_x, new_car_y]
    unit_vector_1 = old_pos / np.linalg.norm(old_pos)
    unit_vector_2 = actual_pos / np.linalg.norm(actual_pos)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    theta = np.arccos(dot_product)
    return theta


def get_angle_from_start(car_x, car_y):
    old_pos = [1, 0]
    actual_pos = [car_x, car_y]
    unit_vector_1 = old_pos / np.linalg.norm(old_pos)
    unit_vector_2 = actual_pos / np.linalg.norm(actual_pos)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    theta = np.arccos(dot_product)
    theta = np.deg2rad(theta)
    return theta


#######################################################################################################################

class Racer:
    def __init__(self):
        self.curves = 20
        self.t_step = 0.1
        self.max_acc = 0.1
        self.max_turn = np.pi / 6

        self.cs, self.cs_in, self.cs_out = create_random_track(self.curves)
        self.map, legal_map = create_route_map(self.cs_in, self.cs_out)

        self.car_theta = 0  # polar angle w.r.t center of the route
        self.car_angle = 0
        self.car_x, self.car_y = self.cs(0)
        self.car_vx, self.car_vy = -self.cs(0, 1)
        self.done = False
        self.completed = False

    def reset(self):
        legal_map = False
        # map creation may fail in pathological cases
        # we try until a legal map is created
        while not legal_map:
            self.cs, self.cs_in, self.cs_out = create_random_track(self.curves)
            self.map, legal_map = create_route_map(self.cs_in, self.cs_out)
        self.car_theta = 0  # polar angle w.r.t center of the route
        self.car_angle = 0
        self.car_x, self.car_y = self.cs(0)
        self.car_vx, self.car_vy = -self.cs(0, 1)
        self.done = False
        self.completed = False
        v = np.random.uniform() * .5
        print("initial speed = ", v)
        v_norm = v / ((self.car_vx ** 2 + self.car_vy ** 2) ** .5)
        self.car_vx *= v_norm
        self.car_vy *= v_norm
        assert (self.map[int(self.car_x * 500) + 650, int(self.car_y * 500) + 650])
        lidar_signal = lidar_grid(self.car_x, self.car_y, self.car_vx, self.car_vy, self.map)

        return lidar_signal, v

    def step(self, action, reward_type):
        max_incr = self.max_acc * self.t_step
        acc, turn = action
        v = (self.car_vx ** 2 + self.car_vy ** 2) ** .5
        new_v = max(0, v + acc * max_incr)
        car_dir = np.arctan2(self.car_vy, self.car_vx)
        new_dir = car_dir - turn * self.max_turn
        new_car_vx = new_v * np.cos(new_dir)
        new_car_vy = new_v * np.sin(new_dir)
        new_car_x = self.car_x + new_car_vx * self.t_step
        new_car_y = self.car_y + new_car_vy * self.t_step
        new_car_theta = np.arctan2(new_car_y, new_car_x)
        on_route = self.map[int(new_car_x * 500) + 650, int(new_car_y * 500) + 650]
        if on_route and no_inversion(new_car_theta, self.car_theta):
            # reward based on angle between start and actual pos
            # reward = get_angle_from_start(self.car_x, self.car_y)
            if reward_type == 'polar':
                reward = get_new_angle(self.car_x, self.car_y, new_car_x, new_car_y)
            # TODO check reward value
            # reward based on increasing speed
            # reward = new_v * self.t_step
            else:
                reward = v * self.t_step
            # reward = self.t_step
            self.car_x = new_car_x
            self.car_y = new_car_y
            self.car_vx = new_car_vx
            self.car_vy = new_car_vy
            lidar_signal = lidar_grid(self.car_x, self.car_y, self.car_vx, self.car_vy, self.map)
            # dir,dist = max_lidar2(obs)
            if complete(new_car_theta, self.car_theta):
                print("completed")
                self.done = True
                self.completed = True
            self.car_theta = new_car_theta
            # TODO Check v -- new_v for reward value
            return (lidar_signal, v), reward, self.done, self.completed

        else:
            if not on_route:
                print("crossing border")
            else:
                print("wrong direction")
            self.done = True
            reward = -np.pi
            state = None
        return state, reward, True, False


def new_run(racer, actor, run_n):
    state = racer.reset()
    cs, cs_in, cs_out = racer.cs, racer.cs_in, racer.cs_out
    car_x, car_y = racer.car_x, racer.car_y
    fig, ax = plt.subplots(figsize=(6, 6))
    xs = 2 * np.pi * np.linspace(0, 1, 200)
    ax.plot(cs_in(xs)[:, 0], cs_in(xs)[:, 1])
    ax.plot(cs_out(xs)[:, 0], cs_out(xs)[:, 1])
    ax.axes.set_aspect('equal')

    line, = plt.plot([], [], color='b')
    x_data, y_data = [car_x], [car_y]

    acc = 0
    turn = 0

    def init():
        line.set_data([], [])
        return line,

    def counter():
        n = 0
        while not racer.done:
            n += 1
            yield n

    def animate(i):
        nonlocal state
        # t1 = time.time()
        action = actor(state)
        # t2 = time.time()
        # print("time taken by action = {} sec.".format(t2-t1))
        # t1 = time.time()
        state, reward, done, _ = racer.step(action)
        # t2 = time.time()
        # print("time taken by step = {} sec.".format(t2 - t1))
        x_data.append(racer.car_x)
        y_data.append(racer.car_y)
        line.set_data(x_data, y_data)
        return line,

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=counter, interval=5, blit=True, repeat=False)
    anim.save(f'animations/animation_{run_n}.gif')
    # plt.show()


def new_multi_run(actor, simulations=2):
    fig, axes = plt.subplots(simulations, simulations, figsize=(8, 8))
    for x_ax in axes:
        for ax in x_ax:
            ax.set_xticks([])
            ax.set_yticks([])
    line_grid = [[[] for j in range(simulations)] for i in range(simulations)]
    x_data_grid = [[[] for j in range(simulations)] for i in range(simulations)]
    y_data_grid = [[[] for j in range(simulations)] for i in range(simulations)]
    racer_grid = [[[] for j in range(simulations)] for i in range(simulations)]
    state_grid = [[[] for j in range(simulations)] for i in range(simulations)]
    done_grid = [[False for j in range(simulations)] for i in range(simulations)]
    completed_grid = [[False for j in range(simulations)] for i in range(simulations)]

    for sim_x in range(simulations):
        for sim_y in range(simulations):
            racer = Racer()
            racer_grid[sim_x][sim_y] = racer
            state = racer.reset()
            state_grid[sim_x][sim_y] = state

            cs, cs_in, cs_out = racer.cs, racer.cs_in, racer.cs_out
            car_x, car_y = racer.car_x, racer.car_y
            xs = 2 * np.pi * np.linspace(0, 1, 200)
            axes[sim_x, sim_y].plot(cs_in(xs)[:, 0], cs_in(xs)[:, 1])
            axes[sim_x, sim_y].plot(cs_out(xs)[:, 0], cs_out(xs)[:, 1])
            axes[sim_x, sim_y].axes.set_aspect('equal')

            line, = axes[sim_x, sim_y].plot([], [], lw=2)
            x_data, y_data = [car_x], [car_y]

            line_grid[sim_x][sim_y] = line
            x_data_grid[sim_x][sim_y] = x_data
            y_data_grid[sim_x][sim_y] = y_data

        def init():
            for sim_x in range(simulations):
                for sim_y in range(simulations):
                    line_grid[sim_x][sim_y].set_data([], [])
            return line,

        def counter():
            n = 0
            while any(not el for row in done_grid for el in row):
                n += 1
                yield n

        def animate(i):
            for sim_x in range(simulations):
                for sim_y in range(simulations):

                    tmp_done = False
                    completed_color = 'cyan'
                    error_color = 'red'

                    if done_grid[sim_x][sim_y]:
                        if completed_grid[sim_x][sim_y]:
                            color = completed_color
                        else:
                            color = error_color
                        line_grid[sim_x][sim_y], = axes[sim_x, sim_y].plot([], [], lw=2, color=color)

                    else:
                        action = actor(state_grid[sim_x][sim_y])
                        state_grid[sim_x][sim_y], reward, tmp_done, completed_grid[sim_x][sim_y] = racer_grid[sim_x][
                            sim_y].step(action)

                    done_grid[sim_x][sim_y] = done_grid[sim_x][sim_y] or tmp_done

                    x_data_grid[sim_x][sim_y].append(racer_grid[sim_x][sim_y].car_x)
                    y_data_grid[sim_x][sim_y].append(racer_grid[sim_x][sim_y].car_y)

                    line_grid[sim_x][sim_y].set_data(x_data_grid[sim_x][sim_y], y_data_grid[sim_x][sim_y])

            flat_grid = [item for sublist in line_grid for item in sublist]
            return flat_grid

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=counter, interval=5, save_count=250, blit=True,
                                   repeat=False)
    anim.save(f'animations/grid_animation.gif')

# racer = Racer()
# new_run(racer, my_actor)
