import numpy as np
import random as rd
import typing
from perlin_numpy.perlin2d import generate_fractal_noise_2d

class PerlinMap:
    # Pravi objekat PerlinMap, mora na pocetku
    def __init__(self, map: np.ndarray, map_dim: int, climber_x: int, climber_y: int, final_x: int, final_y: int, steps: int, max_steps: int, done: bool = False):
        self.map: np.ndarray = map
        self.map_dim: int = map_dim
        self.climber_x: int = climber_x
        self.climber_y: int = climber_y
        self.final_x: int = final_x
        self.final_y: int = final_y
        self.steps: int = steps
        self.max_steps: int = max_steps
        self.done: bool = done

    # Pravi perlin noise mapu
    # Resolution, dimension, octaves, persistence, lacunarity
    # To create perlin noise height map dim must be a multiple of lacunarity^(octaves-1)*res
    def create_map(self, res: int, oct: int = 1, pers: float = 0.5) -> None| int:

        # Is able to create perlin noise map?
        lac = 2
        if self.map_dim % pow(lac, oct-1)*res != 0:
            return None

        map_raw: np.ndarray = generate_fractal_noise_2d((self.map_dim, self.map_dim), (res, res), oct, pers)
        map_scaled: np.ndarray = (map_raw - map_raw.min())/(map_raw.max() - map_raw.min())
        self.map = map_scaled
        return 1

    # Proverava da li je penjac dostaigao cilj
    def check_fin(self, x: int, y: int) -> bool:
        if x == self.final_x and y == self.final_y:
            self.done = True
            return True
        else:
            return False

    # Racuna razliku u visini izmedju trenutno polozaja i polozaja u koji ce se pomeriti
    def h_diff(self, new_climber_x: int, new_climber_y: int) -> float:
        return self.map[new_climber_y, new_climber_x] - self.map[self.climber_y, self.climber_x]

    def euclid_dist(self, new_climber_x: int, new_climber_y: int) -> float:
        dist_x = self.final_x - new_climber_x
        dist_y = self.final_y - new_climber_y

        return pow(dist_x**2 + dist_y**2,0.5)

    # Racuna nagradu
    def calc_rew(self, new_climber_x: int, new_climber_y: int) -> float:
        rew = 0
        step_penalty = 0.5
        climbing_penalty_factor = 6
        euclid_dist_factor = 2

        # Reward for reaching the final pos
        if self.check_fin(new_climber_x, new_climber_y): rew += 100

        # Penalty for taking steps and going uphill
        # if self.h_diff(new_climber_x, new_climber_y) > 0:
        #     rew = rew - step_penalty - climbing_penalty_factor * self.h_diff(new_climber_x, new_climber_y)
        # else:
        #    rew = rew - step_penalty

        # Penalty for taking steps and for changing hight and for euclidan distance
        rew = rew - step_penalty - climbing_penalty_factor * self.h_diff(new_climber_x, new_climber_y)**2 - euclid_dist_factor * self.euclid_dist(new_climber_x, new_climber_y) / self.map_dim

        return rew

    # Izvrsava korak
    def step(self, dir: int) -> tuple[np.ndarray, float, bool, dict[typing.Never, typing.Never]]:
        # Dir:
        # -------------
        # 0 up      y-1
        # 1 right   x+1
        # 2 down    y+1
        # 3 left    x-1
        # -------------

        if self.steps >= self.max_steps: self.done = True

        new_climber_x, new_climber_y = 0,0
        match dir:
            case 0:
                if self.climber_y - 1 < 0: return (self.get_state(), -20, False, {})
                new_climber_x = self.climber_x
                new_climber_y = self.climber_y - 1
            case 1:
                if self.climber_x + 1 >= self.map_dim: return (self.get_state(), -20, False, {})
                new_climber_x = self.climber_x + 1
                new_climber_y = self.climber_y
            case 2:
                if self.climber_y + 1 >= self.map_dim: return (self.get_state(), -20, False, {})
                new_climber_x = self.climber_x
                new_climber_y = self.climber_y + 1
            case 3:
                if self.climber_x - 1 < 0: return (self.get_state(), -20, False, {})
                new_climber_x = self.climber_x - 1
                new_climber_y = self.climber_y

        rew = self.calc_rew(new_climber_x, new_climber_y)
        self.climber_x, self.climber_y = new_climber_x, new_climber_y
        self.steps += 1
        return (self.get_state(), rew, self.done, {})

    # Vraca trenutnu pozociju penjaca i sta vidi oko sebe
    def get_state(self) -> np.ndarray:
        relativ_x: float = (self.final_x - self.climber_x) / self.map_dim
        relativ_y: float = (self.final_y - self.climber_y) / self.map_dim

        if self.climber_y-1 >= 0:
            h0 = self.h_diff(self.climber_x, self.climber_y-1)
        else: h0 = 1

        if self.climber_x+1 < self.map_dim:
            h1 = self.h_diff(self.climber_x+1, self.climber_y)
        else: h1 = 1

        if self.climber_y+1 < self.map_dim:
            h2 = self.h_diff(self.climber_x, self.climber_y+1)
        else: h2 = 1

        if self.climber_x-1 >= 0:
            h3 = self.h_diff(self.climber_x-1, self.climber_y)
        else: h3 = 1

        return np.array([relativ_x, relativ_y, h0, h1, h2 ,h3], dtype=np.float32)

    # Restartuje simulaciju
    def reset(self) -> np.ndarray:
        self.steps = 0
        self.done = False
        #self.climber_x = rd.randint(0, self.map_dim-1)
        #self.climber_y = rd.randint(0, self.map_dim-1)
        #self.final_x = rd.randint(0, self.map_dim-1)
        #self.final_y = rd.randint(0, self.map_dim-1)

        self.climber_x, self.climber_y = 0, 0
        self.final_x, self.final_y = self.map_dim - 1, self.map_dim - 1
        return self.get_state()
