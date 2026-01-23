# wrappers.py

import numpy as np
import gymnasium as gym
import cv2
from gymnasium import spaces
from gymnasium.wrappers import RecordEpisodeStatistics
from collections import deque


class CarRacingPreprocess(gym.ObservationWrapper):
    """
    Preprocessing obrazu:
    - grayscale (opcjonalnie)
    - resize do 84x84
    - uint8 0..255 (oszczędność RAM w replay buffer)
    """
    def __init__(self, env: gym.Env, out_size=(84, 84), grayscale=True):
        super().__init__(env)
        self.out_w, self.out_h = out_size[0], out_size[1]
        self.grayscale = grayscale
        c = 1 if grayscale else 3

        self.observation_space = spaces.Box(
            low=0, high=255, shape=(self.out_h, self.out_w, c), dtype=np.uint8
        )

    def observation(self, obs):
        img = obs  # (96,96,3) uint8 RGB
        if self.grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # (96,96)
            img = cv2.resize(img, (self.out_w, self.out_h), interpolation=cv2.INTER_AREA)
            img = img.astype(np.uint8)
            img = img[..., None]  # (84,84,1)
        else:
            img = cv2.resize(img, (self.out_w, self.out_h), interpolation=cv2.INTER_AREA)
            img = img.astype(np.uint8)  # (84,84,3)
        return img


class FrameStack(gym.Wrapper):
    """
    Stack K ostatnich klatek po kanale.
    Wejście: (H,W,C)
    Wyjście: (H,W,C*K)
    """
    def __init__(self, env: gym.Env, k: int = 4):
        super().__init__(env)
        assert k >= 1
        self.k = k
        self.frames = deque(maxlen=k)

        h, w, c = env.observation_space.shape
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(h, w, c * k), dtype=np.uint8
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.frames.clear()
        for _ in range(self.k):
            self.frames.append(obs)
        return self._get_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        return np.concatenate(list(self.frames), axis=-1)


class NoBrakeAction(gym.Wrapper):
    """
    Akcja zewnętrzna: (steer, gas) 2D
    Do środka: (steer, gas, brake=0) 3D
    + zapis do info dla debug
    """
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0], dtype=np.float32),
            high=np.array([+1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

    def step(self, act2d):
        act2d = np.asarray(act2d, dtype=np.float32)

        steer = float(np.clip(act2d[0], -1.0, 1.0))
        gas   = float(np.clip(act2d[1],  0.0, 1.0))
        brake = 0.0

        act3d = np.array([steer, gas, brake], dtype=np.float32)

        obs, reward, terminated, truncated, info = self.env.step(act3d)

        info = dict(info)
        info["action_2d"] = np.array([steer, gas], dtype=np.float32)
        info["action_3d"] = act3d.copy()

        return obs, reward, terminated, truncated, info

class ActionSmoothing(gym.Wrapper):
    """
    Filtracja akcji 2D (steer, gas) + stabilizacja jazdy:
    - minimalny gaz (żeby nie pełzać / nie stać),
    - gaz zależny od skrętu (zwalnia na zakrętach),
    - LIMIT skrętu i LIMIT tempa zmiany skrętu (anti-bączek),
    - soft-start pierwsze N kroków (żeby nie wpadał w poślizg od startu).

    UWAGA: tutaj operujemy WYŁĄCZNIE na 2D.
    Brake jest doklejany niżej przez NoBrakeAction.
    """
    def __init__(
        self,
        env: gym.Env,
        alpha: float = 0.25,
        min_gas: float = 0.15,
        gas_scale: float = 0.75,
        steer_clip: float = 0.65,
        steer_rate: float = 0.08,
        soft_start_steps: int = 20,
        turn_slow_k: float = 0.50,
        turn_min_factor: float = 0.45
    ):
        super().__init__(env)
        self.alpha = float(alpha)
        self.min_gas = float(min_gas)
        self.gas_scale = float(gas_scale)

        self.steer_clip = float(steer_clip)
        self.steer_rate = float(steer_rate)

        self.soft_start_steps = int(soft_start_steps)
        self.turn_slow_k = float(turn_slow_k)
        self.turn_min_factor = float(turn_min_factor)

        self.prev = None
        self.t = 0

        # sanity-check: ten wrapper zakłada 2D
        if getattr(env.action_space, "shape", None) != (2,):
            print(f"[WARN] ActionSmoothing oczekuje action_space (2,), a jest {getattr(env.action_space,'shape',None)}")

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.t = 0
        # start spokojny: steer=0, gas=min_gas
        self.prev = np.array([0.0, self.min_gas], dtype=np.float32)
        return obs, info

    def step(self, action):
        self.t += 1
        action = np.asarray(action, dtype=np.float32).copy()

        steer = float(action[0])
        gas = float(action[1])

        # clamp wejścia
        steer = float(np.clip(steer, -1.0, 1.0))
        gas = float(np.clip(gas, 0.0, 1.0))

        # limit maks skrętu
        steer = float(np.clip(steer, -self.steer_clip, self.steer_clip))

        # limit tempa zmiany skrętu
        prev_steer = float(self.prev[0])
        d = float(np.clip(steer - prev_steer, -self.steer_rate, self.steer_rate))
        steer = prev_steer + d

        # bazowy gaz: min + skala
        gas = self.min_gas + self.gas_scale * gas
        gas = float(np.clip(gas, 0.0, 1.0))

        # zwalnianie na zakrętach
        turn = abs(steer) / max(1e-6, self.steer_clip)  # 0..1
        
        turn_factor = 1.0 - self.turn_slow_k * turn
        turn_factor = float(np.clip(turn_factor, self.turn_min_factor, 1.0))
        gas *= turn_factor
        
        # hard cap gazu przy mocnym skręcie (anty "za szybko w ostre")
        if turn > 0.70:
            gas = min(gas, 0.35)   # bardzo ostry skręt
        elif turn > 0.55:
            gas = min(gas, 0.50)   # ostry skręt
        elif turn > 0.35:
            gas = min(gas, 0.65)   # średni skręt
        
        gas = float(np.clip(gas, 0.0, 1.0))        

        # soft-start
        if self.t <= self.soft_start_steps:
            gas *= 0.7
            gas = float(np.clip(gas, 0.0, 1.0))

        a = np.array([steer, gas], dtype=np.float32)

        # smoothing
        smooth = self.alpha * self.prev + (1.0 - self.alpha) * a
        self.prev = smooth

        obs, reward, terminated, truncated, info = self.env.step(smooth)

        info = dict(info)
        info["action_2d_smoothed"] = smooth.copy()

        return obs, reward, terminated, truncated, info


def make_env(render_mode=None, seed: int = 0, frame_stack: int = 4):
    """
    CarRacing-v3 (Gymnasium) -> stats -> NoBrake(2D) -> preprocess -> frame stack -> smoothing(2D)
    """
    
    from stable_baselines3.common.monitor import Monitor
    
    env = gym.make("CarRacing-v3", render_mode=render_mode)

    env = Monitor(env)
    # statystyki epizodów (odpowiednik Monitor)
    #env = RecordEpisodeStatistics(env)

    # SEED
    env.reset(seed=seed)
    try:
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
    except Exception:
        pass

    # najpierw redukcja akcji do 2D (dokleja brake=0 pod spodem)
    env = NoBrakeAction(env)

    # obserwacje
    env = CarRacingPreprocess(env, out_size=(84, 84), grayscale=True)
    env = FrameStack(env, k=frame_stack)

    # na końcu smoothing 2D
    #env = ActionSmoothing(env, alpha=0.25, min_gas=0.15, gas_scale=0.75)
    env = ActionSmoothing(
        env,
        alpha=0.10,
        min_gas=0.05,
        gas_scale=0.90,
        steer_clip=0.90,
        #steer_rate=0.20,
        steer_rate=0.12,
        soft_start_steps=10,
        #turn_slow_k=0.25,
        turn_slow_k=0.35,
        #turn_min_factor=0.70
        turn_min_factor=0.60
    )

    return env
