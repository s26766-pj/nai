# eval.py

import os
import argparse
import time
import csv
from typing import Any, Dict, Optional, Tuple, List

import numpy as np

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

from wrappers import make_env, ActionSmoothing


def make_env_fn(seed: int, frame_stack: int = 4, render: bool = True):
    def _init():
        return make_env(render_mode="human" if render else None, seed=seed, frame_stack=frame_stack)
    return _init


def find_wrapper(root_env, wrapper_cls):
    """
    Szuka wrappera typu wrapper_cls w łańcuchu .env -> .env -> ...
    Działa dla zwykłego gym env (nie VecEnv).
    """
    e = root_env
    while e is not None:
        if isinstance(e, wrapper_cls):
            return e
        e = getattr(e, "env", None)
    return None


def safe_get_action(info: Dict[str, Any], key: str) -> Optional[np.ndarray]:
    v = info.get(key, None)
    if v is None:
        return None
    try:
        arr = np.asarray(v, dtype=np.float32).copy()
        return arr
    except Exception:
        return None


def classify_event(
    turn_norm: Optional[float],
    smooth_steer: Optional[float],
    smooth_gas: Optional[float],
    prev_smooth_steer: Optional[float],
    steer_rate_limit: Optional[float],
) -> List[str]:
    """
    Proste “flagowanie” typowych problemów:
    - ostry skręt + wysoki gaz
    - duże szarpnięcie kierownicą (jerk)
    """
    events: List[str] = []

    if turn_norm is not None and smooth_gas is not None:
        # bardzo ostry skręt + zbyt duży gaz = bączki / wyjazdy
        if turn_norm > 0.70 and smooth_gas > 0.55:
            events.append("RISK_TURN_FAST")
        elif turn_norm > 0.55 and smooth_gas > 0.70:
            events.append("RISK_MEDTURN_TOOFAST")

    # jerk kierownicy (zmiana skrętu na krok)
    if prev_smooth_steer is not None and smooth_steer is not None:
        jerk = abs(smooth_steer - prev_smooth_steer)
        # jeśli znamy limit steer_rate z wrappera, to porównajmy
        if steer_rate_limit is not None:
            if jerk > steer_rate_limit * 1.2:
                events.append("STEER_JERK")
        else:
            # fallback: absolutny próg
            if jerk > 0.20:
                events.append("STEER_JERK")

    return events


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True, help="Ścieżka do .zip")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--fps", type=float, default=60.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--deterministic", action="store_true")

    # debug print pierwszych kroków (jak miałeś)
    ap.add_argument("--debug_steps", type=int, default=0, help="Ile pierwszych kroków wypisać (0 = off)")

    # NOWE: logowanie do CSV
    ap.add_argument("--log_csv", type=str, default=None,
                    help="Ścieżka do CSV. Np: logs/eval_run.csv (None = nie zapisuj)")
    ap.add_argument("--log_every", type=int, default=1,
                    help="Zapisuj co N kroków (1 = każdy krok)")

    # NOWE: sterowanie renderem / maks kroków
    ap.add_argument("--no_render", action="store_true", help="Nie renderuj okna (szybciej)")
    ap.add_argument("--max_steps", type=int, default=0,
                    help="Maks kroków na epizod (0 = bez limitu, ale env i tak ma 1000)")

    args = ap.parse_args()

    if not os.path.exists(args.model):
        raise FileNotFoundError(args.model)

    # 1 env, identyczny pipeline jak trening
    env = DummyVecEnv([make_env_fn(args.seed, frame_stack=4, render=not args.no_render)])
    env = VecTransposeImage(env)

    # spróbujmy znaleźć ActionSmoothing i wyciągnąć jego parametry (steer_clip, steer_rate)
    base_env = env.venv.envs[0]  # oryginalny gym env (z wrapperami)
    smooth_wrap = find_wrapper(base_env, ActionSmoothing)
    steer_clip = getattr(smooth_wrap, "steer_clip", None) if smooth_wrap else None
    steer_rate = getattr(smooth_wrap, "steer_rate", None) if smooth_wrap else None

    if smooth_wrap:
        print(f"[INFO] Found ActionSmoothing: steer_clip={steer_clip} steer_rate={steer_rate} alpha={smooth_wrap.alpha}")
    else:
        print("[WARN] Nie znaleziono ActionSmoothing w łańcuchu wrapperów — turn_norm/jerk będą ograniczone.")

    model = SAC.load(args.model, env=env)

    delay = 1.0 / max(1.0, args.fps)

    # CSV writer
    csv_file = None
    csv_writer = None
    if args.log_csv:
        os.makedirs(os.path.dirname(args.log_csv) or ".", exist_ok=True)
        csv_file = open(args.log_csv, "w", newline="", encoding="utf-8")
        csv_writer = csv.DictWriter(csv_file, fieldnames=[
            "episode", "step", "reward", "ep_reward", "done",
            "raw_steer", "raw_gas",
            "smooth_steer", "smooth_gas",
            "env_steer", "env_gas", "env_brake",
            "turn_norm",
            "event"
        ])
        csv_writer.writeheader()
        print(f"[OK] Logging CSV -> {args.log_csv}")

    # global stats
    global_events_count: Dict[str, int] = {}
    global_ep_rewards: List[float] = []

    try:
        for ep in range(1, args.episodes + 1):
            obs = env.reset()
            done = False
            ep_reward = 0.0
            step = 0

            prev_smooth_steer: Optional[float] = None
            ep_events_count: Dict[str, int] = {}

            print(f"\n=== EPISODE {ep} ===")

            while not done:
                raw_action, _ = model.predict(obs, deterministic=args.deterministic)

                obs, rewards, dones, infos = env.step(raw_action)
                done = bool(dones[0])

                r = float(rewards[0])
                ep_reward += r

                info0 = infos[0] if infos and len(infos) > 0 else {}

                raw = raw_action[0]  # (2,)
                raw_steer, raw_gas = float(raw[0]), float(raw[1])

                smoothed = safe_get_action(info0, "action_2d_smoothed")  # (2,)
                applied2d = safe_get_action(info0, "action_2d")          # (2,)
                applied3d = safe_get_action(info0, "action_3d")          # (3,)

                smooth_steer = float(smoothed[0]) if smoothed is not None else None
                smooth_gas = float(smoothed[1]) if smoothed is not None else None

                env_steer = float(applied2d[0]) if applied2d is not None else None
                env_gas = float(applied2d[1]) if applied2d is not None else None

                env_brake = float(applied3d[2]) if applied3d is not None else None

                # turn_norm (0..1) względem steer_clip z wrappera
                turn_norm = None
                if steer_clip is not None and smooth_steer is not None:
                    turn_norm = abs(smooth_steer) / max(1e-6, float(steer_clip))

                # klasyfikuj eventy
                events = classify_event(
                    turn_norm=turn_norm,
                    smooth_steer=smooth_steer,
                    smooth_gas=smooth_gas,
                    prev_smooth_steer=prev_smooth_steer,
                    steer_rate_limit=float(steer_rate) if steer_rate is not None else None
                )

                if smooth_steer is not None:
                    prev_smooth_steer = smooth_steer

                # DEBUG print
                if args.debug_steps > 0 and step < args.debug_steps:
                    msg = f"[{step:03d}] r={r:+.3f} RAW steer={raw_steer:+.3f} gas={raw_gas:+.3f}"
                    if smooth_steer is not None:
                        msg += f" | SMOOTH steer={smooth_steer:+.3f} gas={smooth_gas:+.3f}"
                    if turn_norm is not None:
                        msg += f" | turn_norm={turn_norm:.2f}"
                    if events:
                        msg += f" | EVENTS={','.join(events)}"
                    print(msg)

                # CSV log
                if csv_writer and (step % max(1, args.log_every) == 0):
                    if not events:
                        events_to_write = [""]
                    else:
                        events_to_write = events

                    for ev in events_to_write:
                        csv_writer.writerow({
                            "episode": ep,
                            "step": step,
                            "reward": r,
                            "ep_reward": ep_reward,
                            "done": int(done),

                            "raw_steer": raw_steer,
                            "raw_gas": raw_gas,

                            "smooth_steer": smooth_steer if smooth_steer is not None else "",
                            "smooth_gas": smooth_gas if smooth_gas is not None else "",

                            "env_steer": env_steer if env_steer is not None else "",
                            "env_gas": env_gas if env_gas is not None else "",
                            "env_brake": env_brake if env_brake is not None else "",

                            "turn_norm": turn_norm if turn_norm is not None else "",
                            "event": ev
                        })

                # liczniki eventów
                for ev in events:
                    ep_events_count[ev] = ep_events_count.get(ev, 0) + 1
                    global_events_count[ev] = global_events_count.get(ev, 0) + 1

                step += 1

                if args.max_steps and step >= args.max_steps:
                    print(f"[WARN] Osiągnięto max_steps={args.max_steps} -> kończę epizod.")
                    break

                if not args.no_render:
                    time.sleep(delay)

            global_ep_rewards.append(ep_reward)
            print(f"[EP {ep}] reward={ep_reward:.2f} | events={ep_events_count}")

        # końcowe podsumowanie
        if global_ep_rewards:
            mean_r = float(np.mean(global_ep_rewards))
            std_r = float(np.std(global_ep_rewards))
            print("\n=== SUMMARY ===")
            print(f"Episodes: {len(global_ep_rewards)}")
            print(f"Reward mean±std: {mean_r:.2f} ± {std_r:.2f}")
            print(f"Events total: {global_events_count}")

    finally:
        env.close()
        if csv_file:
            csv_file.flush()
            csv_file.close()


if __name__ == "__main__":
    main()
