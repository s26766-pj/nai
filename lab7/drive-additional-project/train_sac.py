#train_sac.py

import os
import argparse
import signal
from datetime import datetime

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

from wrappers import make_env


def make_env_fn(seed: int, render_mode=None, frame_stack: int = 4):
    """Factory dla DummyVecEnv."""
    def _init():
        return make_env(render_mode=render_mode, seed=seed, frame_stack=frame_stack)
    return _init


def calls_from_timesteps(desired_timesteps: int, n_envs: int) -> int:
    """
    SB3 callbacki mają save_freq/eval_freq w "calls", nie w timesteps.
    Przy VecEnv: 1 call ~= n_envs timesteps.
    """
    desired_timesteps = int(desired_timesteps)
    n_envs = int(n_envs)
    return max(1, desired_timesteps // max(1, n_envs))


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--steps", type=int, default=500_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--logdir", type=str, default="runs")
    ap.add_argument("--modeldir", type=str, default="models")
    ap.add_argument("--n_envs", type=int, default=4)

    ap.add_argument("--resume", type=str, default=None,
                    help="Ścieżka do .zip (ckpt_*.zip / best_model.zip / emergency_model.zip)")
    ap.add_argument("--resume_weights_only", action="store_true",
                    help="Jeśli ustawione: wczyta tylko wagi policy do NOWEGO modelu (świeże optymalizery / ent_coef).")

    # częstotliwości w REALNYCH timesteps
    ap.add_argument("--ckpt_every", type=int, default=10_000,
                    help="Co ile REALNYCH timesteps robić checkpoint (domyślnie 10000)")
    ap.add_argument("--eval_every", type=int, default=10_000,
                    help="Co ile REALNYCH timesteps robić eval (domyślnie 10000)")
    ap.add_argument("--eval_episodes", type=int, default=3,
                    help="Ile epizodów w ewaluacji (domyślnie 3)")

    # SAC hiperparametry (ważne pod CPU)
    ap.add_argument("--learning_rate", type=float, default=3e-4)
    ap.add_argument("--buffer_size", type=int, default=200_000)   # było 50k
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--learning_starts", type=int, default=20_000)  # było 10k
    ap.add_argument("--tau", type=float, default=0.005)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--train_freq", type=int, default=1)
    ap.add_argument("--gradient_steps", type=int, default=1)

    # entropia: stała (bo chcesz przewidywalnie + bez hamulca)
    ap.add_argument("--ent_coef", type=float, default=0.2,
                    help="Stała entropia SAC (domyślnie 0.2)")

    args = ap.parse_args()

    os.makedirs(args.logdir, exist_ok=True)
    os.makedirs(args.modeldir, exist_ok=True)

    run_name = f"sac_carracing_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    tb_log = os.path.join(args.logdir, run_name)
    model_path = os.path.join(args.modeldir, run_name)
    os.makedirs(model_path, exist_ok=True)

    # --- ENV: trening (wiele envów równolegle)
    env = DummyVecEnv([make_env_fn(args.seed + i, None, 4) for i in range(args.n_envs)])
    env = VecTransposeImage(env)

    # --- ENV: ewaluacja (1 env)
    eval_env = DummyVecEnv([make_env_fn(args.seed + 123, None, 4)])
    eval_env = VecTransposeImage(eval_env)

    # debug: sprawdź action_space po nowych wrapperach (powinno być 2D)
    try:
        act_shape = env.action_space.shape
    except Exception:
        act_shape = None
    print(f"[INFO] action_space.shape={act_shape} (oczekiwane: (2,))")

    # --- Logger
    new_logger = configure(tb_log, ["stdout", "tensorboard"])

    # --- Budowa modelu (fresh albo resume)
    def build_fresh_model():
        m = SAC(
            policy="CnnPolicy",
            env=env,
            verbose=1,
            tensorboard_log=args.logdir,
            seed=args.seed,

            learning_rate=args.learning_rate,
            buffer_size=args.buffer_size,
            batch_size=args.batch_size,

            learning_starts=args.learning_starts,
            tau=args.tau,
            gamma=args.gamma,

            train_freq=args.train_freq,
            gradient_steps=args.gradient_steps,

            ent_coef=args.ent_coef,
        )
        m.set_logger(new_logger)
        return m

    if args.resume and not args.resume_weights_only:
        print(f"[INFO] Resuming FULL STATE from: {args.resume}")
        model = SAC.load(args.resume, env=env, tensorboard_log=args.logdir)
        model.set_logger(new_logger)
    elif args.resume and args.resume_weights_only:
        print(f"[INFO] Resuming WEIGHTS ONLY from: {args.resume}")
        old = SAC.load(args.resume, env=env, tensorboard_log=args.logdir)
        model = build_fresh_model()
        # upewnij się, że moduły są zbudowane
        model._setup_model()
        # wagi policy (actor+critic) przenosimy
        model.policy.load_state_dict(old.policy.state_dict(), strict=True)
    else:
        print("[INFO] Starting NEW training (fresh model).")
        model = build_fresh_model()

    # --- Awaryjny zapis na CTRL+C
    def save_emergency(sig, frame):
        print("\n[WARN] Przerwano (CTRL+C) — zapisuję awaryjnie emergency_model.zip ...")
        model.save(os.path.join(model_path, "emergency_model"))
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, save_emergency)

    # --- Przeliczenie freq (timesteps -> calls)
    save_freq_calls = calls_from_timesteps(args.ckpt_every, args.n_envs)
    eval_freq_calls = calls_from_timesteps(args.eval_every, args.n_envs)

    print(f"[INFO] n_envs={args.n_envs}")
    print(f"[INFO] ckpt_every={args.ckpt_every} timesteps -> save_freq={save_freq_calls} calls (~{save_freq_calls * args.n_envs} timesteps)")
    print(f"[INFO] eval_every={args.eval_every} timesteps -> eval_freq={eval_freq_calls} calls (~{eval_freq_calls * args.n_envs} timesteps)")
    print(f"[INFO] eval_episodes={args.eval_episodes}")

    print(f"[INFO] learning_rate={args.learning_rate}")
    print(f"[INFO] buffer_size={args.buffer_size}")
    print(f"[INFO] batch_size={args.batch_size}")
    print(f"[INFO] learning_starts={args.learning_starts} timesteps")
    print(f"[INFO] train_freq={args.train_freq} gradient_steps={args.gradient_steps}")
    print(f"[INFO] ent_coef={args.ent_coef}")

    # --- Checkpointy
    checkpoint_cb = CheckpointCallback(
        save_freq=save_freq_calls,
        save_path=model_path,
        name_prefix="ckpt"
    )

    # --- Ewaluacja i zapis best_model.zip
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=model_path,
        log_path=model_path,
        eval_freq=eval_freq_calls,
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        render=False
    )

    try:
        model.learn(
            total_timesteps=args.steps,
            callback=[checkpoint_cb, eval_cb],
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("[INFO] Zatrzymano trening. Masz emergency_model.zip w folderze runa.")
        return

    final_path = os.path.join(model_path, "final_model")
    model.save(final_path)
    print(f"\n[OK] Zapisano final model: {final_path}.zip")


if __name__ == "__main__":
    main()
