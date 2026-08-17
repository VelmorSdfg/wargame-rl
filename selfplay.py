"""Противник-политика для self-play: пул снапшотов на диске.

Почему пул, а не «последняя копия себя»: против одной свежей копии политики скатываются в циклы
(камень-ножницы-бумага) — агент затачивается под текущую версию соперника, тот под него, и обе
ходят по кругу без роста силы. Пул прошлых версий заставляет быть сильным против РАСПРЕДЕЛЕНИЯ
соперников, а не против одного.

Объект живёт ВНУТРИ воркера (создаётся в factory, которая исполняется в подпроцессе), поэтому
никакого IPC не нужно: колбэк обучения просто пишет новые .zip в папку, воркеры их подхватывают.
Пока папка пуста — __call__ возвращает None, и среда сама падает обратно на скриптового бота.
"""
import glob
import os
import random


class PoolOpponent:
    def __init__(self, snap_dir, rescan_every=20, algo="ppo"):
        """rescan_every: раз во сколько эпизодов перечитывать папку и переcэмплировать соперника.
        Слишком часто — лишние загрузки с диска; слишком редко — воркер долго играет против одной
        старой версии."""
        self.snap_dir = snap_dir
        self.rescan_every = max(1, int(rescan_every))
        self.algo = algo
        self._model = None
        self._calls = 0
        self._episodes = 0

    def _load_class(self):
        if self.algo == "lstm":
            from sb3_contrib import RecurrentPPO
            return RecurrentPPO
        from stable_baselines3 import PPO
        return PPO

    def _maybe_resample(self):
        paths = sorted(glob.glob(os.path.join(self.snap_dir, "snap_*.zip")))
        if not paths:
            self._model = None
            return
        pick = random.choice(paths)
        try:
            self._model = self._load_class().load(pick, device="cpu")
        except Exception:
            self._model = None      # снапшот мог быть недописан в момент чтения — просто пропускаем

    def new_episode(self):
        self._episodes += 1
        if self._model is None or self._episodes % self.rescan_every == 0:
            self._maybe_resample()

    def __call__(self, obs):
        """obs уже построен ЗЕРКАЛЬНО (см. WarGameEnv._obs(side=1)), поэтому политика,
        обученная за синих, применима напрямую. Возврат None = «противника нет»."""
        if self._model is None:
            self._maybe_resample()
            if self._model is None:
                return None
        action, _ = self._model.predict(obs, deterministic=False)
        return action
