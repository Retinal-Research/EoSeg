from pathlib import Path

from lightning.pytorch.callbacks import Callback


class ACDCValLogger(Callback):
    def __init__(
        self,
        history_filename: str = "val_metrics_history.txt",
        best_filename: str = "best_metric.txt",
    ) -> None:
        super().__init__()
        self.history_filename = history_filename
        self.best_filename = best_filename

    def _run_dir(self, trainer) -> Path:
        log_dir = getattr(trainer.logger, "log_dir", None)
        if log_dir is not None:
            return Path(log_dir)

        save_dir = getattr(trainer.logger, "save_dir", None)
        name = getattr(trainer.logger, "name", "default")
        version = getattr(trainer.logger, "version", "version_0")
        return Path(save_dir) / str(name) / str(version)

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        metrics = trainer.callback_metrics
        if "metrics/val_dice" in metrics:
            score_name = "metrics/val_dice"
            score_key = "val_dice"
        elif "metrics/test_mean_dice" in metrics:
            score_name = "metrics/test_mean_dice"
            score_key = "test_mean_dice"
        else:
            return

        run_dir = self._run_dir(trainer)
        run_dir.mkdir(parents=True, exist_ok=True)

        epoch = int(trainer.current_epoch)
        score = float(metrics[score_name].item())

        history_path = run_dir / self.history_filename
        best_path = run_dir / self.best_filename

        with history_path.open("a", encoding="utf-8") as f:
            f.write(f"epoch={epoch} {score_key}={score:.6f}\n")

        best_so_far = None
        if best_path.is_file():
            for raw_line in best_path.read_text(encoding="utf-8").splitlines():
                if raw_line.startswith(f"best_{score_key}="):
                    best_so_far = float(raw_line.split("=", 1)[1].strip())

        if best_so_far is None or score > best_so_far:
            with best_path.open("w", encoding="utf-8") as f:
                f.write(f"best_epoch={epoch}\n")
                f.write(f"best_{score_key}={score:.6f}\n")
                f.write(f"last_update_epoch={epoch}\n")

