from pathlib import Path

from lightning.pytorch.callbacks import Callback


class ACDCTestLogger(Callback):
    def __init__(
        self,
        history_filename: str = "acdc_test_history.csv",
        best_filename: str = "acdc_best_result.txt",
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
        required = [
            "metrics/test_mean_dice",
            "metrics/test_dice_class_1",
            "metrics/test_dice_class_2",
            "metrics/test_dice_class_3",
        ]
        if not all(k in metrics for k in required):
            return

        run_dir = self._run_dir(trainer)
        run_dir.mkdir(parents=True, exist_ok=True)

        epoch = int(trainer.current_epoch)
        avg_dsc = float(metrics["metrics/test_mean_dice"].item())
        rv = float(metrics["metrics/test_dice_class_1"].item())
        myo = float(metrics["metrics/test_dice_class_2"].item())
        lv = float(metrics["metrics/test_dice_class_3"].item())

        history_path = run_dir / self.history_filename
        best_path = run_dir / self.best_filename

        if not history_path.exists():
            history_path.write_text("epoch,avg_dsc,rv,myo,lv\n", encoding="utf-8")

        with history_path.open("a", encoding="utf-8") as f:
            f.write(f"{epoch},{avg_dsc:.6f},{rv:.6f},{myo:.6f},{lv:.6f}\n")

        best_so_far = None
        if best_path.is_file():
            for raw_line in best_path.read_text(encoding="utf-8").splitlines():
                if raw_line.startswith("best_avg_dsc="):
                    best_so_far = float(raw_line.split("=", 1)[1].strip())

        if best_so_far is None or avg_dsc > best_so_far:
            ckpt_cb = getattr(trainer, "checkpoint_callback", None)
            best_model_path = getattr(ckpt_cb, "best_model_path", "") if ckpt_cb else ""
            last_model_path = getattr(ckpt_cb, "last_model_path", "") if ckpt_cb else ""

            with best_path.open("w", encoding="utf-8") as f:
                f.write(f"best_epoch={epoch}\n")
                f.write(f"best_avg_dsc={avg_dsc:.6f}\n")
                f.write(f"best_rv={rv:.6f}\n")
                f.write(f"best_myo={myo:.6f}\n")
                f.write(f"best_lv={lv:.6f}\n")
                if best_model_path:
                    f.write(f"best_model_path={best_model_path}\n")
                if last_model_path:
                    f.write(f"last_model_path={last_model_path}\n")

