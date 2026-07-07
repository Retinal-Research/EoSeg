from pathlib import Path

from lightning.pytorch.callbacks import Callback


class MedicalValLogger(Callback):
    def __init__(
        self,
        history_filename: str = "val_metrics_history.txt",
        best_filename: str = "best_metric.txt",
        monitor: str = "metrics/val_dice",
    ) -> None:
        super().__init__()
        self.history_filename = history_filename
        self.best_filename = best_filename
        self.monitor = monitor

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
        if self.monitor not in metrics or "metrics/val_dice" not in metrics:
            return

        run_dir = self._run_dir(trainer)
        run_dir.mkdir(parents=True, exist_ok=True)

        epoch = int(trainer.current_epoch)
        val_dice = float(metrics["metrics/val_dice"].item())
        val_iou = (
            float(metrics["metrics/val_iou"].item())
            if "metrics/val_iou" in metrics
            else None
        )
        val_jaccard = (
            float(metrics["metrics/val_jaccard"].item())
            if "metrics/val_jaccard" in metrics
            else val_iou
        )
        val_recall = (
            float(metrics["metrics/val_recall"].item())
            if "metrics/val_recall" in metrics
            else None
        )
        val_precision = (
            float(metrics["metrics/val_precision"].item())
            if "metrics/val_precision" in metrics
            else None
        )
        val_acc = (
            float(metrics["metrics/val_acc"].item())
            if "metrics/val_acc" in metrics
            else None
        )
        val_f2 = (
            float(metrics["metrics/val_f2"].item())
            if "metrics/val_f2" in metrics
            else None
        )

        history_path = run_dir / self.history_filename
        best_path = run_dir / self.best_filename
        monitor_name = self.monitor.split("/")[-1]
        monitor_value = float(metrics[self.monitor].item())

        line = f"epoch={epoch} val_dice={val_dice:.6f}"
        if val_iou is not None:
            line += f" val_iou={val_iou:.6f}"
        if val_jaccard is not None:
            line += f" val_jaccard={val_jaccard:.6f}"
        if val_recall is not None:
            line += f" val_recall={val_recall:.6f}"
        if val_precision is not None:
            line += f" val_precision={val_precision:.6f}"
        if val_acc is not None:
            line += f" val_acc={val_acc:.6f}"
        if val_f2 is not None:
            line += f" val_f2={val_f2:.6f}"
        line += "\n"

        with history_path.open("a", encoding="utf-8") as f:
            f.write(line)

        best_so_far = None
        best_epoch = None
        if best_path.is_file():
            for raw_line in best_path.read_text(encoding="utf-8").splitlines():
                if raw_line.startswith("best_epoch="):
                    best_epoch = raw_line.split("=", 1)[1].strip()
                elif raw_line.startswith(f"best_{monitor_name}="):
                    best_so_far = float(raw_line.split("=", 1)[1].strip())

        if best_so_far is None or monitor_value > best_so_far:
            with best_path.open("w", encoding="utf-8") as f:
                f.write(f"best_epoch={epoch}\n")
                f.write(f"monitor={self.monitor}\n")
                f.write(f"best_{monitor_name}={monitor_value:.6f}\n")
                f.write(f"best_val_dice={val_dice:.6f}\n")
                if val_iou is not None:
                    f.write(f"best_val_iou={val_iou:.6f}\n")
                if val_jaccard is not None:
                    f.write(f"best_val_jaccard={val_jaccard:.6f}\n")
                if val_recall is not None:
                    f.write(f"best_val_recall={val_recall:.6f}\n")
                if val_precision is not None:
                    f.write(f"best_val_precision={val_precision:.6f}\n")
                if val_acc is not None:
                    f.write(f"best_val_acc={val_acc:.6f}\n")
                if val_f2 is not None:
                    f.write(f"best_val_f2={val_f2:.6f}\n")
                f.write(f"last_update_epoch={epoch}\n")
