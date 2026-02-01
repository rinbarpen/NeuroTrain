from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from tqdm import tqdm


def format_progress_desc(
    prefix: str,
    step: int,
    total_steps: int,
    current_batch: int,
    total_batch: int,
    epoch: int,
    total_epoch: int,
) -> str:
    """Format progress bar description as prefix step/total_steps[current_batch/total_batch](epoch/total_epoch)."""
    return f"{prefix} {step}/{total_steps}[{current_batch}/{total_batch}]({epoch}/{total_epoch})"


class ProgressBar:
    """Wrap tqdm progress bar usage to provide a consistent interface."""

    def __init__(self, total: int, *, desc: str | None = None, log_interval: int = 1):
        self.log_interval = max(1, log_interval)
        self._pbar = tqdm(total=total, desc=desc)

    def __enter__(self) -> "ProgressBar":
        self._pbar.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._pbar.__exit__(exc_type, exc_value, traceback)

    def update(self, n: int = 1) -> None:
        self._pbar.update(n)

    def set_postfix(self, mapping: Mapping[str, Any]) -> None:
        self._pbar.set_postfix(dict(mapping))

    def set_description(self, desc: str) -> None:
        self._pbar.set_description(desc)

    @property
    def format_dict(self) -> dict[str, Any] | None:
        return getattr(self._pbar, "format_dict", None)

    def update_step(
        self,
        *,
        epoch: int,
        current_step: int,
        total_steps: int,
        loss_label: str,
        loss_value: float,
        global_step: int | None = None,
        force: bool = False,
        prefix: str | None = None,
        total_global_steps: int | None = None,
        total_epochs: int | None = None,
        metrics: Mapping[str, Any] | None = None,
    ) -> None:
        if not force and self.log_interval > 1 and current_step % self.log_interval != 0:
            return

        if prefix is not None and total_global_steps is not None and total_epochs is not None:
            current_n = self._pbar.n
            self.set_description(
                format_progress_desc(
                    prefix, current_n, total_global_steps,
                    current_step, total_steps, epoch, total_epochs,
                )
            )

        eta = None

        format_dict = self.format_dict
        if format_dict is not None:
            remaining_seconds = format_dict.get("remaining")
            if remaining_seconds is not None and remaining_seconds != float("inf"):
                eta = tqdm.format_interval(remaining_seconds)

        postfix: dict[str, Any] = {
            loss_label: loss_value,
        }

        if global_step is not None:
            postfix["global_step"] = global_step

        if eta is not None:
            postfix["eta"] = eta

        if metrics:
            for k, v in metrics.items():
                postfix[k] = f"{float(v):.4f}" if isinstance(v, (int, float)) else v

        self.set_postfix(postfix)

    def close(self) -> None:
        self._pbar.close()

