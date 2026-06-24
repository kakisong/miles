"""
trackio tracking backend for miles.

trackio (https://github.com/gradio-app/trackio) is a lightweight, wandb-API
compatible, local-first experiment tracker from HuggingFace.

Key differences from wandb that shape this backend:
  - A run is identified by ``project + name`` (no opaque run id), so we propagate
    the resolved ``name`` to secondary ranks via ``args.trackio_run_name``.
  - ``trackio.init`` has no ``dir`` parameter -- the local SQLite location is
    controlled by the ``TRACKIO_DIR`` env var instead.
  - There is no ``define_metric``; we log the ``*/step`` counters as plain
    scalars and let the dashboard pick the x-axis (mirrors WandbBackend.log).
  - There is no native distributed "shared" run. Coalescing the driver / train
    actor / rollout_manager processes into one run only works against a remote
    server (``server_url`` / ``space_id``). For the pure-local case we log from
    the primary rank only; secondary ranks no-op (see ``init_trackio``).
"""

from __future__ import annotations

import logging
import os
from typing import Any

from .wandb_utils import _compute_config_for_logging

logger = logging.getLogger(__name__)

# Emit the "local mode does not coalesce across processes" warning only once per
# process (a secondary rank may call init multiple times across a job).
_warned_local_secondary = False


def _resolve_identity(args) -> tuple[str | None, str | None, str | None]:
    # trackio identifies a run by project + name; fall back to the wandb naming
    # so a single --wandb-* config drives both backends.
    project = args.trackio_project or args.wandb_project
    name = args.trackio_run_name or args.wandb_group
    group = args.trackio_group or args.wandb_group
    return project, name, group


def _remote_kwargs(args) -> dict[str, str]:
    # Only a remote backend (self-hosted server or HF Space) can safely coalesce
    # multi-process / multi-node logging into one run. Env vars act as fallback.
    kwargs: dict[str, str] = {}
    server_url = args.trackio_server_url or os.environ.get("TRACKIO_SERVER_URL")
    space_id = args.trackio_space_id or os.environ.get("TRACKIO_SPACE_ID")
    if server_url:
        kwargs["server_url"] = server_url
    if space_id:
        kwargs["space_id"] = space_id
    return kwargs


def init_trackio(args, *, primary: bool = True, **kwargs) -> bool:
    """Initialise trackio for this process.

    Returns whether this process should log (the backend stores this and no-ops
    log/finish when False).
    """
    global _warned_local_secondary

    if not args.use_trackio:
        args.trackio_run_name = None
        return False

    import trackio

    project, name, group = _resolve_identity(args)
    remote = _remote_kwargs(args)

    # trackio.init has no dir param; the local SQLite path is set via TRACKIO_DIR.
    # Only on the primary rank in local mode: secondary ranks no-op in local mode,
    # and a remote server/space ignores TRACKIO_DIR -- creating the dir on other
    # nodes risks a PermissionError/FileNotFoundError.
    if primary and not remote and args.trackio_dir and not os.environ.get("TRACKIO_DIR"):
        os.makedirs(args.trackio_dir, exist_ok=True)
        os.environ["TRACKIO_DIR"] = args.trackio_dir
        logger.info("trackio local logs will be stored in: %s", args.trackio_dir)

    if primary:
        run = trackio.init(
            project=project,
            name=name,
            group=group,
            config=_compute_config_for_logging(args),
            resume="allow",
            **remote,
        )
        # Propagate the resolved run name so secondary ranks attach to the same
        # run (rides along with args, like wandb_run_id). Capture the name trackio
        # actually used, in case it auto-generated one (name was None).
        name = getattr(run, "name", None) or name
        args.trackio_run_name = name
        logger.info(
            "trackio run started: project=%s name=%s (%s)",
            project,
            name,
            "remote" if remote else "local",
        )
        return True

    # Secondary rank.
    if remote:
        trackio.init(
            project=project,
            name=args.trackio_run_name or name,
            group=group,
            resume="allow",
            **remote,
        )
        return True

    # Local-only: multiple processes writing one SQLite db (esp. across nodes) is
    # unsafe, so only the primary logs. Warn once so the missing rollout/eval
    # metrics (produced in the rollout_manager process when not colocated) are
    # not silent.
    if not _warned_local_secondary:
        _warned_local_secondary = True
        logger.warning(
            "trackio is running in local mode without --trackio-server-url/"
            "--trackio-space-id; only the primary process logs. Metrics produced "
            "solely in secondary processes (e.g. rollout/eval from rollout_manager "
            "when not colocated) will be missing. Configure a remote server to "
            "coalesce all ranks into one run."
        )
    return False


def log_metrics(metrics: dict[str, Any], step: int | None = None) -> None:
    import trackio

    # Mirror WandbBackend.log: do not pass step. trackio has no define_metric, and
    # the */step counters differ per metric family; they are logged as plain
    # scalars so the dashboard can be set to use any of them as the x-axis.
    trackio.log(metrics)


def finish() -> None:
    import trackio

    trackio.finish()
