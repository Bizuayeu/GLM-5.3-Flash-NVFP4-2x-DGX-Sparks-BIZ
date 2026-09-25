"""Two-rank stop/start transaction, independent of its process transport."""

# A transport raises these reasons when it loses a rank's reply; the switch
# then journals one of the two statuses, which only resume carries forward.
TRANSPORT_TIMEOUT = "transport-timeout"
SSH_UNAVAILABLE = "ssh-unavailable"
READINESS_UNCONFIRMED = "readiness-unconfirmed"
RECOVERY_READINESS_UNCONFIRMED = "recovery-readiness-unconfirmed"


class OperationFailure(RuntimeError):
    """Structured operational evidence without commands, credentials or payloads."""

    def __init__(self, action, rank, reason, exit_code=None):
        self.evidence = {
            "action": action,
            "rank": rank,
            "reason": reason,
            "exit_code": exit_code,
        }
        super().__init__(
            f"Rank {rank} {action}: {reason}"
            + (f" (exit {exit_code})" if exit_code is not None else "")
        )


def observation_lost(error):
    return (
        isinstance(error, OperationFailure)
        and error.evidence["action"] == "poll"
        and error.evidence["reason"] in (TRANSPORT_TIMEOUT, SSH_UNAVAILABLE)
    )


def failure_details(error):
    return {
        "error": type(error).__name__,
        **({"failure": error.evidence} if isinstance(error, OperationFailure) else {}),
    }


def finish(backend, report, *, save, config=None):
    """What a complete new pair gets, from a switch or from a resumed one."""
    # Only a complete pair gets its profile file: after a recovery the old file
    # still describes what runs. Like the ladder, a failed write is recorded
    # and never rolls the pair back.
    if config is not None:
        try:
            report["config"] = [
                {
                    "rank": row["rank"],
                    **backend.install(row["rank"], row["identity"], config),
                }
                for row in sorted(report["new"], key=lambda row: row["rank"])
            ]
        except Exception as error:  # noqa: BLE001 - keep the completed pair and record why
            report["config"] = {"failed": True, **failure_details(error)}
        save(report)
    # The pair is complete before the ladder runs: warmup is evidence for the
    # operator, not a readiness gate, so its failure never triggers recovery.
    try:
        report["warmup"] = backend.warmup(report["new"])
    except Exception as error:  # noqa: BLE001 - keep the completed pair and record why
        report["warmup"] = {"failed": True, **failure_details(error)}
    save(report)
    return report


def switch(backend, launch, *, save, config=None):
    """backend operations must address explicit owned launch identities.

    prepare checks static assets/fabric only. start performs the post-stop memory
    check. ready checks the new head API and both rank identities. This is a
    recoverable stop/start, not an atomic or zero-downtime deployment. config is
    the profile text both ranks write to the launch's configuration path once
    the new pair is complete; None leaves the remote files alone.
    """
    report = {"status": "preparing", "new": [], "stopped": [], "recovery": []}
    save(report)
    old = [backend.current(rank) for rank in (0, 1)]
    if (old[0] is None) != (old[1] is None):
        raise ValueError(
            "Only one old rank is running; preserve it and resolve the incomplete pair first"
        )
    first = [backend.prepare(rank, launch) for rank in (0, 1)]
    if first[0]["common"] != first[1]["common"]:
        raise ValueError(
            "Rank assets or common launch configuration differ; nothing stopped"
        )
    # Verify that any live profile is recoverable before making an interruption.
    old_assets = []
    for rank, previous in enumerate(old):
        if previous is not None:
            old_assets.append(backend.prepare(rank, previous["launch"], recovery=True))
    if old_assets and old_assets[0]["common"] != old_assets[1]["common"]:
        raise ValueError("Old ranks differ; a common recoverable profile is required")
    second = [backend.prepare(rank, launch) for rank in (0, 1)]
    old_again = [
        backend.prepare(rank, previous["launch"], recovery=True)
        for rank, previous in enumerate(old)
        if previous is not None
    ]
    if (
        first != second
        or old_assets != old_again
        or old != [backend.current(rank) for rank in (0, 1)]
    ):
        raise ValueError("Launch assets or running identities changed before stop")
    report["assets"] = first
    report["recovery_assets"] = old_assets
    report["status"] = "starting"
    save(report)
    try:
        for rank, previous in enumerate(old):
            if previous is not None:
                report["stop_pending"] = rank
                save(report)
                backend.stop(rank, previous)
                report.pop("stop_pending")
                report["stopped"].append(rank)
                save(report)
        for rank in (1, 0):
            # Reserve identity before start: a failed/ambiguous transport must
            # still allow cleanup of only the new attempt's container/process.
            new = backend.reserve(rank, launch)
            report["new"].append({"rank": rank, "identity": new})
            save(report)
            backend.start(rank, new)
        backend.ready(report["new"])
        report["status"] = "complete"
        save(report)
    except Exception as error:  # noqa: BLE001 - every transport failure requires owned cleanup
        report["status"] = "failed"
        report["error"] = type(error).__name__
        if isinstance(error, OperationFailure):
            report["failure"] = error.evidence
            if observation_lost(error):
                # A lost observation is not a failed model. Keep the already
                # owned, supervised attempts under their memory/deadline guards
                # and require identity-checked readiness resumption.
                report["status"] = READINESS_UNCONFIRMED
                save(report)
                raise RuntimeError(
                    "Readiness observation lost; resume this recorded attempt without replaying start"
                ) from None
        cleanup_failed = "stop_pending" in report
        for row in reversed(report["new"]):
            try:
                backend.stop(row["rank"], row["identity"])
            except Exception as stop_error:  # noqa: BLE001 - retain each cleanup failure
                cleanup_failed = True
                report.setdefault("cleanup_errors", []).append(
                    {"rank": row["rank"], **failure_details(stop_error)}
                )
        # Never compete with an unconfirmed new process for the same GPU/RAM.
        if not cleanup_failed:
            for rank in (1, 0):
                if rank in report["stopped"]:
                    try:
                        identity = backend.reserve(
                            rank, old[rank]["launch"], recovery=True
                        )
                        row = {"rank": rank, "identity": identity}
                        report["recovery"].append(row)
                        save(report)
                        backend.start(rank, identity)
                    except Exception as recovery_error:  # noqa: BLE001 - retain recovery outcome
                        report.setdefault("recovery_errors", []).append(
                            {"rank": rank, **failure_details(recovery_error)}
                        )
            if report["recovery"] and not report.get("recovery_errors"):
                try:
                    backend.ready(report["recovery"])
                    report["recovered"] = True
                except Exception as recovery_error:  # noqa: BLE001 - preserve failed recovery
                    if observation_lost(recovery_error):
                        report["status"] = RECOVERY_READINESS_UNCONFIRMED
                        report["recovery_observation_failure"] = recovery_error.evidence
                        save(report)
                        raise RuntimeError(
                            "Recovery readiness observation lost; resume the recorded recovering pair"
                        ) from None
                    report["recovery_errors"] = [failure_details(recovery_error)]
            if report.get("recovery_errors"):
                for row in report["recovery"]:
                    try:
                        backend.stop(row["rank"], row["identity"])
                    except Exception as stop_error:  # noqa: BLE001 - retain remaining owned cleanup failures
                        report.setdefault("cleanup_errors", []).append(
                            {"rank": row["rank"], **failure_details(stop_error)}
                        )
        save(report)
        raise RuntimeError(
            "Switch failed; inspect the saved cleanup and recovery result"
        ) from None
    return finish(backend, report, save=save, config=config)


def resume(backend, report, *, config=None, save=lambda report: None):
    recovering = report["status"] == RECOVERY_READINESS_UNCONFIRMED
    rows = report["recovery"] if recovering else report["new"]
    if report["status"] not in (
        READINESS_UNCONFIRMED,
        RECOVERY_READINESS_UNCONFIRMED,
    ) or {r["rank"] for r in rows} != {0, 1}:
        raise ValueError(
            "Only a recorded, unconfirmed two-rank readiness observation can resume"
        )
    assets = report["recovery_assets"] if recovering else report["assets"]
    for row in rows:
        current_rank = backend.current(row["rank"])
        identity = row["identity"]
        if current_rank is None or any(
            current_rank[key] != identity[key] for key in ("name", "fingerprint")
        ):
            raise ValueError("Running identity changed; readiness cannot resume")
        if backend.prepare(row["rank"], identity["launch"]) != assets[row["rank"]]:
            raise ValueError("Assets changed since the recorded launch")
    backend.ready(rows)
    if recovering:
        report["prior_recovery_observation_failure"] = report.pop(
            "recovery_observation_failure"
        )
        report["recovered"] = True
        report["status"] = (
            "failed"  # The candidate still failed; the old profile recovered.
        )
    else:
        report["prior_observation_failure"] = report.pop("failure")
        report.pop("error")
        report["status"] = "complete"
        save(report)
        # The switch that lost its observation never reached its last steps.
        finish(backend, report, save=save, config=config)
    return report
