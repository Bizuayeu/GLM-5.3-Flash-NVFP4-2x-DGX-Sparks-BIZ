"""Two-rank stop/start transaction, independent of its process transport."""


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


def switch(backend, launch, *, save):
    """backend operations must address explicit owned launch identities.

    prepare checks static assets/fabric only. start performs the post-stop memory
    check. ready checks the new head API and both rank identities. This is a
    recoverable stop/start, not an atomic or zero-downtime deployment.
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
            old_assets.append(backend.prepare(rank, previous["launch"]))
    if old_assets and old_assets[0]["common"] != old_assets[1]["common"]:
        raise ValueError("Old ranks differ; a common recoverable profile is required")
    second = [backend.prepare(rank, launch) for rank in (0, 1)]
    old_again = [
        backend.prepare(rank, previous["launch"])
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
        return report
    except Exception as error:  # noqa: BLE001 - every transport failure requires owned cleanup
        report["status"] = "failed"
        report["error"] = type(error).__name__
        if isinstance(error, OperationFailure):
            report["failure"] = error.evidence
            if error.evidence["action"] == "poll" and error.evidence["reason"] in (
                "transport-timeout",
                "ssh-unavailable",
            ):
                # A lost observation is not a failed model. Keep the already
                # owned, supervised attempts under their memory/deadline guards
                # and require identity-checked readiness resumption.
                report["status"] = "readiness-unconfirmed"
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
                    {"rank": row["rank"], "error": type(stop_error).__name__}
                )
        # Never compete with an unconfirmed new process for the same GPU/RAM.
        if not cleanup_failed:
            for rank in (1, 0):
                if rank in report["stopped"]:
                    try:
                        identity = backend.reserve(rank, old[rank]["launch"])
                        row = {"rank": rank, "identity": identity}
                        report["recovery"].append(row)
                        save(report)
                        backend.start(rank, identity)
                    except Exception as recovery_error:  # noqa: BLE001 - retain recovery outcome
                        report.setdefault("recovery_errors", []).append(
                            {"rank": rank, "error": type(recovery_error).__name__}
                        )
            if report["recovery"] and not report.get("recovery_errors"):
                try:
                    backend.ready(report["recovery"])
                    report["recovered"] = True
                except Exception as recovery_error:  # noqa: BLE001 - preserve failed recovery
                    report["recovery_errors"] = [
                        {"error": type(recovery_error).__name__}
                    ]
            if report.get("recovery_errors"):
                for row in report["recovery"]:
                    try:
                        backend.stop(row["rank"], row["identity"])
                    except Exception as stop_error:  # noqa: BLE001 - retain remaining owned cleanup failures
                        report.setdefault("cleanup_errors", []).append(
                            {"rank": row["rank"], "error": type(stop_error).__name__}
                        )
        save(report)
        raise RuntimeError(
            "Switch failed; inspect the saved cleanup and recovery result"
        ) from None
