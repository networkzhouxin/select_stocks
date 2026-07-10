# -*- coding: utf-8 -*-
"""Research-budget governance for the cross-signal training program."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable


ALLOWED_STATUSES = {"adopted", "exhausted", "open", "blocked"}
REQUIRED_RECORD_FIELDS = ("Date", "Version", "Experiment")


@dataclass(frozen=True)
class FailedExperimentRecord:
    date: str
    version: str
    experiment: str
    why_failed: str


@dataclass(frozen=True)
class ResearchFamily:
    key: str
    label: str
    status: str
    max_new_experiments: int
    rationale: str
    planned_experiment: str | None = None


@dataclass(frozen=True)
class ResearchBudget:
    strategy_scope: str
    training_start: str
    training_end: str
    validation_tuning_forbidden: bool
    expected_failed_experiment_count: int
    max_total_open_experiments: int
    families: tuple[ResearchFamily, ...]


@dataclass(frozen=True)
class ExperimentGateDecision:
    allowed: bool
    reason: str


@dataclass(frozen=True)
class ResearchBudgetAudit:
    failed_experiment_count: int
    expected_failed_experiment_count: int
    duplicate_experiments: tuple[str, ...]
    errors: tuple[str, ...]


def parse_failed_experiments(text: str) -> tuple[FailedExperimentRecord, ...]:
    """Parse the append-only failed-experiment ledger's core fields."""
    blocks: list[dict[str, str]] = []
    current: dict[str, str] | None = None
    for raw_line in str(text).splitlines():
        line = raw_line.strip()
        if line.startswith("Date:"):
            date_value = line.split(":", 1)[1].strip()
            if not date_value:
                continue
            if current is not None:
                blocks.append(current)
            current = {"Date": date_value}
            continue
        if current is None or ":" not in line:
            continue
        key, value = line.split(":", 1)
        if key in {"Version", "Experiment"} or key.startswith("Why it "):
            current[key] = value.strip()
    if current is not None:
        blocks.append(current)

    records = []
    for index, block in enumerate(blocks, start=1):
        missing = [field for field in REQUIRED_RECORD_FIELDS if not block.get(field)]
        why = next(
            (value for key, value in block.items() if key.startswith("Why it ") and value),
            None,
        )
        if why is None:
            missing.append("Why it failed")
        if missing:
            raise ValueError(
                "failed experiment %d missing fields: %s"
                % (index, ", ".join(missing))
            )
        records.append(FailedExperimentRecord(
            date=block["Date"],
            version=block["Version"],
            experiment=block["Experiment"],
            why_failed=str(why),
        ))
    return tuple(records)


def load_research_budget(path: str | Path) -> ResearchBudget:
    """Load and validate the structured research budget."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    families = tuple(_load_family(item) for item in payload.get("families", ()))
    keys = [family.key for family in families]
    duplicates = sorted(key for key, count in Counter(keys).items() if count > 1)
    if duplicates:
        raise ValueError("duplicate research family: %s" % ", ".join(duplicates))

    open_families = [family for family in families if family.status == "open"]
    max_total_open = int(payload["max_total_open_experiments"])
    if sum(family.max_new_experiments for family in open_families) > max_total_open:
        raise ValueError("open family budgets exceed max_total_open_experiments")

    return ResearchBudget(
        strategy_scope=str(payload["strategy_scope"]),
        training_start=str(payload["training_window"]["start"]),
        training_end=str(payload["training_window"]["end"]),
        validation_tuning_forbidden=bool(payload["validation_tuning_forbidden"]),
        expected_failed_experiment_count=int(payload["expected_failed_experiment_count"]),
        max_total_open_experiments=max_total_open,
        families=families,
    )


def evaluate_experiment_request(
    budget: ResearchBudget,
    family_key: str,
    planned_variants: int,
) -> ExperimentGateDecision:
    """Check a pre-registered experiment against the remaining family budget."""
    families = {family.key: family for family in budget.families}
    family = families.get(str(family_key))
    if family is None:
        return ExperimentGateDecision(False, "unknown research family")
    if family.status != "open":
        return ExperimentGateDecision(
            False,
            "research family is %s: %s" % (family.status, family.rationale),
        )
    if int(planned_variants) != 1 or family.max_new_experiments != 1:
        return ExperimentGateDecision(
            False,
            "open families permit exactly one pre-registered variant",
        )
    return ExperimentGateDecision(True, "one pre-registered variant is available")


def audit_research_budget(
    failed_experiments_path: str | Path,
    budget_path: str | Path,
) -> ResearchBudgetAudit:
    """Reconcile the research ledger with the frozen structured budget."""
    records = parse_failed_experiments(
        Path(failed_experiments_path).read_text(encoding="utf-8")
    )
    budget = load_research_budget(budget_path)
    counts = Counter(record.experiment for record in records)
    duplicate_experiments = tuple(sorted(
        experiment for experiment, count in counts.items() if count > 1
    ))
    errors = []
    if len(records) != budget.expected_failed_experiment_count:
        errors.append(
            "failed experiment count is %d, expected %d"
            % (len(records), budget.expected_failed_experiment_count)
        )
    if budget.strategy_scope != "cross_signal_strategy":
        errors.append("strategy_scope must be cross_signal_strategy")
    if (budget.training_start, budget.training_end) != (
        "2019-01-01",
        "2021-12-31",
    ):
        errors.append("training window must remain 2019-01-01 to 2021-12-31")
    if not budget.validation_tuning_forbidden:
        errors.append("validation tuning must remain forbidden")
    if duplicate_experiments:
        errors.append("failed experiment ledger contains duplicate experiments")
    return ResearchBudgetAudit(
        failed_experiment_count=len(records),
        expected_failed_experiment_count=budget.expected_failed_experiment_count,
        duplicate_experiments=duplicate_experiments,
        errors=tuple(errors),
    )


def _load_family(payload: dict) -> ResearchFamily:
    status = str(payload["status"])
    if status not in ALLOWED_STATUSES:
        raise ValueError("unsupported research family status: %s" % status)
    max_new = int(payload["max_new_experiments"])
    planned = payload.get("planned_experiment")
    if status == "open":
        if max_new != 1:
            raise ValueError("open research family must have budget 1")
        if not planned:
            raise ValueError("open research family must have a planned_experiment")
    elif max_new != 0:
        raise ValueError("non-open research family must have budget 0")
    return ResearchFamily(
        key=str(payload["key"]),
        label=str(payload["label"]),
        status=status,
        max_new_experiments=max_new,
        rationale=str(payload["rationale"]),
        planned_experiment=str(planned) if planned else None,
    )
