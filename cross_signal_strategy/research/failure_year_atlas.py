# -*- coding: utf-8 -*-
"""Build an auditable training-only map of failed experiment years.

The module deliberately reads only the retained experiment ledger and a small
curated annotation file.  It never opens market-data roots, and it refuses to
infer a failed year when the ledger did not retain explicit annual evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence


DOCS_DIR = pathlib.Path(__file__).resolve().parents[1] / "docs"
FAILED_EXPERIMENTS_PATH = DOCS_DIR / "failed_experiments.md"
ANNOTATIONS_PATH = DOCS_DIR / "failure_year_fragility_annotations.json"
REPORT_PATH = DOCS_DIR / "failure_year_fragility_atlas.md"
TRAINING_YEARS = (2019, 2020, 2021)
MAINLINE_ANNUAL_RETURNS = {
    2019: 0.3584,
    2020: 0.4974,
    2021: 0.0846,
}


@dataclass(frozen=True)
class FailedExperimentRecord:
    record_id: str
    date: str
    version: str
    experiment: str
    hypothesis: str
    raw_text: str


@dataclass(frozen=True)
class FragilityAnnotation:
    record_id: str
    failed_years: tuple[int, ...]
    mechanism: str
    evidence: str
    interpretation: str


@dataclass(frozen=True)
class AnnotationValidation:
    unknown_record_ids: tuple[str, ...]
    duplicate_record_ids: tuple[str, ...]
    invalid_years: tuple[int, ...]
    missing_evidence: tuple[str, ...]
    record_mismatches: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not any((
            self.unknown_record_ids,
            self.duplicate_record_ids,
            self.invalid_years,
            self.missing_evidence,
            self.record_mismatches,
        ))


@dataclass(frozen=True)
class AnnotatedExperiment:
    record: FailedExperimentRecord
    annotation: FragilityAnnotation


@dataclass(frozen=True)
class FailureYearAtlas:
    total_experiments: int
    annotated_experiments: int
    unreported_annual_experiments: int
    failed_year_counts: Mapping[int, int]
    mechanism_counts: Mapping[str, int]
    mainline_annual_returns: Mapping[int, float]
    annotated_records: tuple[AnnotatedExperiment, ...]


def _field(block: str, name: str) -> str:
    match = re.search(rf"(?m)^{re.escape(name)}:\s*(.*)$", block)
    return match.group(1).strip() if match else ""


def _record_id(date: str, version: str, experiment: str) -> str:
    identity = "\n".join((date.strip(), version.strip(), experiment.strip()))
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]


def parse_failed_experiments(text: str) -> tuple[FailedExperimentRecord, ...]:
    """Parse real ledger entries while excluding the empty template."""

    starts = list(re.finditer(r"(?m)^Date:\s*(\d{4}-\d{2}-\d{2})\s*$", text))
    records = []
    for index, start in enumerate(starts):
        end = starts[index + 1].start() if index + 1 < len(starts) else len(text)
        block = text[start.start():end].strip()
        date = start.group(1)
        version = _field(block, "Version")
        experiment = _field(block, "Experiment")
        hypothesis = _field(block, "Hypothesis")
        if not version or not experiment:
            raise ValueError(f"ledger entry on {date} lacks Version or Experiment")
        records.append(FailedExperimentRecord(
            record_id=_record_id(date, version, experiment),
            date=date,
            version=version,
            experiment=experiment,
            hypothesis=hypothesis,
            raw_text=block,
        ))

    record_ids = [record.record_id for record in records]
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("failed-experiment ledger contains duplicate record identities")
    return tuple(records)


def load_annotations(path: pathlib.Path = ANNOTATIONS_PATH) -> tuple[FragilityAnnotation, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported failure-year annotation schema")
    return tuple(
        FragilityAnnotation(
            record_id=str(item["record_id"]),
            failed_years=tuple(int(year) for year in item["failed_years"]),
            mechanism=str(item["mechanism"]).strip(),
            evidence=str(item["evidence"]).strip(),
            interpretation=str(item.get("interpretation", "")).strip(),
        )
        for item in payload.get("annotations", [])
    )


def validate_annotations(
    records: Sequence[FailedExperimentRecord],
    annotations: Sequence[FragilityAnnotation],
) -> AnnotationValidation:
    records_by_id = {record.record_id: record for record in records}
    annotation_ids = [annotation.record_id for annotation in annotations]
    duplicates = sorted(
        record_id for record_id, count in Counter(annotation_ids).items() if count > 1
    )
    unknown = sorted(set(annotation_ids) - set(records_by_id))
    invalid_years = sorted({
        year
        for annotation in annotations
        for year in annotation.failed_years
        if year not in TRAINING_YEARS
    })
    missing_evidence = sorted(
        annotation.record_id
        for annotation in annotations
        if not annotation.evidence or not annotation.mechanism or not annotation.failed_years
    )
    mismatches = []
    for annotation in annotations:
        record = records_by_id.get(annotation.record_id)
        if record is None:
            continue
        years_in_record = {int(year) for year in re.findall(r"\b(2019|2020|2021)\b", record.raw_text)}
        if not set(annotation.failed_years) <= years_in_record:
            mismatches.append(annotation.record_id)

    return AnnotationValidation(
        unknown_record_ids=tuple(unknown),
        duplicate_record_ids=tuple(duplicates),
        invalid_years=tuple(invalid_years),
        missing_evidence=tuple(missing_evidence),
        record_mismatches=tuple(sorted(mismatches)),
    )


def build_failure_year_atlas(
    records: Sequence[FailedExperimentRecord],
    annotations: Sequence[FragilityAnnotation],
) -> FailureYearAtlas:
    validation = validate_annotations(records, annotations)
    if not validation.ok:
        raise ValueError(f"invalid failure-year annotations: {validation}")

    records_by_id = {record.record_id: record for record in records}
    annotated_records = tuple(
        AnnotatedExperiment(records_by_id[annotation.record_id], annotation)
        for annotation in annotations
    )
    year_counts = Counter({year: 0 for year in TRAINING_YEARS})
    mechanism_counts = Counter()
    for annotation in annotations:
        year_counts.update(annotation.failed_years)
        mechanism_counts[annotation.mechanism] += 1

    return FailureYearAtlas(
        total_experiments=len(records),
        annotated_experiments=len(annotations),
        unreported_annual_experiments=len(records) - len(annotations),
        failed_year_counts=dict(sorted(year_counts.items())),
        mechanism_counts=dict(sorted(mechanism_counts.items())),
        mainline_annual_returns=dict(MAINLINE_ANNUAL_RETURNS),
        annotated_records=annotated_records,
    )


MECHANISM_LABELS = {
    "parameter_instability": "参数变化的跨年不稳定",
    "regime_reversal": "市场状态反转",
    "sample_concentration": "样本不足或年度集中",
    "execution_inconsistency": "执行机制跨年不一致",
    "tail_execution": "少数极端成交尾部",
}


def _escape_table(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ").strip()


def format_failure_year_atlas(atlas: FailureYearAtlas) -> str:
    lines = [
        "# 上穿下穿策略失败年份脆弱性地图",
        "",
        "## 口径与边界",
        "",
        "- 数据边界：只读取 `failed_experiments.md` 和人工审计标注；未读取任何行情目录。",
        "- 只统计台账中明确记录的逐年反例；没有逐年证据的实验保留为“未报告”，不猜测失败年份。",
        "- 本报告只做研究治理和根因归类，不是新指标实验，不得据此修改策略。",
        "- 未读取验证期行情，也不得利用验证期结果重新选择规则。",
        "",
        "## 正式主线年度背景",
        "",
        "| 年份 | `cross-v0.3.2` 本地训练年收益 |",
        "| --- | ---: |",
    ]
    for year, annual_return in atlas.mainline_annual_returns.items():
        lines.append(f"| {year} | {annual_return:.2%} |")

    lines.extend([
        "",
        "2020 年不是正式主线弱年；它是正式主线三个训练年度中收益最高的一年。",
        "2020 只在固定分钟限价实验中成为失败年，不能把执行层尾部问题误判为信号层失效。",
        "",
        "## 年度反例覆盖",
        "",
        f"- 台账实验总数：{atlas.total_experiments}",
        f"- 具有明确逐年反例并完成审计标注：{atlas.annotated_experiments}",
        f"- 未保留足够逐年证据、因此不归因：{atlas.unreported_annual_experiments}",
        "",
        "| 年份 | 明确年度反例次数 |",
        "| --- | ---: |",
    ])
    for year, count in atlas.failed_year_counts.items():
        lines.append(f"| {year} | {count} |")

    lines.extend([
        "",
        "同一个实验可以在多个年份触发门槛失败，因此次数不是实验数量，也不是年度优劣评分。",
        "从现有明确证据看，2021 出现反例最频繁，主要表现为低广度反转、波动扩张、",
        "资金份额方向和延迟执行等关系相对 2019-2020 发生反转。",
        "",
        "## 根因类型",
        "",
        "| 根因 | 实验数 |",
        "| --- | ---: |",
    ])
    for mechanism, count in atlas.mechanism_counts.items():
        lines.append(f"| {MECHANISM_LABELS.get(mechanism, mechanism)} | {count} |")

    lines.extend([
        "",
        "## 明确逐年反例明细",
        "",
        "| 日期 | 版本/观察 | 未通过年份 | 根因 | 明确证据 |",
        "| --- | --- | --- | --- | --- |",
    ])
    for item in atlas.annotated_records:
        years = ", ".join(str(year) for year in item.annotation.failed_years)
        lines.append(
            "| {date} | {version} | {years} | {mechanism} | {evidence} |".format(
                date=item.record.date,
                version=_escape_table(item.record.version),
                years=years,
                mechanism=MECHANISM_LABELS.get(
                    item.annotation.mechanism, item.annotation.mechanism
                ),
                evidence=_escape_table(item.annotation.evidence),
            )
        )

    lines.extend([
        "",
        "## 结论与下一研究方向",
        "",
        "1. 正式主线保持 `cross-v0.3.2`，不针对某个失败年份添加年份、ETF 或阈值特例。",
        "2. 2021 的反例集中说明多数传统确认指标具有状态依赖性，不说明应该用 2021 重新调参。",
        "3. 2020 分钟限价失败属于少数追价尾部；继续搜索等待时间、限价偏移或 QDII 特例会事后拟合。",
        "4. 同一 OHLC 派生指标家族已耗尽。下一项仍具有独立性的方向只有预注册的",
        "   QDII 底层指数方向；在官方最终值和历史 `available_at` 证据补齐前保持阻塞。",
        "5. 后续执行研究只能使用单独批准的前瞻影子样本，并先冻结机制和门槛；不得回头继续挖",
        "   2019-2021，也不得查看验证期来决定参数。",
        "",
    ])
    return "\n".join(lines)


def write_failure_year_atlas(
    output_path: pathlib.Path = REPORT_PATH,
    ledger_path: pathlib.Path = FAILED_EXPERIMENTS_PATH,
    annotations_path: pathlib.Path = ANNOTATIONS_PATH,
) -> FailureYearAtlas:
    records = parse_failed_experiments(ledger_path.read_text(encoding="utf-8"))
    atlas = build_failure_year_atlas(records, load_annotations(annotations_path))
    output_path.write_text(format_failure_year_atlas(atlas), encoding="utf-8")
    return atlas


def _list_record_ids(records: Iterable[FailedExperimentRecord]) -> str:
    return "\n".join(
        f"{record.record_id}\t{record.date}\t{record.version}"
        for record in records
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-records", action="store_true")
    parser.add_argument("--output", type=pathlib.Path, default=REPORT_PATH)
    args = parser.parse_args(argv)

    records = parse_failed_experiments(FAILED_EXPERIMENTS_PATH.read_text(encoding="utf-8"))
    if args.list_records:
        print(_list_record_ids(records))
        return 0

    atlas = build_failure_year_atlas(records, load_annotations(ANNOTATIONS_PATH))
    args.output.write_text(format_failure_year_atlas(atlas), encoding="utf-8")
    print(
        "failure-year atlas written: "
        f"experiments={atlas.total_experiments} "
        f"annotated={atlas.annotated_experiments} "
        f"output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
