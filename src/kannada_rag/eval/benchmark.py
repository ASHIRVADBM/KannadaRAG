"""The evaluation benchmark: schema, loading and validation.

Every question carries the provenance a reader needs in order to judge whether
the benchmark is sound: who wrote it, who validated it, whether the writer was
independent of system development, which passages contain the evidence, and
which version of the reference answer is in force.

Two design points are deliberate and both answer reviewer objections that the
earlier evaluation could not:

**Reference answer versioning.** ``reference`` is a list of accepted answers
with a version tag, not a single string. Scoring against multiple valid
phrasings is standard for QA benchmarks in morphologically rich languages, and
the version tag makes it impossible for two tables to be computed against
different reference sets without that being visible in the logs.

**Unanswerable questions.** ``answerable = false`` marks questions whose
answers are deliberately absent from the corpus. They cost nothing to write,
require no annotation to score, and are the only clean automatic evidence that
a grounded system declines rather than fabricates.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator, Literal, Sequence

__all__ = ["Question", "Benchmark", "load_benchmark"]

Difficulty = Literal["easy", "medium", "hard", "unanswerable"]


@dataclass
class Question:
    """One benchmark item."""

    question_id: str
    question: str
    difficulty: Difficulty
    references: list[str] = field(default_factory=list)
    reference_version: str = "v1"

    answerable: bool = True
    gold_chunk_ids: list[str] = field(default_factory=list)
    gold_pages: list[int] = field(default_factory=list)
    """Page-level gold evidence. More robust than chunk ids: re-chunking the
    corpus invalidates chunk ids but not page numbers, so retrieval metrics
    survive an ablation over chunk size."""

    source_file: str | None = None
    author: str | None = None
    """Who wrote the question."""
    author_independent: bool = False
    """True when the author took no part in developing or tuning the system.
    Recorded per question so the proportion can be reported honestly."""
    validated_by: list[str] = field(default_factory=list)
    """Domain or language experts who confirmed the question and reference."""
    notes: str = ""

    def __post_init__(self) -> None:
        if self.answerable and not self.references:
            raise ValueError(
                f"{self.question_id}: an answerable question needs at least one reference"
            )
        if self.difficulty == "unanswerable" and self.answerable:
            raise ValueError(
                f"{self.question_id}: difficulty 'unanswerable' conflicts with answerable=True"
            )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Benchmark:
    """A validated collection of questions with summary provenance."""

    questions: list[Question] = field(default_factory=list)
    name: str = "kannada-heritage-qa"
    version: str = "v1"

    def __len__(self) -> int:
        return len(self.questions)

    def __iter__(self) -> Iterator[Question]:
        return iter(self.questions)

    def by_difficulty(self, difficulty: str) -> list[Question]:
        return [q for q in self.questions if q.difficulty == difficulty]

    @property
    def answerable(self) -> list[Question]:
        return [q for q in self.questions if q.answerable]

    def provenance(self) -> dict:
        """Summary a reader needs to judge the benchmark's independence.

        ``independent_fraction`` is the number reviewers ask for when they ask
        whether question writers were independent of system development.
        """
        total = len(self.questions)
        if not total:
            return {}
        independent = sum(1 for q in self.questions if q.author_independent)
        validated = sum(1 for q in self.questions if q.validated_by)
        with_gold = sum(1 for q in self.questions if q.gold_pages or q.gold_chunk_ids)

        counts: dict[str, int] = {}
        for q in self.questions:
            counts[q.difficulty] = counts.get(q.difficulty, 0) + 1

        return {
            "name": self.name,
            "version": self.version,
            "n_questions": total,
            "by_difficulty": counts,
            "n_answerable": len(self.answerable),
            "n_unanswerable": total - len(self.answerable),
            "n_independent_authors": independent,
            "independent_fraction": round(independent / total, 3),
            "n_expert_validated": validated,
            "expert_validated_fraction": round(validated / total, 3),
            "n_with_gold_evidence": with_gold,
            "gold_evidence_fraction": round(with_gold / total, 3),
            "reference_versions": sorted({q.reference_version for q in self.questions}),
        }

    def validate(self) -> list[str]:
        """Return a list of problems rather than raising on the first one."""
        problems: list[str] = []
        seen: set[str] = set()

        for q in self.questions:
            if q.question_id in seen:
                problems.append(f"duplicate question_id: {q.question_id}")
            seen.add(q.question_id)

            if q.answerable and not (q.gold_pages or q.gold_chunk_ids):
                problems.append(
                    f"{q.question_id}: no gold evidence, retrieval metrics cannot be computed"
                )
            if q.answerable and not q.validated_by:
                problems.append(f"{q.question_id}: not expert-validated")

        versions = {q.reference_version for q in self.questions}
        if len(versions) > 1:
            problems.append(
                f"mixed reference versions in one benchmark: {sorted(versions)}"
            )

        return problems


def load_benchmark(path: str | Path) -> Benchmark:
    """Load a JSONL benchmark file.

    The first line may be a metadata object (``{"_meta": {...}}``); every
    other line is a question.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Benchmark not found: {path}")

    questions: list[Question] = []
    name, version = "kannada-heritage-qa", "v1"

    with path.open(encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            payload = json.loads(line)

            if "_meta" in payload:
                name = payload["_meta"].get("name", name)
                version = payload["_meta"].get("version", version)
                continue

            try:
                questions.append(Question(**payload))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from None

    return Benchmark(questions=questions, name=name, version=version)


def write_benchmark(benchmark: Benchmark, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write(
            json.dumps(
                {"_meta": {"name": benchmark.name, "version": benchmark.version}},
                ensure_ascii=False,
            )
            + "\n"
        )
        for question in benchmark.questions:
            fh.write(json.dumps(question.to_dict(), ensure_ascii=False) + "\n")
    return path
