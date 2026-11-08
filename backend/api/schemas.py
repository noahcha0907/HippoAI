"""Response and request shapes shared with the front end.

Every response carries `mock`. While it is true, the numbers are placeholders
and must not be shown as results.
"""
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class Stage(str, Enum):
    intake = "intake"
    baseline = "baseline"
    locate = "locate"
    patch = "patch"
    validate = "validate"
    sealed_test = "sealed_test"
    report = "report"
    awaiting_decision = "awaiting_decision"
    approved = "approved"
    rejected = "rejected"
    failed = "failed"


STAGE_ORDER = [
    Stage.intake,
    Stage.baseline,
    Stage.locate,
    Stage.patch,
    Stage.validate,
    Stage.sealed_test,
    Stage.report,
    Stage.awaiting_decision,
]


class ModelInfo(BaseModel):
    id: str
    label: str
    damage_type: Literal["compression_targeted", "compression_global", "training_error", "control"]
    description: str
    base_model: str
    num_parameters: int


class ModelList(BaseModel):
    mock: bool
    models: list[ModelInfo]


class Budget(BaseModel):
    max_minutes: int = Field(default=60, ge=1, le=240)
    max_forward_passes: int = Field(default=25, ge=0, le=100)


class RepairRequest(BaseModel):
    model_id: str
    complaint: str = Field(min_length=3, max_length=2000)
    budget: Budget = Budget()


class RepairCreated(BaseModel):
    mock: bool
    id: str
    stage: Stage


class LocatedRegion(BaseModel):
    tensor: str
    rationale: str
    confidence: Literal["low", "medium", "high"]


class DevMetrics(BaseModel):
    clinical_accuracy: float
    broken_clinical_accuracy: float | None = None
    random_control_accuracy: float | None = None
    collateral_loss: float | None = None
    random_control_collateral: float | None = None
    general_net_change: float | None = None
    medmcqa_accuracy: float | None = None


class Attempt(BaseModel):
    number: int
    summary: str
    tensors_touched: list[str]
    weights_changed: int
    weights_changed_pct: float
    dev_metrics: DevMetrics
    outcome: Literal["accepted", "rolled_back"]
    reason: str


class RepairStatus(BaseModel):
    mock: bool
    id: str
    model_id: str
    complaint: str
    stage: Stage
    elapsed_minutes: float
    forward_passes_used: int
    budget: Budget
    located_regions: list[LocatedRegion]
    attempts: list[Attempt]
    is_replay: bool = False
    error: str | None = None


class SealedMetrics(BaseModel):
    clinical_accuracy: float
    unsafe_answer_rate: float
    medmcqa_accuracy: float


class Claim(BaseModel):
    text: str
    verified: bool
    evidence: str


class ExampleAnswer(BaseModel):
    question: str
    broken_answer: str
    repaired_answer: str
    expected_answer: str


class RepairReport(BaseModel):
    mock: bool
    id: str
    outcome: Literal["repaired", "could_not_repair_safely", "nothing_to_fix"]
    cause: str
    broken: SealedMetrics
    repaired: SealedMetrics
    clean: SealedMetrics
    recovery: float | None
    recovery_ci95: tuple[float, float] | None
    collateral_loss: float
    weights_changed: int
    weights_changed_pct: float
    tensors_touched: list[str]
    same_format_as_broken: bool
    claims: list[Claim]
    remaining_failures: list[str]
    examples: list[ExampleAnswer]
    minutes: float
    forward_passes: int
    tokens: int
    disclaimer: str


class DecisionRequest(BaseModel):
    decision: Literal["approve", "reject"]
    note: str | None = Field(default=None, max_length=2000)


class DecisionResponse(BaseModel):
    mock: bool
    id: str
    stage: Stage


class GenerateRequest(BaseModel):
    model_version: Literal["broken", "repaired", "clean"]
    prompt: str = Field(min_length=1, max_length=4000)
    repair_id: str | None = Field(default=None, description="Required for 'repaired'")
    model_id: str | None = Field(default=None, description="Broken model to use; defaults to the repair's model")


class GenerateResponse(BaseModel):
    mock: bool
    model_version: str
    text: str


class BenchmarkRow(BaseModel):
    case: str
    method: Literal["fixer_repair", "direction_removal", "random_edit"]
    broken_accuracy: float
    repaired_accuracy: float
    clean_accuracy: float
    recovery: float | None
    collateral_loss: float
    weights_changed_pct: float
    minutes: float
    passed: bool


class Benchmark(BaseModel):
    mock: bool
    rows: list[BenchmarkRow]


class CompareRequest(BaseModel):
    repair_id: str
    question_ids: list[str] | None = Field(default=None, description="Omit to get showcase questions the repair fixed")
    count: int = Field(default=4, ge=1, le=20)


class CompareAnswer(BaseModel):
    letter: str
    text: str
    correct: bool
    unsafe: bool


class CompareItem(BaseModel):
    id: str
    category: str
    question: str
    options: dict[str, str]
    expected: str
    unsafe_options: list[str]
    answers: dict[Literal["broken", "repaired", "clean"], CompareAnswer]


class CompareResponse(BaseModel):
    mock: bool
    repair_id: str
    items: list[CompareItem]


class QuestionSummary(BaseModel):
    id: str
    category: str
    question: str


class QuestionList(BaseModel):
    mock: bool
    questions: list[QuestionSummary]


class TimelineEvent(BaseModel):
    seconds: float | None
    kind: str
    title: str
    detail: dict


class Timeline(BaseModel):
    mock: bool
    repair_id: str
    is_replay: bool
    events: list[TimelineEvent]


class AttributionComparison(BaseModel):
    mock: bool
    repair_id: str
    suite: str
    note: str
    models: dict[Literal["broken", "repaired", "clean"], dict]


class EditSummary(BaseModel):
    tensors_touched: list[str]
    weights_changed: int
    weights_changed_pct: float


class AttemptCounts(BaseModel):
    total: int
    accepted: int
    rolled_back: int


class BestAttempt(BaseModel):
    number: int
    outcome: str
    dev_accuracy: float | None
    broken_dev_accuracy: float | None
    random_control_accuracy: float | None
    reason: str


class RepairSummary(BaseModel):
    mock: bool
    repair_id: str
    model_id: str
    complaint: str
    outcome: str
    cause: str
    accuracy: dict[str, float]
    unsafe_answer_rate: dict[str, float]
    recovery: float | None
    recovery_ci95: tuple[float, float] | None
    sealed_questions_note: str
    edit: EditSummary
    attempts: AttemptCounts
    best_attempt: BestAttempt | None
    minutes: float
    tokens: int
    is_replay: bool
    gate_note: str
    disclaimer: str
