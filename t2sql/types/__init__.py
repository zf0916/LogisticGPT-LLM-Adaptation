from dataclasses import dataclass
from typing import Any


@dataclass
class Document:
    id: str | None
    question: str | None
    document: str
    metadata: dict[Any] | None


@dataclass
class Example:
    id: str | None
    question: str
    additional_questions: str
    sql: str
    metadata: dict[Any] | None


@dataclass
class TrainingPlanItem:
    item_type: str
    item_group: str
    item_name: str
    item_value: str

    def __str__(self):
        if self.item_type == self.ITEM_TYPE_SQL:
            return f"Train on SQL: {self.item_group} {self.item_name}"
        elif self.item_type == self.ITEM_TYPE_DDL:
            return f"Train on DDL: {self.item_group} {self.item_name}"
        elif self.item_type == self.ITEM_TYPE_IS:
            return f"Train on Information Schema: {self.item_group} {self.item_name}"

    ITEM_TYPE_SQL = "sql"
    ITEM_TYPE_DDL = "ddl"
    ITEM_TYPE_IS = "is"


class TrainingPlan:
    _plan: list[TrainingPlanItem]

    def __init__(self, plan: list[TrainingPlanItem]):
        self._plan = plan

    def __str__(self):
        return "\n".join(self.get_summary())

    def __repr__(self):
        return self.__str__()

    def get_summary(self) -> list[str]:
        return [f"{item}" for item in self._plan]

    def remove_item(self, item: str):
        for plan_item in self._plan:
            if str(plan_item) == item:
                self._plan.remove(plan_item)
                break
