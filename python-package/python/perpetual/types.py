import numpy as np
from typing_extensions import Self
from typing import Any, Dict, Iterable, Protocol, Set


class BoosterType(Protocol):
    monotone_constraints: Dict[int, int]
    terminate_missing_features: Set[int]
    number_of_trees: int
    base_score: float

    def fit(
        self,
        flat_data: np.ndarray,
        rows: int,
        cols: int,
        y: np.ndarray,
        budget: float,
        sample_weight: np.ndarray,
        parallel: bool = False,
    ):
        """Fit method"""

    def predict(
        self,
        flat_data: np.ndarray,
        rows: int,
        cols: int,
        parallel: bool = True,
    ) -> np.ndarray:
        """predict method"""

    def predict_proba(
        self,
        flat_data: np.ndarray,
        rows: int,
        cols: int,
        parallel: bool = True,
    ) -> np.ndarray:
        """predict probabilities method"""

    def predict_contributions(
        self,
        flat_data: np.ndarray,
        rows: int,
        cols: int,
        method: str,
        parallel: bool = True,
    ) -> np.ndarray:
        """method"""

    def value_partial_dependence(
        self,
        feature: int,
        value: float,
    ) -> float:
        """pass"""

    def calculate_feature_importance(
        self,
        method: str,
        normalize: bool,
    ) -> Dict[int, float]:
        """pass"""

    def text_dump(self) -> Iterable[str]:
        """pass"""

    @classmethod
    def load_booster(cls, path: str) -> Self:
        """pass"""

    def save_booster(self, path: str):
        """pass"""

    @classmethod
    def from_json(cls, json_str: str) -> Self:
        """pass"""

    def json_dump(self) -> str:
        """pass"""

    def get_params(self) -> Dict[str, Any]:
        """pass"""

    def insert_metadata(self, key: str, value: str) -> None:
        """pass"""

    def get_metadata(self, key: str) -> str:
        """pass"""


class MultiOutputBoosterType(Protocol):
    monotone_constraints: Dict[int, int]
    terminate_missing_features: Set[int]
    number_of_trees: Iterable[int]
    base_score: Iterable[float]

    def fit(
        self,
        flat_data: np.ndarray,
        rows: int,
        cols: int,
        y: np.ndarray,
        budget: float,
        sample_weight: np.ndarray,
        parallel: bool = False,
    ):
        """Fit method"""

    def predict(
        self,
        flat_data: np.ndarray,
        rows: int,
        cols: int,
        parallel: bool = True,
    ) -> np.ndarray:
        """predict method"""

    def predict_proba(
        self,
        flat_data: np.ndarray,
        rows: int,
        cols: int,
        parallel: bool = True,
    ) -> np.ndarray:
        """predict probabilities method"""

    @classmethod
    def load_booster(cls, path: str) -> Self:
        """pass"""

    def save_booster(self, path: str):
        """pass"""

    @classmethod
    def from_json(cls, json_str: str) -> Self:
        """pass"""

    def json_dump(self) -> str:
        """pass"""

    def get_params(self) -> Dict[str, Any]:
        """pass"""

    def insert_metadata(self, key: str, value: str) -> None:
        """pass"""

    def get_metadata(self, key: str) -> str:
        """pass"""
