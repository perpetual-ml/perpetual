"""Protocol (structural typing) definitions for the Rust-backed booster types."""

from typing import Any, Dict, Iterable, List, Protocol, Set

import numpy as np
from typing_extensions import Self


class BoosterType(Protocol):
    """Protocol for the single-output booster interface exposed by the Rust crate."""

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
        """Fit the booster on flat (column-major) data."""

    def predict(
        self,
        flat_data: np.ndarray,
        rows: int,
        cols: int,
        parallel: bool = True,
    ) -> np.ndarray:
        """Return raw predictions."""

    def predict_proba(
        self,
        flat_data: np.ndarray,
        rows: int,
        cols: int,
        parallel: bool = True,
    ) -> np.ndarray:
        """Return class probabilities (sigmoid of log-odds)."""

    def predict_contributions(
        self,
        flat_data: np.ndarray,
        rows: int,
        cols: int,
        method: str,
        parallel: bool = True,
    ) -> np.ndarray:
        """Return per-feature contribution values."""

    def value_partial_dependence(
        self,
        feature: int,
        value: float,
    ) -> float:
        """Return partial dependence for a single feature value."""

    def calculate_feature_importance(
        self,
        method: str,
        normalize: bool,
    ) -> Dict[int, float]:
        """Return feature importance scores."""

    def text_dump(self) -> List[str]:
        """Return a text representation of each tree."""

    @classmethod
    def load_booster(cls, path: str) -> Self:
        """Load a booster from a file."""

    def save_booster(self, path: str):
        """Save the booster to a file."""

    @classmethod
    def from_json(cls, json_str: str) -> Self:
        """Deserialize a booster from a JSON string."""

    def json_dump(self) -> str:
        """Serialize the booster to a JSON string."""

    def get_params(self) -> Dict[str, Any]:
        """Return the booster's configuration parameters."""

    def insert_metadata(self, key: str, value: str) -> None:
        """Insert a key-value pair into the booster's metadata."""

    def get_metadata(self, key: str) -> str:
        """Retrieve a metadata value by key."""


class MultiOutputBoosterType(Protocol):
    """Protocol for the multi-output booster interface exposed by the Rust crate."""

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
        """Fit the multi-output booster on flat (column-major) data."""

    def predict(
        self,
        flat_data: np.ndarray,
        rows: int,
        cols: int,
        parallel: bool = True,
    ) -> np.ndarray:
        """Return raw predictions for each output."""

    def predict_proba(
        self,
        flat_data: np.ndarray,
        rows: int,
        cols: int,
        parallel: bool = True,
    ) -> np.ndarray:
        """Return class probabilities for each output."""

    def predict_contributions(
        self,
        flat_data: np.ndarray,
        rows: int,
        cols: int,
        method: str,
        parallel: bool = True,
    ) -> np.ndarray:
        """Return per-feature contribution values."""

    def value_partial_dependence(
        self,
        feature: int,
        value: float,
    ) -> float:
        """Return partial dependence for a single feature value."""

    def calculate_feature_importance(
        self,
        method: str,
        normalize: bool,
    ) -> Dict[int, float]:
        """Return feature importance scores."""

    def text_dump(self) -> List[str]:
        """Return a text representation of each tree."""

    @classmethod
    def load_booster(cls, path: str) -> Self:
        """Load a multi-output booster from a file."""

    def save_booster(self, path: str):
        """Save the multi-output booster to a file."""

    @classmethod
    def from_json(cls, json_str: str) -> Self:
        """Deserialize a multi-output booster from a JSON string."""

    def json_dump(self) -> str:
        """Serialize the multi-output booster to a JSON string."""

    def get_params(self) -> Dict[str, Any]:
        """Return the booster's configuration parameters."""

    def insert_metadata(self, key: str, value: str) -> None:
        """Insert a key-value pair into the booster's metadata."""

    def get_metadata(self, key: str) -> str:
        """Retrieve a metadata value by key."""


class IVBoosterType(Protocol):
    """Protocol for the Instrumental Variable booster interface."""

    def fit(
        self,
        flat_x: np.ndarray,
        x_rows: int,
        x_cols: int,
        flat_z: np.ndarray,
        z_rows: int,
        z_cols: int,
        y: np.ndarray,
        w: np.ndarray,
    ):
        """Fit the IV booster (2-stage)."""

    def predict(
        self,
        flat_x: np.ndarray,
        x_rows: int,
        x_cols: int,
        w_counterfactual: np.ndarray,
    ) -> np.ndarray:
        """Predict outcomes under counterfactual treatments."""

    @classmethod
    def from_json(cls, json_str: str) -> Self:
        """Deserialize an IV booster from a JSON string."""

    def json_dump(self) -> str:
        """Serialize the IV booster to a JSON string."""

    def get_params(self) -> Dict[str, Any]:
        """Return the booster's configuration parameters."""
