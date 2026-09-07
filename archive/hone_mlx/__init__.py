"""hone: a model- and dataset-agnostic supervised fine-tuning pipeline."""

from hone.config import REQUIRED_KEYS, load, save, validate
from hone.errors import (
    BackendError,
    ConfigError,
    DataError,
    HoneError,
    PipelineError,
    ValidationError,
)
from hone.jsonl import Reader, Writer
from hone.log import LOGGER, get, setup
from hone.model import Example, JsonScalar, Message, Meta, Role
from hone.normalize import Normalizer, SweNormalizer
from hone.split import MIN_VALID, Splitter

__version__ = "0.2.0"

__all__ = [
    "LOGGER",
    "MIN_VALID",
    "REQUIRED_KEYS",
    "BackendError",
    "ConfigError",
    "DataError",
    "Example",
    "HoneError",
    "JsonScalar",
    "Message",
    "Meta",
    "Normalizer",
    "PipelineError",
    "Reader",
    "Role",
    "Splitter",
    "SweNormalizer",
    "ValidationError",
    "Writer",
    "__version__",
    "get",
    "load",
    "save",
    "setup",
    "validate",
]
