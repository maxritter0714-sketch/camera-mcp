"""Configuration models and loading: YAML, ``${VAR}`` substitution and validation."""

import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any, Literal
from urllib.parse import urlsplit

import yaml
from dotenv import dotenv_values
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

CONFIG_ENV_VAR = "CAMERA_MCP_CONFIG"
DEFAULT_CONFIG_PATH = Path("config.yaml")

_PLACEHOLDER = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_STREAM_SCHEMES = frozenset({"rtsp", "rtsps", "http", "https"})

Coordinate = Annotated[float, Field(ge=0.0, le=1.0)]
Point = tuple[Coordinate, Coordinate]


class ConfigError(Exception):
    """The configuration cannot be used. The message never contains configuration values."""


class _StrictModel(BaseModel):
    """Rejects unknown keys so that typos in the YAML surface as errors."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class DetectorConfig(_StrictModel):
    model: str = Field(default="yolo26n.pt", min_length=1)
    device: Literal["auto", "cuda", "cpu"] = "auto"
    default_confidence: float = Field(default=0.4, ge=0.0, le=1.0)
    imgsz: int = Field(default=640, gt=0)


class FileSource(_StrictModel):
    type: Literal["file"]
    path: Path
    loop: bool = True


class WebcamSource(_StrictModel):
    type: Literal["webcam"]
    index: int = Field(default=0, ge=0)


class RtspSource(_StrictModel):
    """Network stream. Named after the config type; HTTP streams are accepted as well."""

    type: Literal["rtsp"]
    url: str

    @field_validator("url")
    @classmethod
    def _check_scheme(cls, url: str) -> str:
        if urlsplit(url).scheme.lower() not in _STREAM_SCHEMES:
            allowed = ", ".join(sorted(_STREAM_SCHEMES))
            raise ValueError(f"must be a stream URL with one of the schemes: {allowed}")
        return url


Source = Annotated[FileSource | WebcamSource | RtspSource, Field(discriminator="type")]


class CameraConfig(_StrictModel):
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    source: Source


class ZoneConfig(_StrictModel):
    name: str = Field(min_length=1)
    polygon: list[Point] = Field(min_length=3)


class LimitsConfig(_StrictModel):
    max_image_width: int = Field(default=1024, gt=0)
    jpeg_quality: int = Field(default=80, ge=1, le=100)
    max_frame_age_s: float = Field(default=5, gt=0)
    reconnect_interval_s: float = Field(default=5, gt=0)


class AppConfig(_StrictModel):
    detector: DetectorConfig = DetectorConfig()
    cameras: list[CameraConfig] = Field(min_length=1)
    zones: dict[str, list[ZoneConfig]] = Field(default_factory=dict)
    limits: LimitsConfig = LimitsConfig()

    @model_validator(mode="after")
    def _check_references(self) -> "AppConfig":
        camera_ids = [camera.id for camera in self.cameras]
        duplicates = sorted({cid for cid in camera_ids if camera_ids.count(cid) > 1})
        if duplicates:
            raise ValueError(f"duplicate camera ids: {', '.join(duplicates)}")

        for camera_id, zones in self.zones.items():
            if camera_id not in camera_ids:
                known = ", ".join(sorted(camera_ids))
                raise ValueError(f"zones refer to unknown camera '{camera_id}' (known: {known})")
            names = [zone.name for zone in zones]
            repeated = sorted({name for name in names if names.count(name) > 1})
            if repeated:
                raise ValueError(
                    f"duplicate zone names for camera '{camera_id}': {', '.join(repeated)}"
                )
        return self


def _substitute(value: Any, variables: Mapping[str, str], location: str) -> Any:
    """Recursively replace ``${VAR}`` placeholders in the string values of parsed YAML."""
    if isinstance(value, dict):
        return {
            key: _substitute(item, variables, f"{location}.{key}") for key, item in value.items()
        }
    if isinstance(value, list):
        return [_substitute(item, variables, f"{location}[{i}]") for i, item in enumerate(value)]
    if not isinstance(value, str):
        return value

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in variables:
            raise ConfigError(
                f"{location}: environment variable '{name}' is not set. "
                "Set it in the environment or in a .env file next to the config file."
            )
        return variables[name]

    return _PLACEHOLDER.sub(replace, value)


def _read_variables(env_file: Path) -> dict[str, str]:
    """Merge ``.env`` values with the process environment, which takes precedence."""
    try:
        parsed = dotenv_values(env_file)
    except UnicodeDecodeError as exc:
        raise ConfigError(f"Environment file {env_file} is not valid UTF-8.") from exc
    from_file = {key: value for key, value in parsed.items() if value is not None}
    return from_file | dict(os.environ)


def _format_validation_error(path: Path, error: ValidationError) -> str:
    """List location and cause only: pydantic's default text echoes values, including secrets."""
    lines = [f"Invalid configuration in {path}:"]
    for item in error.errors(include_input=False, include_url=False):
        location = ".".join(str(part) for part in item["loc"]) or "(top level)"
        message = item["msg"].removeprefix("Value error, ")
        lines.append(f"  - {location}: {message}")
    return "\n".join(lines)


def _resolve_source_paths(config: AppConfig, base_dir: Path) -> AppConfig:
    """Anchor relative file sources to the config directory, independent of the working dir."""
    cameras = [
        camera.model_copy(
            update={
                "source": camera.source.model_copy(update={"path": base_dir / camera.source.path})
            }
        )
        if isinstance(camera.source, FileSource)
        else camera
        for camera in config.cameras
    ]
    return config.model_copy(update={"cameras": cameras})


def resolve_config_path() -> Path:
    """Return the path from ``CAMERA_MCP_CONFIG``, or ``./config.yaml`` by default."""
    return Path(os.environ.get(CONFIG_ENV_VAR) or DEFAULT_CONFIG_PATH)


def load_config(path: Path) -> AppConfig:
    """Load and validate a configuration file.

    Raises:
        ConfigError: If the file is missing, unreadable or invalid.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(
            f"Cannot read configuration file {path}: {exc.strerror}. "
            f"Copy config.example.yaml to config.yaml or set {CONFIG_ENV_VAR}."
        ) from exc
    except UnicodeDecodeError as exc:
        raise ConfigError(f"Configuration file {path} is not valid UTF-8.") from exc

    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        position = f" at line {mark.line + 1}, column {mark.column + 1}" if mark else ""
        raise ConfigError(f"Configuration file {path} is not valid YAML{position}.") from exc

    if not isinstance(raw, dict):
        raise ConfigError(
            f"Configuration file {path} must contain a YAML mapping at the top level."
        )

    base_dir = path.resolve().parent
    substituted = _substitute(raw, _read_variables(base_dir / ".env"), "config")

    try:
        config = AppConfig.model_validate(substituted)
    except ValidationError as exc:
        raise ConfigError(_format_validation_error(path, exc)) from exc

    return _resolve_source_paths(config, base_dir)
