from pathlib import Path

import pytest
import yaml

from camera_mcp.config import (
    CONFIG_ENV_VAR,
    ConfigError,
    FileSource,
    RtspSource,
    load_config,
    resolve_config_path,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
SECRET_URL = "rtsp://user:s3cret@camera.example:8554/stream"


def valid_config() -> dict:
    return {
        "cameras": [
            {
                "id": "sample",
                "name": "Sample",
                "source": {"type": "file", "path": "clips/sample.mp4"},
            },
            {"id": "desk", "name": "Desk", "source": {"type": "webcam", "index": 0}},
        ],
        "zones": {
            "desk": [{"name": "door", "polygon": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]}],
        },
    }


@pytest.fixture
def write_config(tmp_path: Path):
    def write(data: dict) -> Path:
        path = tmp_path / "config.yaml"
        path.write_text(yaml.safe_dump(data), encoding="utf-8")
        return path

    return write


def test_valid_config_loads_and_applies_defaults(write_config) -> None:
    config = load_config(write_config(valid_config()))

    assert [camera.id for camera in config.cameras] == ["sample", "desk"]
    assert config.detector.device == "auto"
    assert config.limits.max_image_width == 1024
    assert config.zones["desk"][0].name == "door"


def test_example_config_is_valid(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PHONE_STREAM_URL", SECRET_URL)

    config = load_config(REPO_ROOT / "config.example.yaml")

    assert [camera.id for camera in config.cameras] == ["sample", "desk", "phone"]
    assert config.zones["phone"][0].name == "tuer"


def test_relative_file_source_is_anchored_to_config_directory(
    write_config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = write_config(valid_config())
    monkeypatch.chdir(REPO_ROOT)

    source = load_config(path).cameras[0].source

    assert isinstance(source, FileSource)
    assert source.path == tmp_path.resolve() / "clips" / "sample.mp4"


def rtsp_config() -> dict:
    data = valid_config()
    data["cameras"].append(
        {"id": "phone", "name": "Phone", "source": {"type": "rtsp", "url": "${TEST_STREAM}"}}
    )
    return data


def phone_url(path: Path) -> str:
    source = load_config(path).cameras[2].source
    assert isinstance(source, RtspSource)
    return source.url


def test_placeholder_is_read_from_environment(
    write_config, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TEST_STREAM", SECRET_URL)

    assert phone_url(write_config(rtsp_config())) == SECRET_URL


def test_placeholder_is_read_from_env_file_next_to_config(
    write_config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("TEST_STREAM", raising=False)
    (tmp_path / ".env").write_text(f'TEST_STREAM="{SECRET_URL}"\n', encoding="utf-8")

    assert phone_url(write_config(rtsp_config())) == SECRET_URL


def test_environment_takes_precedence_over_env_file(
    write_config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TEST_STREAM", SECRET_URL)
    (tmp_path / ".env").write_text("TEST_STREAM=rtsp://from-file/stream\n", encoding="utf-8")

    assert phone_url(write_config(rtsp_config())) == SECRET_URL


def test_special_characters_in_substituted_value_do_not_break_yaml(
    write_config, monkeypatch: pytest.MonkeyPatch
) -> None:
    url = "rtsp://user:p#a: {x}@camera.example/stream"
    monkeypatch.setenv("TEST_STREAM", url)

    assert phone_url(write_config(rtsp_config())) == url


def test_env_file_that_is_not_utf8_gives_clear_error(write_config, tmp_path: Path) -> None:
    (tmp_path / ".env").write_bytes("TEST_STREAM=Passwört\n".encode("latin-1"))

    with pytest.raises(ConfigError, match=r"\.env.*UTF-8"):
        load_config(write_config(valid_config()))


def test_missing_variable_names_variable_and_location(
    write_config, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("TEST_STREAM", raising=False)

    with pytest.raises(ConfigError, match=r"TEST_STREAM.*is not set") as excinfo:
        load_config(write_config(rtsp_config()))

    assert "cameras[2].source.url" in str(excinfo.value)


def test_error_message_does_not_leak_credentials(
    write_config, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TEST_STREAM", "ftp://user:s3cret@camera.example/stream")

    with pytest.raises(ConfigError) as excinfo:
        load_config(write_config(rtsp_config()))

    assert "cameras.2.source.rtsp.url" in str(excinfo.value)
    assert "s3cret" not in str(excinfo.value)


def mutate(data: dict, *keys: object, value: object) -> dict:
    target = data
    for key in keys[:-1]:
        target = target[key]
    target[keys[-1]] = value
    return data


@pytest.mark.parametrize(
    ("keys", "value", "expected"),
    [
        (("detector",), {"device": "gpu"}, "detector.device"),
        (("detector",), {"default_confidence": 1.5}, "detector.default_confidence"),
        (("detector",), {"modell": "typo.pt"}, "detector.modell"),
        (("limits",), {"jpeg_quality": 0}, "limits.jpeg_quality"),
        (("limits",), {"max_image_width": -1}, "limits.max_image_width"),
        (("cameras",), [], "cameras"),
        (("cameras", 1, "source"), {"type": "usb"}, "cameras.1.source"),
        (("cameras", 1, "id"), "sample", "duplicate camera ids: sample"),
        (("zones",), {"ghost": []}, "unknown camera 'ghost'"),
        (("zones", "desk", 0, "polygon"), [[0.0, 0.0], [1.0, 1.0]], "zones.desk.0.polygon"),
        (("zones", "desk", 0, "polygon"), [[0, 0], [1, 0], [1, 1.5]], "zones.desk.0.polygon.2.1"),
    ],
)
def test_invalid_values_are_rejected_with_field_location(
    write_config, keys: tuple, value: object, expected: str
) -> None:
    data = mutate(valid_config(), *keys, value=value)

    with pytest.raises(ConfigError) as excinfo:
        load_config(write_config(data))

    assert expected in str(excinfo.value)


def test_duplicate_zone_names_are_rejected(write_config) -> None:
    data = valid_config()
    data["zones"]["desk"].append(data["zones"]["desk"][0])

    with pytest.raises(ConfigError, match="duplicate zone names for camera 'desk': door"):
        load_config(write_config(data))


def test_missing_file_gives_hint(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="config.example.yaml"):
        load_config(tmp_path / "absent.yaml")


@pytest.mark.parametrize("content", ["cameras: [unclosed", "- just\n- a list\n", ""])
def test_malformed_yaml_is_rejected(tmp_path: Path, content: str) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(ConfigError, match="YAML"):
        load_config(path)


def test_config_path_comes_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(CONFIG_ENV_VAR, "elsewhere/custom.yaml")

    assert resolve_config_path() == Path("elsewhere/custom.yaml")


def test_config_path_defaults_to_working_directory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(CONFIG_ENV_VAR, raising=False)

    assert resolve_config_path() == Path("config.yaml")
