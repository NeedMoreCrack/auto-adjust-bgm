#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
auto_adjust_bgm.py

用途：
    使用 ELUATE / BandIt v2 將影片音訊分離為 Speech / Music / SFX，
    固定丟棄 Music，並分別調整 Speech 與 SFX 音量後重新混音，
    最後將新音訊 mux 回原影片；影像串流不重新編碼。

支援：
    - Linux / WSL2
    - macOS（Apple Silicon 可用 MPS；Intel Mac 通常落到 CPU）

不建議：
    - Windows 原生：ELUATE 目前使用 Unix-specific Python API，建議改用 WSL2。

設定：
    - 預設讀取腳本同目錄的 config.yaml（若存在）
    - 可用 --config 指定其他 YAML
    - CLI 參數會覆蓋 YAML
    - YAML 未指定的值會使用程式預設值
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


# ============================================================
# Constants / defaults
# ============================================================

ELUATE_PACKAGE = "eluate"
TOOL_HOME = Path.home() / ".auto_remove_bgm"
TOOL_VENV = TOOL_HOME / "venv"
ELUATE_HOME = Path.home() / ".eluate"
ELUATE_MODELS_DIR = ELUATE_HOME / "models"

CHECKPOINTS = {"multi", "eng", "deu", "fra", "spa", "cmn", "fao"}
DEVICES = {"cuda", "cpu", "mps"}

SUPPORTED_EXTENSIONS = {
    ".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v",
    ".mpeg", ".mpg", ".ts", ".mts", ".m2ts", ".flv",
    ".wmv", ".3gp", ".3g2",
}

DEFAULT_CONFIG: dict[str, Any] = {
    "input": None,
    "output": None,
    "eluate": {
        "checkpoint": "multi",
        "device": None,
        "force": False,
    },
    "audio": {
        "speech_volume": 1.0,
        "sfx_volume": 1.0,
        "codec": "aac",
        "bitrate": "256k",
        "limiter": True,
        "limiter_level": 0.95,
    },
    "processing": {
        "overwrite": True,
        "keep_stems": False,
    },
}


# ============================================================
# Console
# ============================================================

def print_header(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def print_info(message: str) -> None:
    print(f"[INFO] {message}")


def print_ok(message: str) -> None:
    print(f"[OK] {message}")


def print_warn(message: str) -> None:
    print(f"[WARN] {message}")


def print_error(message: str) -> None:
    print(f"[ERROR] {message}", file=sys.stderr)


# ============================================================
# Platform
# ============================================================

def is_windows() -> bool:
    return os.name == "nt"


def is_linux() -> bool:
    return sys.platform.startswith("linux")


def is_macos() -> bool:
    return sys.platform == "darwin"


def is_wsl() -> bool:
    if not is_linux():
        return False
    text = f"{platform.release()} {platform.version()}".lower()
    return "microsoft" in text or "wsl" in text


def show_platform_info() -> None:
    print_info(
        f"平台：{platform.system()} {platform.release()} ({platform.machine()})"
    )
    print_info(f"Python：{sys.version.split()[0]}")
    print_info(f"Python executable：{sys.executable}")

    if is_wsl():
        print_ok("偵測到 WSL 環境")

    if is_macos():
        arch = platform.machine().lower()
        if arch in {"x86_64", "amd64"}:
            print_warn(
                "偵測到 Intel Mac；新版 PyTorch 對 Intel macOS 的 binary "
                "支援有限，通常只能使用 CPU，甚至可能遇到套件版本限制。"
            )
        elif arch in {"arm64", "aarch64"}:
            print_ok("偵測到 Apple Silicon Mac")


def validate_platform() -> None:
    if is_windows():
        raise RuntimeError(
            "目前不建議在 Windows 原生環境執行 ELUATE。\n"
            "ELUATE 目前使用 Unix-specific Python API；請改用 WSL2 / Linux。\n"
            "例如：python3 auto_adjust_bgm.py --config config.yaml"
        )


# ============================================================
# YAML configuration
# ============================================================

def deep_copy_dict(value: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(value))


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deep_copy_dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "工具 venv 中缺少 PyYAML。正常情況 bootstrap 會自動安裝；"
            "請重新執行本程式一次。"
        ) from exc

    if not path.exists():
        raise FileNotFoundError(f"找不到 YAML 設定檔：{path}")

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError("YAML 最外層必須是 mapping/object。")
    return data


def resolve_config_file(cli_config: str | None) -> Path | None:
    if cli_config:
        return Path(cli_config).expanduser().resolve()

    default_path = Path(__file__).resolve().with_name("config.yaml")
    if default_path.exists():
        return default_path

    return None


def resolve_path_from_config(value: str | None, config_file: Path | None) -> str | None:
    if value is None or str(value).strip() == "":
        return None

    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return str(path)

    if config_file is not None:
        return str((config_file.parent / path).resolve())

    return str(path.resolve())


def number_in_range(name: str, value: Any, minimum: float, maximum: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必須是數字，目前為：{value!r}") from exc

    if not minimum <= number <= maximum:
        raise ValueError(
            f"{name} 必須介於 {minimum} ~ {maximum}，目前為：{number}"
        )
    return number


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    eluate = config.setdefault("eluate", {})
    audio = config.setdefault("audio", {})
    processing = config.setdefault("processing", {})

    checkpoint = eluate.get("checkpoint", "multi")
    if checkpoint not in CHECKPOINTS:
        raise ValueError(
            f"eluate.checkpoint 不合法：{checkpoint}；可用值：{sorted(CHECKPOINTS)}"
        )

    device = eluate.get("device")
    if device is not None and device not in DEVICES:
        raise ValueError(
            f"eluate.device 不合法：{device}；可用值：{sorted(DEVICES)} 或 null"
        )

    audio["speech_volume"] = number_in_range(
        "audio.speech_volume", audio.get("speech_volume", 1.0), 0.0, 2.0
    )
    audio["sfx_volume"] = number_in_range(
        "audio.sfx_volume", audio.get("sfx_volume", 1.0), 0.0, 2.0
    )
    audio["limiter_level"] = number_in_range(
        "audio.limiter_level", audio.get("limiter_level", 0.95), 0.1, 1.0
    )

    if not isinstance(eluate.get("force", False), bool):
        raise ValueError("eluate.force 必須是 true / false。")
    if not isinstance(audio.get("limiter", True), bool):
        raise ValueError("audio.limiter 必須是 true / false。")
    if not isinstance(processing.get("overwrite", True), bool):
        raise ValueError("processing.overwrite 必須是 true / false。")
    if not isinstance(processing.get("keep_stems", False), bool):
        raise ValueError("processing.keep_stems 必須是 true / false。")

    codec = str(audio.get("codec", "aac")).strip()
    if not codec:
        raise ValueError("audio.codec 不可為空。")
    audio["codec"] = codec

    bitrate = audio.get("bitrate", "256k")
    audio["bitrate"] = None if bitrate in (None, "") else str(bitrate).strip()

    return config


def build_config(args: argparse.Namespace) -> tuple[dict[str, Any], Path | None]:
    config_file = resolve_config_file(args.config)
    config = deep_copy_dict(DEFAULT_CONFIG)

    if config_file:
        print_info(f"讀取 YAML：{config_file}")
        config = deep_merge(config, load_yaml(config_file))
    else:
        print_warn("未找到 config.yaml，使用程式預設值 / CLI 參數。")

    # CLI > YAML > defaults
    if args.input is not None:
        config["input"] = args.input
    if args.output is not None:
        config["output"] = args.output
    if args.checkpoint is not None:
        config["eluate"]["checkpoint"] = args.checkpoint
    if args.device is not None:
        config["eluate"]["device"] = args.device
    if args.force is not None:
        config["eluate"]["force"] = args.force
    if args.speech_volume is not None:
        config["audio"]["speech_volume"] = args.speech_volume
    if args.sfx_volume is not None:
        config["audio"]["sfx_volume"] = args.sfx_volume
    if args.audio_codec is not None:
        config["audio"]["codec"] = args.audio_codec
    if args.audio_bitrate is not None:
        config["audio"]["bitrate"] = args.audio_bitrate
    if args.limiter is not None:
        config["audio"]["limiter"] = args.limiter
    if args.overwrite is not None:
        config["processing"]["overwrite"] = args.overwrite
    if args.keep_stems is not None:
        config["processing"]["keep_stems"] = args.keep_stems

    config["input"] = resolve_path_from_config(config.get("input"), config_file)
    config["output"] = resolve_path_from_config(config.get("output"), config_file)

    if not config.get("input"):
        raise ValueError(
            "沒有指定 input。請在 config.yaml 設定 input，"
            "或使用 CLI：python3 auto_adjust_bgm.py video.mp4"
        )

    return validate_config(config), config_file


# ============================================================
# Linux package helpers / FFmpeg
# ============================================================

def sudo_prefix() -> list[str]:
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        return []

    sudo = shutil.which("sudo")
    if sudo:
        return [sudo]

    raise RuntimeError("需要 root 權限，但找不到 sudo。")


def detect_linux_package_manager() -> str | None:
    for manager in ("apt-get", "dnf", "yum", "pacman", "zypper"):
        if shutil.which(manager):
            return manager
    return None


def install_ffmpeg_linux() -> None:
    manager = detect_linux_package_manager()
    if not manager:
        raise RuntimeError("找不到支援的 Linux 套件管理器，無法自動安裝 FFmpeg。")

    prefix = sudo_prefix()
    print_info(f"Linux：使用 {manager} 安裝 FFmpeg...")

    if manager == "apt-get":
        subprocess.check_call(prefix + [manager, "update"])
        subprocess.check_call(prefix + [manager, "install", "-y", "ffmpeg"])
    elif manager in {"dnf", "yum"}:
        subprocess.check_call(prefix + [manager, "install", "-y", "ffmpeg"])
    elif manager == "pacman":
        subprocess.check_call(prefix + [manager, "-Sy", "--noconfirm", "ffmpeg"])
    elif manager == "zypper":
        subprocess.check_call(
            prefix + [manager, "--non-interactive", "install", "ffmpeg"]
        )


def install_ffmpeg_macos() -> None:
    brew = shutil.which("brew")
    if not brew:
        raise RuntimeError(
            "找不到 Homebrew。請先安裝 Homebrew，再執行：brew install ffmpeg"
        )
    print_info("macOS：使用 Homebrew 安裝 FFmpeg...")
    subprocess.check_call([brew, "install", "ffmpeg"])


def find_ffmpeg(auto_install: bool = True) -> str:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg

    if not auto_install:
        raise RuntimeError("找不到 FFmpeg。")

    if is_linux():
        install_ffmpeg_linux()
    elif is_macos():
        install_ffmpeg_macos()
    else:
        raise RuntimeError("目前平台不支援自動安裝 FFmpeg。")

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("FFmpeg 安裝完成後仍找不到 ffmpeg。")
    return ffmpeg


def check_ffmpeg() -> str:
    ffmpeg = find_ffmpeg(auto_install=True)
    result = subprocess.run(
        [ffmpeg, "-version"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg 無法執行：\n{result.stderr}")

    first_line = result.stdout.splitlines()[0] if result.stdout else ffmpeg
    print_ok(f"找到 FFmpeg：{ffmpeg}")
    print_ok(first_line)
    return ffmpeg


# ============================================================
# Runtime bootstrap / dedicated venv
# ============================================================

BOOTSTRAP_PACKAGE = "PyYAML"
BOOTSTRAP_MODULE = "yaml"


def get_venv_python() -> Path:
    if is_windows():
        return TOOL_VENV / "Scripts" / "python.exe"
    return TOOL_VENV / "bin" / "python"


def get_venv_eluate() -> Path:
    if is_windows():
        return TOOL_VENV / "Scripts" / "eluate.exe"
    return TOOL_VENV / "bin" / "eluate"


def is_valid_venv_python(python_path: Path) -> bool:
    """實際執行目標 Python，確認它確實是 TOOL_VENV 的 venv Python。"""
    if not python_path.exists():
        return False

    code = (
        "import json,sys; "
        "print(json.dumps({'prefix': sys.prefix, 'base_prefix': sys.base_prefix}))"
    )

    result = subprocess.run(
        [str(python_path), "-c", code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )

    if result.returncode != 0:
        return False

    try:
        info = json.loads(result.stdout.strip().splitlines()[-1])
        prefix = Path(info["prefix"]).expanduser().resolve()
        base_prefix = Path(info["base_prefix"]).expanduser().resolve()
        expected = TOOL_VENV.expanduser().resolve()
    except Exception:
        return False

    return prefix == expected and prefix != base_prefix


def is_running_in_tool_venv() -> bool:
    try:
        return Path(sys.prefix).expanduser().resolve() == TOOL_VENV.expanduser().resolve()
    except Exception:
        return False


def install_python_venv_support_linux() -> None:
    """Ubuntu / Debian 缺少 ensurepip/venv 時，自動用 apt 補齊。"""
    if not is_linux():
        return

    apt = shutil.which("apt-get")
    if not apt:
        raise RuntimeError(
            "目前 Python 無法建立 venv，而且找不到 apt-get。\n"
            "請先安裝對應的 python3-venv 套件。"
        )

    major = sys.version_info.major
    minor = sys.version_info.minor
    version_pkg = f"python{major}.{minor}-venv"
    prefix = sudo_prefix()

    print_info(f"嘗試自動安裝 {version_pkg}...")
    subprocess.check_call(prefix + [apt, "update"])

    result = subprocess.run(
        prefix + [apt, "install", "-y", version_pkg],
        check=False,
        )

    if result.returncode != 0:
        print_warn(f"{version_pkg} 安裝失敗，改嘗試 python3-venv。")
        subprocess.check_call(prefix + [apt, "install", "-y", "python3-venv"])


def ensure_pip(python_path: Path) -> None:
    result = subprocess.run(
        [str(python_path), "-m", "pip", "--version"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )

    if result.returncode == 0:
        return

    print_warn("venv 中找不到 pip，嘗試使用 ensurepip 修復。")
    subprocess.check_call([str(python_path), "-m", "ensurepip", "--upgrade"])


def create_tool_venv() -> Path:
    TOOL_HOME.mkdir(parents=True, exist_ok=True)
    python_path = get_venv_python()

    print_info(f"建立工具專用 Python venv：{TOOL_VENV}")

    result = subprocess.run(
        [sys.executable, "-m", "venv", str(TOOL_VENV)],
        check=False,
    )

    if result.returncode != 0:
        if is_linux():
            install_python_venv_support_linux()
            subprocess.check_call(
                [sys.executable, "-m", "venv", str(TOOL_VENV)]
            )
        else:
            raise RuntimeError("建立工具專用 Python venv 失敗。")

    if not is_valid_venv_python(python_path):
        raise RuntimeError(
            "Python venv 建立完成後仍無法通過驗證：\n"
            f"{TOOL_VENV}"
        )

    ensure_pip(python_path)
    print_ok(f"工具專用 venv 已建立：{TOOL_VENV}")
    return python_path


def ensure_tool_venv() -> Path:
    """
    確保 TOOL_VENV 是真正有效的 venv。

    舊版只看 bin/python 是否存在，可能留下壞掉或曾被移動的環境，
    最後導致 pip 誤判成系統 Python並觸發 PEP 668。
    """
    python_path = get_venv_python()

    if is_valid_venv_python(python_path):
        ensure_pip(python_path)
        return python_path

    if TOOL_VENV.exists():
        if is_running_in_tool_venv():
            raise RuntimeError(
                "目前正在一個無效的工具 venv 中執行，無法安全自我刪除。\n"
                "請離開該 shell 後，再使用系統 python3 執行本程式一次。"
            )

        print_warn(
            "偵測到損壞、被移動或不是有效 venv 的舊環境，將自動重建：\n"
            f"{TOOL_VENV}"
        )
        shutil.rmtree(TOOL_VENV, ignore_errors=False)

    return create_tool_venv()


def module_importable(python_executable: str | Path, module_name: str) -> bool:
    result = subprocess.run(
        [str(python_executable), "-c", f"import {module_name}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def run_pip(python_path: Path, *args: str) -> None:
    """所有 pip 安裝都強制透過已驗證的 venv Python 執行。"""
    if not is_valid_venv_python(python_path):
        raise RuntimeError(
            "拒絕執行 pip：目標 Python 不是有效的工具 venv。\n"
            f"{python_path}"
        )

    ensure_pip(python_path)
    subprocess.check_call([str(python_path), "-m", "pip", *args])


def upgrade_tooling(python_path: Path) -> None:
    print_info("更新工具 venv 的 pip / setuptools / wheel...")
    run_pip(
        python_path,
        "install",
        "--upgrade",
        "pip",
        "setuptools",
        "wheel",
    )


def ensure_bootstrap_packages(python_path: Path) -> None:
    """只安裝讀 YAML 所需的最小依賴，ELUATE/PyTorch 等讀完設定再處理。"""
    if module_importable(python_path, BOOTSTRAP_MODULE):
        return

    print_info(f"安裝啟動必要套件：{BOOTSTRAP_PACKAGE}")
    run_pip(
        python_path,
        "install",
        "--upgrade",
        BOOTSTRAP_PACKAGE,
    )

    if not module_importable(python_path, BOOTSTRAP_MODULE):
        raise RuntimeError(
            f"{BOOTSTRAP_PACKAGE} 安裝完成後仍無法 import {BOOTSTRAP_MODULE}。"
        )


def bootstrap_runtime() -> None:
    """
    讓 Git clone 後可以直接執行：
        python3 auto_adjust_bgm.py

    第一次執行：
      1. 建立 / 修復 ~/.auto_remove_bgm/venv
      2. 安裝 PyYAML
      3. 使用該 venv Python 重新啟動本程式

    重新啟動後才讀 YAML，再依 device 安裝/修復 ELUATE 與 PyTorch。
    """
    validate_platform()

    python_path = ensure_tool_venv()
    ensure_bootstrap_packages(python_path)

    if is_running_in_tool_venv():
        return

    print_info(f"切換到工具專用 Python：{python_path}")
    os.execv(
        str(python_path),
        [
            str(python_path),
            str(Path(__file__).resolve()),
            *sys.argv[1:],
        ],
    )


def install_eluate(requested_device: str | None = None) -> str:
    python_path = ensure_tool_venv()
    upgrade_tooling(python_path)

    package_spec = "eluate[cpu]" if requested_device == "cpu" else ELUATE_PACKAGE

    print_info(f"開始安裝 / 更新 ELUATE：{package_spec}")
    run_pip(
        python_path,
        "install",
        "--upgrade",
        package_spec,
    )

    eluate = get_venv_eluate()
    if not eluate.exists():
        raise RuntimeError(f"ELUATE 安裝完成，但找不到 CLI：{eluate}")

    print_ok(f"ELUATE 安裝完成：{eluate}")
    return str(eluate)


def install_pytorch(python_path: Path, requested_device: str | None) -> None:
    """修復極端情況：ELUATE 已存在，但 torch / torchaudio 不完整。"""
    print_warn("ELUATE 專用環境缺少可用的 PyTorch / torchaudio。")

    if requested_device == "cpu":
        print_info("重新安裝 CPU 模式 ELUATE 依賴...")
        package_args = ["eluate[cpu]"]
    else:
        print_info("開始安裝 / 更新 PyTorch 與 torchaudio...")
        package_args = ["torch", "torchaudio"]

    try:
        run_pip(
            python_path,
            "install",
            "--upgrade",
            *package_args,
        )
    except subprocess.CalledProcessError as exc:
        if is_macos() and platform.machine().lower() in {"x86_64", "amd64"}:
            raise RuntimeError(
                "PyTorch 安裝失敗。這台是 Intel Mac；目前新版 PyTorch 對 macOS "
                "x86_64 的 wheel 支援有限。可能需要使用較舊且相容的 Python / "
                "PyTorch，或改在 Linux / WSL / Apple Silicon Mac 執行。"
            ) from exc

        raise RuntimeError(
            "PyTorch 自動安裝失敗。工具 venv 已經是隔離環境，因此不應使用 "
            "--break-system-packages。請檢查網路、Python 版本與 PyTorch wheel 相容性。\n"
            f"venv Python：{python_path}"
        ) from exc

    if not module_importable(python_path, "torch"):
        raise RuntimeError("安裝完成後仍無法 import torch。")
    if not module_importable(python_path, "torchaudio"):
        raise RuntimeError("安裝完成後仍無法 import torchaudio。")

    print_ok("PyTorch / torchaudio 已就緒。")


def ensure_eluate(requested_device: str | None = None) -> str:
    """
    自動修復 ELUATE runtime：
      - venv 無效 -> 自動重建
      - ELUATE 不存在 -> 安裝
      - torch / torchaudio 遺失 -> 自動補裝
      - eluate import 失敗 -> 嘗試更新 ELUATE
    """
    python_path = ensure_tool_venv()
    eluate = get_venv_eluate()

    if not eluate.exists():
        print_warn("找不到 ELUATE CLI，準備安裝。")
        install_eluate(requested_device)
        eluate = get_venv_eluate()

    if not module_importable(python_path, "torch") or not module_importable(
            python_path, "torchaudio"
    ):
        install_pytorch(python_path, requested_device)

    if not module_importable(python_path, "eluate"):
        print_warn("ELUATE package 無法載入，嘗試更新 / 修復 ELUATE。")
        install_eluate(requested_device)

    missing = [
        module
        for module in ("torch", "torchaudio", "eluate")
        if not module_importable(python_path, module)
    ]

    if missing:
        raise RuntimeError(
            "ELUATE runtime 修復後仍有模組無法載入："
            + ", ".join(missing)
            + "\n可嘗試刪除 ~/.auto_remove_bgm/venv 後重新執行本程式。"
        )

    print_ok(f"ELUATE 專用環境就緒：{eluate}")
    return str(eluate)


def ensure_checkpoint(eluate_path: str, checkpoint: str) -> None:
    model_path = ELUATE_MODELS_DIR / f"checkpoint-{checkpoint}.ckpt"
    if model_path.exists():
        return

    print_warn(f"找不到 checkpoint：{model_path}")
    print_info("執行 eluate setup 下載模型...")

    result = subprocess.run([eluate_path, "setup"])
    if result.returncode != 0:
        raise RuntimeError(f"eluate setup 失敗，exit code = {result.returncode}")

    if checkpoint == "multi" and not model_path.exists():
        raise RuntimeError(
            "eluate setup 執行完成，但仍找不到 multi checkpoint：\n"
            f"{model_path}"
        )


def python_for_eluate(eluate_path: str) -> str:
    python_path = get_venv_python()
    if is_valid_venv_python(python_path):
        return str(python_path)
    raise RuntimeError(f"ELUATE 專用 Python venv 無效：{TOOL_VENV}")


# ============================================================
# PyTorch device inspection
# ============================================================

def get_torch_status(python_executable: str) -> dict[str, Any]:
    code = r'''
import json
try:
    import torch
    cuda_available = bool(torch.cuda.is_available())
    mps_built = bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_built())
    mps_available = bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
    data = {
        "torch_installed": True,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": cuda_available,
        "cuda_device_name": torch.cuda.get_device_name(0) if cuda_available else None,
        "mps_built": mps_built,
        "mps_available": mps_available,
    }
except Exception as e:
    data = {"torch_installed": False, "error": f"{type(e).__name__}: {e}"}
print(json.dumps(data, ensure_ascii=False))
'''

    result = subprocess.run(
        [python_executable, "-c", code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    if result.returncode != 0:
        return {
            "torch_installed": False,
            "error": result.stderr.strip() or "無法執行 PyTorch 檢查",
        }

    try:
        return json.loads(result.stdout.strip().splitlines()[-1])
    except Exception:
        return {
            "torch_installed": False,
            "error": f"無法解析 PyTorch 狀態：{result.stdout.strip()}",
        }


def show_device_status(eluate_path: str, requested_device: str | None) -> dict[str, Any]:
    python_executable = python_for_eluate(eluate_path)
    print_info(f"檢查 ELUATE Python 環境：{python_executable}")

    status = get_torch_status(python_executable)
    if not status.get("torch_installed"):
        raise RuntimeError(
            "ELUATE 環境無法載入 PyTorch：\n"
            f"{status.get('error', '未知錯誤')}"
        )

    print_ok(f"PyTorch：{status.get('torch_version')}")

    cuda_available = bool(status.get("cuda_available"))
    mps_available = bool(status.get("mps_available"))

    if status.get("torch_cuda_version"):
        print_info(f"PyTorch CUDA runtime：{status['torch_cuda_version']}")

    if cuda_available:
        print_ok(f"CUDA 可用：{status.get('cuda_device_name')}")
    elif is_linux():
        print_warn("CUDA 不可用。")

    if mps_available:
        print_ok("MPS 可用：Apple Metal GPU")
    elif is_macos():
        if status.get("mps_built"):
            print_warn("PyTorch 包含 MPS 支援，但目前裝置無法使用 MPS。")
        else:
            print_warn("目前 PyTorch 沒有 MPS backend。")

    if requested_device == "cuda" and not cuda_available:
        raise RuntimeError("指定 --device cuda，但 ELUATE 的 PyTorch 無法使用 CUDA。")

    if requested_device == "mps" and not mps_available:
        raise RuntimeError("指定 --device mps，但目前 PyTorch 無法使用 MPS。")

    if requested_device is None:
        if cuda_available:
            print_info("自動裝置預期使用：CUDA")
        elif mps_available:
            print_info("自動裝置預期使用：MPS")
        else:
            print_info("自動裝置預期使用：CPU")

    return status


# ============================================================
# Input / output
# ============================================================

def validate_input(path_text: str) -> Path:
    input_path = Path(path_text).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"找不到影片：{input_path}")
    if not input_path.is_file():
        raise ValueError(f"輸入路徑不是檔案：{input_path}")

    extension = input_path.suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        print_warn(
            f"副檔名 {extension} 不在常見影片格式清單，仍交給 FFmpeg / ELUATE 嘗試。"
        )
    return input_path


def create_output_path(input_path: Path, output_text: str | None) -> Path:
    if output_text:
        output = Path(output_text).expanduser().resolve()
    else:
        output = input_path.with_name(
            f"{input_path.stem}_no_bgm{input_path.suffix}"
        )

    if output == input_path:
        raise ValueError("輸出檔案不能與輸入檔案相同。")

    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def format_size(size: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(size)
    for unit in units:
        if value < 1024:
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{value:.2f} PB"


# ============================================================
# ELUATE Python API runner
# ============================================================

ELUATE_API_RUNNER = r'''
import json
import sys
from pathlib import Path
import eluate

input_path = Path(sys.argv[1])
output_dir = Path(sys.argv[2])
metadata_path = Path(sys.argv[3])
checkpoint = sys.argv[4]
device_arg = sys.argv[5]
force = sys.argv[6] == "1"

device = None if device_arg == "auto" else device_arg

stage_names = {
    "extract": "Extract",
    "load_model": "Load model",
    "separate": "Separate",
    "compile": "Compile",
}

last_percent = -1
last_stage = None

def on_progress(fraction: float, stage: str) -> None:
    global last_percent, last_stage
    percent = max(0, min(100, int(round(float(fraction) * 100))))
    label = stage_names.get(stage, stage)
    if percent != last_percent or stage != last_stage:
        bar_width = 30
        filled = int(bar_width * percent / 100)
        bar = "█" * filled + "-" * (bar_width - filled)
        print(f"\r[ELUATE] {label:<12} [{bar}] {percent:3d}%", end="", flush=True)
        last_percent = percent
        last_stage = stage
    if percent >= 100:
        print(flush=True)

result = eluate.elute(
    input_path,
    outputs=("speech", "sfx"),
    output_dir=output_dir,
    overwrite=True,
    force=force,
    on_progress=on_progress,
    device=device,
    checkpoint=checkpoint,
)

payload = {
    "speech": str(result.speech) if result.speech else None,
    "sfx": str(result.sfx) if result.sfx else None,
    "duration": result.duration,
    "processing_time": result.processing_time,
}
metadata_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
'''


def extract_stems(
        eluate_path: str,
        input_path: Path,
        work_dir: Path,
        checkpoint: str,
        device: str | None,
        force: bool,
) -> tuple[Path, Path, dict[str, Any]]:
    eluate_python = python_for_eluate(eluate_path)
    metadata_path = work_dir / "eluate_result.json"

    print_header("ELUATE AI 分離")
    print_info("Music：固定丟棄")
    print_info("輸出 stems：Speech + SFX")
    print()

    command = [
        eluate_python,
        "-c",
        ELUATE_API_RUNNER,
        str(input_path),
        str(work_dir),
        str(metadata_path),
        checkpoint,
        device or "auto",
        "1" if force else "0",
        ]

    process = subprocess.run(command)
    if process.returncode != 0:
        model_path = ELUATE_MODELS_DIR / f"checkpoint-{checkpoint}.ckpt"
        print_warn(
            "ELUATE 執行失敗。如果上方訊息包含 checkpoint / SHA256 mismatch，"
            "可刪除模型後重新 setup："
        )
        print(f"  rm -f {model_path}")
        print(f"  {eluate_path} setup")
        raise RuntimeError(f"ELUATE API 執行失敗，exit code = {process.returncode}")

    if not metadata_path.exists():
        raise RuntimeError("ELUATE 已結束，但找不到結果 metadata。")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    speech = Path(metadata["speech"]) if metadata.get("speech") else None
    sfx = Path(metadata["sfx"]) if metadata.get("sfx") else None

    if speech is None or not speech.exists():
        raise RuntimeError("ELUATE 沒有產生 speech stem。")
    if sfx is None or not sfx.exists():
        raise RuntimeError("ELUATE 沒有產生 sfx stem。")

    print_ok(f"Speech stem：{speech}")
    print_ok(f"SFX stem：{sfx}")
    return speech, sfx, metadata


# ============================================================
# FFmpeg mix / mux
# ============================================================

def build_audio_filter(
        speech_volume: float,
        sfx_volume: float,
        limiter: bool,
        limiter_level: float,
) -> str:
    parts = [
        f"[1:a]volume={speech_volume}[speech]",
        f"[2:a]volume={sfx_volume}[sfx]",
        "[speech][sfx]amix=inputs=2:duration=longest:normalize=0[mix]",
    ]

    if limiter:
        parts.append(f"[mix]alimiter=limit={limiter_level}[aout]")
    else:
        parts.append("[mix]anull[aout]")

    return ";".join(parts)


def mux_adjusted_audio(
        ffmpeg: str,
        input_path: Path,
        speech_path: Path,
        sfx_path: Path,
        output_path: Path,
        *,
        speech_volume: float,
        sfx_volume: float,
        audio_codec: str,
        audio_bitrate: str | None,
        limiter: bool,
        limiter_level: float,
        overwrite: bool,
) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"輸出檔已存在，而且 processing.overwrite=false：{output_path}"
        )

    temp_output = output_path.with_name(
        f".{output_path.stem}.processing{output_path.suffix}"
    )
    if temp_output.exists():
        temp_output.unlink()

    filter_complex = build_audio_filter(
        speech_volume,
        sfx_volume,
        limiter,
        limiter_level,
    )

    command = [
        ffmpeg,
        "-y",
        "-i", str(input_path),
        "-i", str(speech_path),
        "-i", str(sfx_path),
        "-filter_complex", filter_complex,
        "-map", "0:v:0",
        "-map", "[aout]",
        "-map_metadata", "0",
        "-c:v", "copy",
        "-c:a", audio_codec,
    ]

    if audio_bitrate:
        command.extend(["-b:a", audio_bitrate])

    # 若原檔有字幕則盡量一併保留；不支援的容器會由 FFmpeg 回報。
    command.extend([
        "-map", "0:s?",
        "-c:s", "copy",
        str(temp_output),
    ])

    print_header("Speech / SFX 音量調整與 Mux")
    print_info(f"Speech volume：{speech_volume:.2f}x")
    print_info(f"SFX volume：{sfx_volume:.2f}x")
    print_info("BGM：0%（固定移除）")
    print_info(f"Limiter：{'ON' if limiter else 'OFF'}")

    result = subprocess.run(command)
    if result.returncode != 0:
        if temp_output.exists():
            temp_output.unlink(missing_ok=True)
        raise RuntimeError(f"FFmpeg 混音 / mux 失敗，exit code = {result.returncode}")

    if not temp_output.exists():
        raise RuntimeError("FFmpeg 顯示成功，但找不到暫存輸出檔。")

    if output_path.exists():
        output_path.unlink()
    temp_output.replace(output_path)


# ============================================================
# Main processing
# ============================================================

def process_video(config: dict[str, Any]) -> Path:
    print_header("Video BGM Remover / Speech & SFX Mixer")
    show_platform_info()
    validate_platform()

    input_path = validate_input(config["input"])
    output_path = create_output_path(input_path, config.get("output"))

    eluate_cfg = config["eluate"]
    audio_cfg = config["audio"]
    processing_cfg = config["processing"]

    print_info(f"輸入：{input_path}")
    print_info(f"輸出：{output_path}")
    print_info(f"Checkpoint：{eluate_cfg['checkpoint']}")
    print_info(f"Device：{eluate_cfg['device'] or 'auto'}")
    print_info(f"Speech：{audio_cfg['speech_volume']:.2f}x")
    print_info(f"SFX：{audio_cfg['sfx_volume']:.2f}x")
    print_info("BGM：0%（固定移除）")

    print_info("檢查 FFmpeg...")
    ffmpeg = check_ffmpeg()

    print_info("檢查 ELUATE...")
    eluate = ensure_eluate(eluate_cfg["device"])
    print_ok(f"ELUATE：{eluate}")

    ensure_checkpoint(eluate, eluate_cfg["checkpoint"])
    show_device_status(eluate, eluate_cfg["device"])

    output_path.parent.mkdir(parents=True, exist_ok=True)

    keep_stems = processing_cfg["keep_stems"]
    if keep_stems:
        work_dir = output_path.parent / f"{output_path.stem}_stems"
        work_dir.mkdir(parents=True, exist_ok=True)
        temp_context = None
    else:
        temp_context = tempfile.TemporaryDirectory(prefix="video_bgm_remover_")
        work_dir = Path(temp_context.name)

    try:
        speech_path, sfx_path, metadata = extract_stems(
            eluate,
            input_path,
            work_dir,
            eluate_cfg["checkpoint"],
            eluate_cfg["device"],
            eluate_cfg["force"],
        )

        mux_adjusted_audio(
            ffmpeg,
            input_path,
            speech_path,
            sfx_path,
            output_path,
            speech_volume=audio_cfg["speech_volume"],
            sfx_volume=audio_cfg["sfx_volume"],
            audio_codec=audio_cfg["codec"],
            audio_bitrate=audio_cfg["bitrate"],
            limiter=audio_cfg["limiter"],
            limiter_level=audio_cfg["limiter_level"],
            overwrite=processing_cfg["overwrite"],
        )

        print_header("處理完成")
        print_ok(f"輸出影片：{output_path}")
        print_info(f"原始大小：{format_size(input_path.stat().st_size)}")
        print_info(f"輸出大小：{format_size(output_path.stat().st_size)}")
        if metadata.get("processing_time") is not None:
            print_info(f"ELUATE 分離時間：{float(metadata['processing_time']):.2f} 秒")

        print()
        print("音訊結果：")
        print("  Music   → 移除")
        print(f"  Speech  → {audio_cfg['speech_volume']:.2f}x")
        print(f"  SFX     → {audio_cfg['sfx_volume']:.2f}x")

        if keep_stems:
            print_info(f"保留 stems：{work_dir}")

        return output_path

    finally:
        if temp_context is not None:
            temp_context.cleanup()


# ============================================================
# CLI
# ============================================================

def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "使用 ELUATE / BandIt v2 固定移除 BGM，"
            "並分別調整 Speech / SFX 音量。"
        )
    )

    parser.add_argument(
        "input",
        nargs="?",
        default=None,
        help="輸入影片；若省略則從 YAML 的 input 讀取。",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="YAML 設定檔；未指定時自動尋找腳本同目錄 config.yaml。",
    )
    parser.add_argument("-o", "--output", default=None, help="覆蓋 YAML output。")
    parser.add_argument("--checkpoint", choices=sorted(CHECKPOINTS), default=None)
    parser.add_argument("--device", choices=sorted(DEVICES), default=None)
    parser.add_argument(
        "--force", action=argparse.BooleanOptionalAction, default=None,
        help="覆蓋 YAML eluate.force。",
    )
    parser.add_argument("--speech-volume", type=float, default=None)
    parser.add_argument("--sfx-volume", type=float, default=None)
    parser.add_argument("--audio-codec", default=None)
    parser.add_argument("--audio-bitrate", default=None)
    parser.add_argument(
        "--limiter", action=argparse.BooleanOptionalAction, default=None,
        help="啟用/停用混音後 limiter。",
    )
    parser.add_argument(
        "--overwrite", action=argparse.BooleanOptionalAction, default=None,
        help="是否允許覆蓋既有輸出檔。",
    )
    parser.add_argument(
        "--keep-stems", action=argparse.BooleanOptionalAction, default=None,
        help="是否保留 speech/sfx WAV stems。",
    )
    return parser


def main() -> int:
    args = create_parser().parse_args()

    try:
        config, _ = build_config(args)
        process_video(config)
        return 0

    except KeyboardInterrupt:
        print()
        print_error("使用者取消處理。")
        return 130

    except Exception as exc:
        print()
        print_header("處理失敗")
        print_error(f"{type(exc).__name__}: {exc}")
        print()

        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    try:
        bootstrap_runtime()
    except KeyboardInterrupt:
        print_error("使用者取消初始化。")
        sys.exit(130)
    except Exception as exc:
        print_header("初始化失敗")
        print_error(f"{type(exc).__name__}: {exc}")
        sys.exit(1)

    sys.exit(main())
