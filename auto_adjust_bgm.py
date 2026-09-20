#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
auto_adjust_bgm.py

用途：
    使用 ELUATE / BandIt v2 移除影片中的背景音樂 BGM，
    盡可能保留：
        - 人聲 Speech
        - 遊戲音效 SFX
        - 環境音效

處理架構：
    Video
      ↓
    FFmpeg
      ↓
    ELUATE / BandIt v2
      ↓
    Speech / Music / SFX
      ↓
    移除 Music
      ↓
    Speech + SFX
      ↓
    FFmpeg mux

特色：
    - 不使用 MoviePy
    - 不使用 pydub
    - 不使用 Demucs
    - 不重新編碼影像
    - FFmpeg 缺少時可自動安裝
    - ELUATE 使用獨立專用 venv
    - 支援 CUDA / MPS / CPU 狀態檢查
    - Windows 原生環境主動阻擋，建議使用 WSL2
    - 避免誤用全域 ELUATE
    - ELUATE / checkpoint 錯誤提供較清楚的診斷
    - 使用暫存輸出檔，成功後才覆蓋正式輸出
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


# ============================================================
# 設定
# ============================================================

ELUATE_PACKAGE = "eluate"

TOOL_HOME = Path.home() / ".auto_remove_bgm"
TOOL_VENV = TOOL_HOME / "venv"

ELUATE_HOME = Path.home() / ".eluate"
ELUATE_MODELS_DIR = ELUATE_HOME / "models"

SUPPORTED_EXTENSIONS = {
    ".mp4",
    ".mkv",
    ".mov",
    ".avi",
    ".webm",
    ".m4v",
    ".mpeg",
    ".mpg",
    ".ts",
    ".mts",
    ".m2ts",
}


# ============================================================
# Console
# ============================================================

def print_header(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def print_error(message: str) -> None:
    print(f"[ERROR] {message}", file=sys.stderr)


def print_info(message: str) -> None:
    print(f"[INFO] {message}")


def print_ok(message: str) -> None:
    print(f"[OK] {message}")


def print_warn(message: str) -> None:
    print(f"[WARN] {message}")


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

    try:
        release = platform.release().lower()
        version = platform.version().lower()

        return (
                "microsoft" in release
                or "wsl" in release
                or "microsoft" in version
        )
    except Exception:
        return False


def get_machine_arch() -> str:
    return platform.machine().lower()


def show_platform_info() -> None:
    print_info(
        f"平台：{platform.system()} "
        f"{platform.release()} "
        f"({platform.machine()})"
    )

    print_info(f"Python：{sys.version.split()[0]}")
    print_info(f"Python executable：{sys.executable}")

    if is_wsl():
        print_ok("偵測到 WSL 環境")

    if is_macos():
        arch = get_machine_arch()

        if arch in ("x86_64", "amd64"):
            print_warn(
                "偵測到 Intel Mac。"
                "新版 PyTorch 對 Intel macOS 的官方 binary 支援有限，"
                "可能只能使用較舊 PyTorch 或 CPU。"
            )
        elif arch in ("arm64", "aarch64"):
            print_ok("偵測到 Apple Silicon Mac")


def validate_platform() -> None:
    """
    ELUATE 現行實作包含 Unix-only 的 resource 模組，
    因此 Windows 原生容易直接失敗。

    Windows 使用者建議改用 WSL2。
    """

    if is_windows():
        raise RuntimeError(
            "目前不建議在 Windows 原生環境執行 ELUATE。\n\n"
            "原因：ELUATE 使用 Unix-specific Python API，"
            "Windows 原生可能出現：\n"
            "ModuleNotFoundError: No module named 'resource'\n\n"
            "請改用 WSL2 / Ubuntu 執行，例如：\n\n"
            "  python3 auto_adjust_bgm.py TestVid.mp4 --device cuda"
        )


# ============================================================
# 權限 / Linux package manager
# ============================================================

def sudo_prefix() -> list[str]:
    if is_windows():
        return []

    if hasattr(os, "geteuid") and os.geteuid() == 0:
        return []

    sudo = shutil.which("sudo")

    if sudo:
        return [sudo]

    raise RuntimeError(
        "需要 root 權限，但系統找不到 sudo。"
    )


def detect_linux_package_manager() -> str | None:
    for manager in (
            "apt-get",
            "dnf",
            "yum",
            "pacman",
            "zypper",
    ):
        if shutil.which(manager):
            return manager

    return None


# ============================================================
# FFmpeg
# ============================================================

def find_windows_ffmpeg() -> str | None:
    ffmpeg = shutil.which("ffmpeg")

    if ffmpeg:
        return ffmpeg

    home = Path.home()

    winget_link = (
            home
            / "AppData"
            / "Local"
            / "Microsoft"
            / "WinGet"
            / "Links"
            / "ffmpeg.exe"
    )

    if winget_link.exists():
        return str(winget_link)

    packages_dir = (
            home
            / "AppData"
            / "Local"
            / "Microsoft"
            / "WinGet"
            / "Packages"
    )

    if packages_dir.exists():
        candidates = list(
            packages_dir.glob(
                "Gyan.FFmpeg*/**/bin/ffmpeg.exe"
            )
        )

        if candidates:
            return str(candidates[0])

        candidates = list(
            packages_dir.glob(
                "Gyan.FFmpeg*/**/ffmpeg.exe"
            )
        )

        if candidates:
            return str(candidates[0])

    return None


def install_ffmpeg_windows() -> None:
    winget = shutil.which("winget")

    if not winget:
        raise RuntimeError(
            "找不到 winget，無法自動安裝 FFmpeg。"
        )

    print_info(
        "Windows：使用 winget 安裝 FFmpeg..."
    )

    result = subprocess.run(
        [
            winget,
            "install",
            "--id",
            "Gyan.FFmpeg",
            "--exact",
            "--accept-source-agreements",
            "--accept-package-agreements",
        ],
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "winget 安裝 FFmpeg 失敗，"
            f"exit code = {result.returncode}"
        )


def install_ffmpeg_linux() -> None:
    manager = detect_linux_package_manager()

    if not manager:
        raise RuntimeError(
            "找不到支援的 Linux 套件管理器，"
            "無法自動安裝 FFmpeg。"
        )

    prefix = sudo_prefix()

    print_info(
        f"Linux：偵測到套件管理器 {manager}"
    )

    if manager == "apt-get":
        subprocess.check_call(
            prefix + [manager, "update"]
        )

        subprocess.check_call(
            prefix
            + [
                manager,
                "install",
                "-y",
                "ffmpeg",
            ]
        )

    elif manager in ("dnf", "yum"):
        subprocess.check_call(
            prefix
            + [
                manager,
                "install",
                "-y",
                "ffmpeg",
            ]
        )

    elif manager == "pacman":
        subprocess.check_call(
            prefix
            + [
                manager,
                "-Sy",
                "--noconfirm",
                "ffmpeg",
            ]
        )

    elif manager == "zypper":
        subprocess.check_call(
            prefix
            + [
                manager,
                "--non-interactive",
                "install",
                "ffmpeg",
            ]
        )


def install_ffmpeg_macos() -> None:
    brew = shutil.which("brew")

    if not brew:
        raise RuntimeError(
            "找不到 Homebrew。\n"
            "請先安裝 Homebrew，"
            "再執行：\n\n"
            "  brew install ffmpeg"
        )

    print_info(
        "macOS：使用 Homebrew 安裝 FFmpeg..."
    )

    subprocess.check_call(
        [
            brew,
            "install",
            "ffmpeg",
        ]
    )


def add_executable_dir_to_path(
        executable: str,
) -> None:

    exe_dir = str(
        Path(executable)
        .resolve()
        .parent
    )

    current = os.environ.get(
        "PATH",
        "",
    )

    parts = (
        current.split(os.pathsep)
        if current
        else []
    )

    if exe_dir not in parts:
        os.environ["PATH"] = (
                exe_dir
                + os.pathsep
                + current
        )


def find_ffmpeg(
        auto_install: bool = True,
) -> str:

    ffmpeg = shutil.which(
        "ffmpeg"
    )

    if ffmpeg:
        add_executable_dir_to_path(
            ffmpeg
        )

        print_ok(
            f"找到 FFmpeg：{ffmpeg}"
        )

        return ffmpeg

    if is_windows():
        ffmpeg = find_windows_ffmpeg()

        if ffmpeg:
            add_executable_dir_to_path(
                ffmpeg
            )

            print_ok(
                f"找到 FFmpeg：{ffmpeg}"
            )

            return ffmpeg

    if not auto_install:
        raise RuntimeError(
            "找不到 FFmpeg。"
        )

    if is_windows():
        install_ffmpeg_windows()

        ffmpeg = (
            find_windows_ffmpeg()
        )

    elif is_linux():
        install_ffmpeg_linux()

        ffmpeg = shutil.which(
            "ffmpeg"
        )

    elif is_macos():
        install_ffmpeg_macos()

        ffmpeg = shutil.which(
            "ffmpeg"
        )

    else:
        ffmpeg = None

    if not ffmpeg:
        raise RuntimeError(
            "FFmpeg 安裝完成後仍找不到 ffmpeg。"
        )

    add_executable_dir_to_path(
        ffmpeg
    )

    print_ok(
        f"FFmpeg 安裝完成：{ffmpeg}"
    )

    return ffmpeg


def check_ffmpeg() -> str:
    ffmpeg = find_ffmpeg(
        auto_install=True
    )

    result = subprocess.run(
        [
            ffmpeg,
            "-version",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    if result.returncode != 0:
        raise RuntimeError(
            "FFmpeg 無法執行：\n"
            f"{ffmpeg}\n\n"
            f"{result.stderr}"
        )

    lines = (
        result.stdout
        .splitlines()
    )

    if lines:
        print_ok(
            lines[0]
        )

    return ffmpeg


# ============================================================
# ELUATE venv
# ============================================================

def get_venv_python() -> Path:
    if is_windows():
        return (
                TOOL_VENV
                / "Scripts"
                / "python.exe"
        )

    return (
            TOOL_VENV
            / "bin"
            / "python"
    )


def get_venv_pip() -> Path:
    if is_windows():
        return (
                TOOL_VENV
                / "Scripts"
                / "pip.exe"
        )

    return (
            TOOL_VENV
            / "bin"
            / "pip"
    )


def get_venv_eluate() -> Path:
    if is_windows():
        return (
                TOOL_VENV
                / "Scripts"
                / "eluate.exe"
        )

    return (
            TOOL_VENV
            / "bin"
            / "eluate"
    )


def install_python_venv_support_linux() -> None:
    if not is_linux():
        return

    apt = shutil.which(
        "apt-get"
    )

    if not apt:
        return

    major = (
        sys.version_info.major
    )

    minor = (
        sys.version_info.minor
    )

    version_pkg = (
        f"python{major}.{minor}-venv"
    )

    prefix = sudo_prefix()

    print_info(
        "Python venv 功能不可用，"
        "嘗試補安裝..."
    )

    subprocess.check_call(
        prefix
        + [
            apt,
            "update",
        ]
    )

    result = subprocess.run(
        prefix
        + [
            apt,
            "install",
            "-y",
            version_pkg,
        ],
        check=False,
        )

    if result.returncode != 0:
        subprocess.check_call(
            prefix
            + [
                apt,
                "install",
                "-y",
                "python3-venv",
            ]
        )


def ensure_tool_venv() -> Path:
    python_path = (
        get_venv_python()
    )

    if python_path.exists():
        return python_path

    TOOL_HOME.mkdir(
        parents=True,
        exist_ok=True,
    )

    print_info(
        "建立 ELUATE 專用 venv："
        f"{TOOL_VENV}"
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "venv",
            str(TOOL_VENV),
        ],
        check=False,
    )

    if result.returncode != 0:

        if is_linux():
            install_python_venv_support_linux()

            subprocess.check_call(
                [
                    sys.executable,
                    "-m",
                    "venv",
                    str(TOOL_VENV),
                ]
            )

        else:
            raise RuntimeError(
                "建立 ELUATE 專用 venv 失敗。"
            )

    if not python_path.exists():
        raise RuntimeError(
            "建立 venv 後仍找不到 Python：\n"
            f"{python_path}"
        )

    print_ok(
        "ELUATE 專用 venv 已建立："
        f"{TOOL_VENV}"
    )

    return python_path


def find_eluate() -> str | None:
    """
    只允許使用本工具自己管理的 ELUATE。

    不使用 shutil.which("eluate") fallback，
    避免誤吃：
        - 系統 Python
        - global pip
        - 其他專案 venv
        - 不同 torch/CUDA 環境
    """

    managed = (
        get_venv_eluate()
    )

    if managed.exists():
        return str(managed)

    return None


def install_eluate() -> str:
    print_info(
        "目前找不到本工具管理的 ELUATE。"
    )

    python_path = (
        ensure_tool_venv()
    )

    print_info(
        "更新 ELUATE venv 的 "
        "pip / setuptools / wheel..."
    )

    subprocess.check_call(
        [
            str(python_path),
            "-m",
            "pip",
            "install",
            "--upgrade",
            "pip",
            "setuptools",
            "wheel",
        ]
    )

    print_info(
        "開始安裝 ELUATE..."
    )

    subprocess.check_call(
        [
            str(python_path),
            "-m",
            "pip",
            "install",
            "--upgrade",
            ELUATE_PACKAGE,
        ]
    )

    eluate = (
        get_venv_eluate()
    )

    if not eluate.exists():
        raise RuntimeError(
            "ELUATE 安裝完成，"
            "但找不到 CLI：\n"
            f"{eluate}"
        )

    print_ok(
        f"ELUATE 安裝完成：{eluate}"
    )

    return str(eluate)


def ensure_eluate() -> str:
    executable = (
        find_eluate()
    )

    if executable:
        return executable

    return install_eluate()


# ============================================================
# PyTorch / Device
# ============================================================

def python_for_eluate(
        eluate_path: str,
) -> str:

    eluate = (
        Path(eluate_path)
        .resolve()
    )

    if is_windows():
        candidate = (
                eluate.parent
                / "python.exe"
        )
    else:
        candidate = (
                eluate.parent
                / "python"
        )

    if candidate.exists():
        return str(candidate)

    raise RuntimeError(
        "無法找到 ELUATE 所屬 Python：\n"
        f"{eluate}"
    )


def get_torch_status(
        python_executable: str,
) -> dict:

    code = r'''
import json

try:
    import torch

    cuda_available = bool(
        torch.cuda.is_available()
    )

    mps_built = bool(
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_built()
    )

    mps_available = bool(
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    )

    cuda_name = None

    if cuda_available:
        try:
            cuda_name = torch.cuda.get_device_name(0)
        except Exception:
            pass

    mps_name = None

    if mps_available:
        try:
            get_name = getattr(
                torch.backends.mps,
                "get_name",
                None
            )

            if callable(get_name):
                mps_name = get_name()
        except Exception:
            pass

    data = {
        "torch_installed": True,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": cuda_available,
        "cuda_device_name": cuda_name,
        "mps_built": mps_built,
        "mps_available": mps_available,
        "mps_device_name": mps_name,
    }

except Exception as e:

    data = {
        "torch_installed": False,
        "error": f"{type(e).__name__}: {e}",
    }

print(
    json.dumps(
        data,
        ensure_ascii=False
    )
)
'''

    result = subprocess.run(
        [
            python_executable,
            "-c",
            code,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    if result.returncode != 0:
        return {
            "torch_installed": False,
            "error": (
                    result.stderr.strip()
                    or
                    "無法執行 PyTorch 檢查"
            ),
        }

    try:
        last_line = (
            result.stdout
            .strip()
            .splitlines()[-1]
        )

        return json.loads(
            last_line
        )

    except Exception:
        return {
            "torch_installed": False,
            "error": (
                "無法解析 PyTorch 狀態：\n"
                f"{result.stdout}"
            ),
        }


def show_device_status(
        eluate_path: str,
        requested_device: str | None = None,
) -> dict:

    python_executable = (
        python_for_eluate(
            eluate_path
        )
    )

    print_info(
        "檢查 ELUATE Python 環境："
        f"{python_executable}"
    )

    status = get_torch_status(
        python_executable
    )

    if not status.get(
            "torch_installed"
    ):
        raise RuntimeError(
            "ELUATE 環境無法載入 PyTorch：\n"
            f"{status.get('error', '未知錯誤')}"
        )

    print_ok(
        "PyTorch："
        f"{status.get('torch_version')}"
    )

    cuda_available = (
        status.get(
            "cuda_available",
            False,
        )
    )

    mps_available = (
        status.get(
            "mps_available",
            False,
        )
    )

    cuda_runtime = (
        status.get(
            "torch_cuda_version"
        )
    )

    if cuda_runtime:
        print_info(
            "PyTorch CUDA runtime："
            f"{cuda_runtime}"
        )

    if cuda_available:
        print_ok(
            "CUDA 可用："
            f"{status.get('cuda_device_name')}"
        )

    elif is_linux():
        print_warn(
            "CUDA 不可用。"
        )

    if mps_available:
        name = (
                status.get(
                    "mps_device_name"
                )
                or
                "Apple Metal GPU"
        )

        print_ok(
            f"MPS 可用：{name}"
        )

    elif is_macos():
        if status.get(
                "mps_built"
        ):
            print_warn(
                "PyTorch 包含 MPS 支援，"
                "但目前裝置無法使用 MPS。"
            )
        else:
            print_warn(
                "目前 PyTorch 沒有 MPS backend。"
            )

    if requested_device == "cuda":

        if not cuda_available:
            raise RuntimeError(
                "你指定了 --device cuda，"
                "但 ELUATE 的 PyTorch "
                "無法使用 CUDA。"
            )

    if requested_device == "mps":

        if not mps_available:
            raise RuntimeError(
                "你指定了 --device mps，"
                "但目前 PyTorch "
                "無法使用 MPS。"
            )

    if requested_device == "cpu":
        print_info(
            "指定使用 CPU。"
        )

    if requested_device is None:

        if cuda_available:
            print_info(
                "建議運算裝置：CUDA"
            )

        elif mps_available:
            print_info(
                "建議運算裝置：MPS"
            )

        else:
            print_warn(
                "目前沒有偵測到 CUDA / MPS，"
                "預期將使用 CPU。"
            )

    return status


# ============================================================
# Input / Output
# ============================================================

def validate_input(
        input_path: Path,
) -> Path:

    input_path = (
        input_path
        .expanduser()
        .resolve()
    )

    if not input_path.exists():
        raise FileNotFoundError(
            "找不到影片：\n"
            f"{input_path}"
        )

    if not input_path.is_file():
        raise ValueError(
            "輸入路徑不是檔案：\n"
            f"{input_path}"
        )

    extension = (
        input_path
        .suffix
        .lower()
    )

    if (
            extension
            not in
            SUPPORTED_EXTENSIONS
    ):
        print_warn(
            f"副檔名 {extension} "
            "不在常見影片格式清單，"
            "仍會交給 ELUATE / FFmpeg 嘗試。"
        )

    return input_path


def create_output_path(
        input_path: Path,
        output_path: str | None,
) -> Path:

    if output_path:
        output = (
            Path(output_path)
            .expanduser()
            .resolve()
        )
    else:
        output = (
            input_path.with_name(
                f"{input_path.stem}"
                f"_no_bgm"
                f"{input_path.suffix}"
            )
        )

    if output == input_path:
        raise ValueError(
            "輸出檔案不能與輸入檔案相同。"
        )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    return output


def create_temp_output_path(
        output_path: Path,
) -> Path:

    return (
        output_path.with_name(
            f"{output_path.stem}"
            ".processing"
            f"{output_path.suffix}"
        )
    )


# ============================================================
# ELUATE
# ============================================================

def print_checkpoint_repair_hint(
        checkpoint: str,
) -> None:

    print()
    print_warn(
        "ELUATE checkpoint "
        "可能下載不完整或 SHA256 不符。"
    )

    model_path = (
            ELUATE_MODELS_DIR
            / f"checkpoint-{checkpoint}.ckpt"
    )

    print_info(
        "可以嘗試刪除模型後重新下載："
    )

    print()

    print(
        f"  rm -f {model_path}"
    )

    print(
        f"  {get_venv_eluate()} setup"
    )

    print()


def remove_bgm(
        input_path: Path,
        output_path: Path,
        *,
        eluate_path: str,
        checkpoint: str = "multi",
        device: str | None = None,
        force: bool = False,
) -> None:

    temp_output = (
        create_temp_output_path(
            output_path
        )
    )

    if temp_output.exists():
        temp_output.unlink()

    command = [
        eluate_path,
        str(input_path),
        "-o",
        str(temp_output),
    ]

    if checkpoint != "multi":
        command.extend(
            [
                "--checkpoint",
                checkpoint,
            ]
        )

    if device:
        command.extend(
            [
                "--device",
                device,
            ]
        )

    if force:
        command.append(
            "--force"
        )

    print_header(
        "開始 AI BGM 移除"
    )

    print_info(
        f"輸入：{input_path}"
    )

    print_info(
        f"輸出：{output_path}"
    )

    print_info(
        f"模型：BandIt v2 / {checkpoint}"
    )

    print_info(
        "Device："
        f"{device if device else '自動偵測'}"
    )

    print()

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    collected_lines: list[str] = []

    assert process.stdout is not None

    for line in process.stdout:
        print(
            line,
            end="",
        )

        collected_lines.append(
            line
        )

        if len(collected_lines) > 300:
            collected_lines.pop(0)

    return_code = (
        process.wait()
    )

    output_text = (
        "".join(
            collected_lines
        )
    )

    if return_code != 0:

        if temp_output.exists():
            try:
                temp_output.unlink()
            except OSError:
                pass

        lower_output = (
            output_text.lower()
        )

        if (
                "sha256 mismatch"
                in lower_output
                or
                "checkpoint integrity check failed"
                in lower_output
                or
                "checkpoint verification failed"
                in lower_output
        ):
            print_checkpoint_repair_hint(
                checkpoint
            )

            raise RuntimeError(
                "ELUATE checkpoint 驗證失敗。"
            )

        raise RuntimeError(
            "ELUATE 執行失敗，"
            f"exit code = {return_code}"
        )

    if not temp_output.exists():
        raise RuntimeError(
            "ELUATE 顯示執行完成，"
            "但找不到暫存輸出影片：\n"
            f"{temp_output}"
        )

    if output_path.exists():
        print_warn(
            "輸出檔已存在，"
            "處理成功後將覆蓋：\n"
            f"{output_path}"
        )

        output_path.unlink()

    temp_output.replace(
        output_path
    )


# ============================================================
# File size
# ============================================================

def format_size(
        size: int,
) -> str:

    units = [
        "B",
        "KB",
        "MB",
        "GB",
        "TB",
    ]

    value = float(size)

    for unit in units:

        if value < 1024:
            return (
                f"{value:.2f} {unit}"
            )

        value /= 1024

    return (
        f"{value:.2f} PB"
    )


# ============================================================
# Main process
# ============================================================

def process_video(
        input_file: str,
        output_file: str | None = None,
        checkpoint: str = "multi",
        device: str | None = None,
        force: bool = False,
) -> Path:

    print_header(
        "Remove Video BGM"
    )

    show_platform_info()

    validate_platform()

    input_path = validate_input(
        Path(input_file)
    )

    output_path = (
        create_output_path(
            input_path,
            output_file,
        )
    )

    print_info(
        "檢查 FFmpeg..."
    )

    check_ffmpeg()

    print_info(
        "檢查 ELUATE..."
    )

    eluate = ensure_eluate()

    print_ok(
        f"ELUATE：{eluate}"
    )

    show_device_status(
        eluate,
        requested_device=device,
    )

    remove_bgm(
        input_path,
        output_path,
        eluate_path=eluate,
        checkpoint=checkpoint,
        device=device,
        force=force,
    )

    print_header(
        "處理完成"
    )

    print_ok(
        "輸出影片：\n"
        f"{output_path}"
    )

    print_info(
        "原始大小："
        + format_size(
            input_path
            .stat()
            .st_size
        )
    )

    print_info(
        "輸出大小："
        + format_size(
            output_path
            .stat()
            .st_size
        )
    )

    print()

    print(
        "音訊處理結果："
    )

    print(
        "  Speech  → 保留"
    )

    print(
        "  SFX     → 保留"
    )

    print(
        "  Music   → 移除"
    )

    return output_path


# ============================================================
# CLI
# ============================================================

def create_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description=(
            "使用 ELUATE / BandIt v2 "
            "移除影片 BGM，"
            "盡可能保留人聲與音效。"
        )
    )

    parser.add_argument(
        "input",
        help="輸入影片",
    )

    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help=(
            "輸出影片。"
            "未指定時自動產生 "
            "*_no_bgm.<ext>"
        ),
    )

    parser.add_argument(
        "--checkpoint",
        default="multi",
        choices=[
            "multi",
            "eng",
            "deu",
            "fra",
            "spa",
            "cmn",
            "fao",
        ],
        help=(
            "BandIt checkpoint。"
            "日文遊戲建議維持 multi。"
        ),
    )

    parser.add_argument(
        "--device",
        default=None,
        choices=[
            "cuda",
            "cpu",
            "mps",
        ],
        help=(
            "指定運算裝置。"
            "NVIDIA/Linux/WSL 使用 cuda；"
            "支援的 macOS GPU 使用 mps；"
            "否則使用 cpu。"
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "傳遞 ELUATE --force，"
            "跳過影片長度 / "
            "磁碟空間預檢查。"
        ),
    )

    return parser


# ============================================================
# Entry Point
# ============================================================

def main() -> int:

    parser = (
        create_parser()
    )

    args = (
        parser.parse_args()
    )

    try:

        process_video(
            input_file=args.input,
            output_file=args.output,
            checkpoint=args.checkpoint,
            device=args.device,
            force=args.force,
        )

        return 0

    except KeyboardInterrupt:

        print()

        print_error(
            "使用者取消處理。"
        )

        return 130

    except Exception as e:

        print()

        print_header(
            "處理失敗"
        )

        print_error(
            f"{type(e).__name__}: {e}"
        )

        print()

        import traceback

        traceback.print_exc()

        return 1


if __name__ == "__main__":
    sys.exit(main())
