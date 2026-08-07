from __future__ import annotations

import json
import os
import platform
from dataclasses import dataclass
from pathlib import Path

APPLE_SILICON_MACHINES = frozenset({"arm64", "aarch64"})
SUPPORTED_SYSTEMS = frozenset({"darwin", "linux", "windows", "win32"})

# DEFAULT_ASR_MODEL for CN: "mlx-community/Qwen3-ASR-1.7B-8bit"
DEFAULT_MLX_EN_ASR_MODEL = "mlx-community/parakeet-tdt-0.6b-v3"
DEFAULT_MLX_CHINESE_ASR_MODEL = "mlx-community/Qwen3-ASR-1.7B-8bit"
DEFAULT_MLX_FALLBACK_ASR_MODEL = "mlx-community/whisper-large-v3-turbo"
DEFAULT_MLX_ALIGNER_MODEL = "mlx-community/Qwen3-ForcedAligner-0.6B-8bit"
DEFAULT_QWEN_ASR_MODEL = "Qwen/Qwen3-ASR-1.7B"
DEFAULT_QWEN_ASR_FALLBACK_MODEL = "Qwen/Qwen3-ASR-0.6B"
DEFAULT_QWEN_ALIGNER_MODEL = "Qwen/Qwen3-ForcedAligner-0.6B"
DEFAULT_EN_ASR_MODEL = DEFAULT_MLX_EN_ASR_MODEL
DEFAULT_CHINESE_ASR_MODEL = DEFAULT_MLX_CHINESE_ASR_MODEL
DEFAULT_FALLBACK_ASR_MODEL = DEFAULT_MLX_FALLBACK_ASR_MODEL
DEFAULT_ALIGNER_MODEL = DEFAULT_MLX_ALIGNER_MODEL
DEFAULT_MLX_TTS_MODEL = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit"
DEFAULT_VOXCPM_TTS_MODEL = "openbmb/VoxCPM2"
DEFAULT_LANGUAGE_ID_MODEL = "beshkenadze/lang-id-voxlingua107-ecapa-mlx"
DEFAULT_TRANSLATION_BACKEND = "py-googletrans"
DEFAULT_ORIGINAL_SUBTITLE_COLOR = "#FFFFFF"
DEFAULT_TRANSLATION_SUBTITLE_COLOR = "#FFA500"


@dataclass(frozen=True)
class RuntimeProfile:
    """Resolved platform profile for model backend and default model decisions."""

    system: str
    machine: str
    asr_backend: str
    tts_backend: str

    def default_asr_model(self, source_lang: str = "en") -> str:
        if self.asr_backend == "qwen":
            return resolve_default_qwen_asr_model()

        normalized = (source_lang or "").strip().lower()
        if normalized.startswith("zh"):
            return DEFAULT_CHINESE_ASR_MODEL
        if normalized.startswith("en"):
            return DEFAULT_EN_ASR_MODEL
        return DEFAULT_FALLBACK_ASR_MODEL

    def default_aligner_model(self) -> str:
        if self.asr_backend == "qwen":
            return resolve_default_qwen_aligner_model()
        return DEFAULT_ALIGNER_MODEL

    def default_tts_model(self) -> str:
        if self.tts_backend == "mlx":
            return resolve_default_mlx_tts_model()
        return resolve_default_voxcpm_tts_model()


def _hf_hub_cache_dir() -> Path:
    explicit_cache = os.environ.get("HF_HUB_CACHE") or os.environ.get("HUGGINGFACE_HUB_CACHE")
    if explicit_cache:
        return Path(explicit_cache).expanduser()
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home).expanduser() / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def _repo_id_to_cache_name(repo_id: str) -> str:
    return "models--" + repo_id.replace("/", "--")


def _snapshot_from_ref(repo_cache_dir: Path, revision: str = "main") -> Path | None:
    ref_path = repo_cache_dir / "refs" / revision
    if not ref_path.exists():
        return None
    commit = ref_path.read_text(encoding="utf-8").strip()
    if not commit:
        return None
    snapshot = repo_cache_dir / "snapshots" / commit
    return snapshot if snapshot.is_dir() else None


def _path_is_available(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def _snapshot_has_complete_weights(snapshot: Path) -> bool:
    index_path = snapshot / "model.safetensors.index.json"
    if index_path.exists():
        try:
            index_payload = json.loads(index_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return False
        shard_names = set((index_payload.get("weight_map") or {}).values())
        return bool(shard_names) and all(_path_is_available(snapshot / name) for name in shard_names)

    direct_weight_names = (
        "model.safetensors",
        "pytorch_model.bin",
        "model.bin",
        "audiovae.pth",
    )
    return any(_path_is_available(snapshot / name) for name in direct_weight_names)


def resolve_hf_cached_snapshot(repo_id: str) -> str | None:
    """Return a complete local Hugging Face snapshot path for ``repo_id`` if one exists."""
    if "/" not in repo_id:
        return None
    repo_cache_dir = _hf_hub_cache_dir() / _repo_id_to_cache_name(repo_id)
    snapshot = _snapshot_from_ref(repo_cache_dir)
    if snapshot is None:
        return None
    if not _snapshot_has_complete_weights(snapshot):
        return None
    return str(snapshot)


def resolve_model_path(model_id_or_path: str) -> str:
    if os.path.exists(os.path.expanduser(model_id_or_path)):
        return os.path.expanduser(model_id_or_path)
    return resolve_hf_cached_snapshot(model_id_or_path) or model_id_or_path


def resolve_first_cached_model(repo_ids: tuple[str, ...]) -> str | None:
    for repo_id in repo_ids:
        cached = resolve_hf_cached_snapshot(repo_id)
        if cached:
            return cached
    return None


def resolve_default_qwen_asr_model() -> str:
    return resolve_first_cached_model((DEFAULT_QWEN_ASR_MODEL, DEFAULT_QWEN_ASR_FALLBACK_MODEL)) or DEFAULT_QWEN_ASR_MODEL


def resolve_default_qwen_aligner_model() -> str:
    return resolve_hf_cached_snapshot(DEFAULT_QWEN_ALIGNER_MODEL) or DEFAULT_QWEN_ALIGNER_MODEL


def resolve_default_mlx_tts_model() -> str:
    cached = resolve_first_cached_model((DEFAULT_MLX_TTS_MODEL, "Qwen/Qwen3-TTS-12Hz-1.7B-Base"))
    return cached or DEFAULT_MLX_TTS_MODEL


def resolve_default_voxcpm_tts_model() -> str:
    return resolve_hf_cached_snapshot(DEFAULT_VOXCPM_TTS_MODEL) or DEFAULT_VOXCPM_TTS_MODEL


def _normalized_system(system: str | None = None) -> str:
    resolved = (system or platform.system()).lower()
    if resolved == "windows":
        return "win32"
    return resolved


def _normalized_machine(machine: str | None = None) -> str:
    return (machine or platform.machine()).lower()


def is_macos_apple_silicon(system: str | None = None, machine: str | None = None) -> bool:
    return _normalized_system(system) == "darwin" and _normalized_machine(machine) in APPLE_SILICON_MACHINES


def is_linux_or_windows(system: str | None = None) -> bool:
    return _normalized_system(system) in {"linux", "win32"}


def is_supported_runtime(system: str | None = None, machine: str | None = None) -> bool:
    return is_macos_apple_silicon(system, machine) or is_linux_or_windows(system)


def select_asr_backend(system: str | None = None, machine: str | None = None) -> str:
    if is_macos_apple_silicon(system, machine):
        return "mlx"
    if is_linux_or_windows(system):
        return "qwen"
    ensure_supported_runtime(system=system, machine=machine)
    raise AssertionError("unreachable")


def select_tts_backend(system: str | None = None, machine: str | None = None) -> str:
    if is_macos_apple_silicon(system, machine):
        return "mlx"
    if is_linux_or_windows(system):
        return "voxcpm"
    ensure_supported_runtime(system=system, machine=machine)
    raise AssertionError("unreachable")


def current_runtime_profile(system: str | None = None, machine: str | None = None) -> RuntimeProfile:
    ensure_supported_runtime(system=system, machine=machine)
    resolved_system = _normalized_system(system)
    resolved_machine = _normalized_machine(machine)
    return RuntimeProfile(
        system=resolved_system,
        machine=resolved_machine,
        asr_backend=select_asr_backend(system=system, machine=machine),
        tts_backend=select_tts_backend(system=system, machine=machine),
    )


def ensure_supported_runtime(system: str | None = None, machine: str | None = None) -> None:
    if is_supported_runtime(system=system, machine=machine):
        return

    resolved_system = system or platform.system()
    resolved_machine = machine or platform.machine()
    raise RuntimeError(
        "pycut supports macOS Apple Silicon, Linux, and Windows "
        f"(got {resolved_system}/{resolved_machine})."
    )
