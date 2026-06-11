from __future__ import annotations

import importlib.util


REQUIRED_PACKAGES = [
    "numpy",
    "pandas",
    "scipy",
    "skimage",
    "torch",
    "gudhi",
]

# porespy/openpnm нужны только для graph-режима scripts/visualize.py
# (извлечение поровой сети). Сегментационный пайплайн работает без них.
OPTIONAL_PACKAGES = [
    "porespy",
    "openpnm",
]


def package_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def check_required_dependencies(packages: list[str] | None = None) -> dict[str, bool]:
    packages = REQUIRED_PACKAGES if packages is None else packages
    status = {name: package_available(name) for name in packages}
    missing = [name for name, ok in status.items() if not ok]
    if missing:
        raise ImportError(
            "Не установлены обязательные зависимости Digital Core: "
            + ", ".join(missing)
            + ". Установите их командой `pip install -r src/requirements.txt`."
        )
    return status


def require_gudhi():
    if not package_available("gudhi"):
        raise ImportError(
            "gudhi нужен для признаков персистентной гомологии. "
            "Установите его командой `pip install gudhi` или `pip install -r src/requirements.txt`."
        )

    import gudhi  # type: ignore

    return gudhi
