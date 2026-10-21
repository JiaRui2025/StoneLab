"""StoneLab 图像访问层。

图像本身存放在 data/images/，索引存放在 data/images.csv。
索引里的 is_measured 字段是可信度的关键：只有标记为「是」的图才是实测影像，
其余一律是程序生成的示意图，界面必须明确区分。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
IMAGE_DIR = DATA_DIR / "images"
INDEX_FILE = DATA_DIR / "images.csv"

REQUIRED_COLUMNS = [
    "image_id",
    "stone_id",
    "image_type",
    "file_name",
    "caption",
    "field_of_view_mm",
    "polarization",
    "source",
    "license",
    "is_measured",
]

# 界面上固定的展示顺序
IMAGE_TYPE_ORDER = ["标本照", "显微照片"]


def load_images() -> pd.DataFrame:
    """读取图像索引表。"""
    if not INDEX_FILE.exists():
        raise FileNotFoundError(
            f"图像索引不存在：{INDEX_FILE}\n"
            "可运行 .venv/bin/python tools/generate_schematics.py 重新生成。"
        )

    df = pd.read_csv(INDEX_FILE, dtype=str, keep_default_na=False, encoding="utf-8")

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"images.csv 缺少必需字段：{missing}")

    for col in df.columns:
        df[col] = df[col].astype(str).str.strip()

    df["field_of_view_mm"] = pd.to_numeric(df["field_of_view_mm"], errors="coerce")

    return df.reset_index(drop=True)


def images_of(images: pd.DataFrame, stone_id: str) -> pd.DataFrame:
    """取出某样品的图像，按「标本照 → 显微照片」排序。"""
    subset = images[images["stone_id"] == stone_id].copy()
    if subset.empty:
        return subset

    subset["_order"] = subset["image_type"].apply(
        lambda t: IMAGE_TYPE_ORDER.index(t) if t in IMAGE_TYPE_ORDER else len(IMAGE_TYPE_ORDER)
    )
    return subset.sort_values("_order").drop(columns="_order").reset_index(drop=True)


def image_path(file_name: str) -> Path:
    """图像文件的绝对路径。"""
    return IMAGE_DIR / file_name


def is_measured(row: pd.Series) -> bool:
    """该图是否为实测影像。"""
    return str(row.get("is_measured", "")).strip() in {"是", "True", "true", "1"}


def missing_files(images: pd.DataFrame) -> list[str]:
    """索引中存在但磁盘上缺失的文件。用于完整性自检。"""
    return [name for name in images["file_name"] if not image_path(name).exists()]


def count_measured(images: pd.DataFrame) -> tuple[int, int]:
    """返回（实测数量, 总数）。"""
    total = len(images)
    measured = sum(1 for _, row in images.iterrows() if is_measured(row))
    return measured, total
