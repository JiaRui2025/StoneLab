"""StoneLab 数据访问层。

职责：读取 CSV、校验字段完整性、提供搜索与关联查询。
刻意不依赖 Streamlit —— 这样这些函数可以被脚本、测试或未来的 API 直接复用。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

STONE_FILE = DATA_DIR / "stones.csv"
MINERAL_FILE = DATA_DIR / "minerals.csv"

# 主表必需字段。缺少任何一列都直接报错，避免界面上出现空白列却无人察觉。
REQUIRED_STONE_COLUMNS = [
    "stone_id",
    "name_cn",
    "name_en",
    "aliases",
    "rock_class",
    "rock_type",
    "origin",
    "color",
    "porosity_pct",
    "water_absorption_pct",
    "bulk_density_gcm3",
    "compressive_strength_mpa",
    "flexural_strength_mpa",
    "freeze_thaw_loss_pct",
    "xrd_status",
    "data_source",
    "notes",
]

REQUIRED_MINERAL_COLUMNS = [
    "stone_id",
    "mineral_name",
    "mineral_formula",
    "mineral_group",
    "content_pct",
    "determination_method",
]

# 数值字段，读入时强制转成数字，非法值变成 NaN 而不是悄悄当字符串用
NUMERIC_STONE_COLUMNS = [
    "porosity_pct",
    "water_absorption_pct",
    "bulk_density_gcm3",
    "compressive_strength_mpa",
    "flexural_strength_mpa",
    "freeze_thaw_loss_pct",
]


def _read_csv(path: Path, required: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"数据文件不存在：{path}")

    df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name} 缺少必需字段：{missing}")

    # 统一去掉首尾空格，避免因录入时多敲空格导致搜索失败
    for col in df.columns:
        df[col] = df[col].astype(str).str.strip()

    return df


def load_stones() -> pd.DataFrame:
    """读取石材主表。"""
    df = _read_csv(STONE_FILE, REQUIRED_STONE_COLUMNS)

    for col in NUMERIC_STONE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.reset_index(drop=True)


def load_minerals() -> pd.DataFrame:
    """读取矿物组成子表。"""
    df = _read_csv(MINERAL_FILE, REQUIRED_MINERAL_COLUMNS)
    df["content_pct"] = pd.to_numeric(df["content_pct"], errors="coerce")
    return df.reset_index(drop=True)


def search_stones(
    df: pd.DataFrame,
    query: str = "",
    rock_classes: list[str] | None = None,
    porosity_range: tuple[float, float] | None = None,
) -> pd.DataFrame:
    """按关键词 + 岩石大类 + 孔隙率区间筛选。

    关键词会在中文名、英文名、别名、岩石类型、产地五个字段里做不区分大小写的子串匹配。
    多个条件之间是「与」的关系。
    """
    result = df

    query = (query or "").strip()
    if query:
        pattern = query.lower()
        fields = ["name_cn", "name_en", "aliases", "rock_type", "origin"]
        mask = pd.Series(False, index=result.index)
        for field in fields:
            mask |= result[field].str.lower().str.contains(pattern, regex=False, na=False)
        result = result[mask]

    if rock_classes:
        result = result[result["rock_class"].isin(rock_classes)]

    if porosity_range:
        low, high = porosity_range
        result = result[result["porosity_pct"].between(low, high, inclusive="both")]

    return result


def minerals_of(minerals: pd.DataFrame, stone_id: str) -> pd.DataFrame:
    """取出某个样品的矿物组成，按含量从高到低排序。"""
    subset = minerals[minerals["stone_id"] == stone_id].copy()
    return subset.sort_values("content_pct", ascending=False).reset_index(drop=True)


def composition_total(minerals: pd.DataFrame, stone_id: str) -> float:
    """矿物含量合计。用于检查录入是否有遗漏或重复。"""
    return float(minerals_of(minerals, stone_id)["content_pct"].sum())
