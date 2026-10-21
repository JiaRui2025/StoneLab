"""StoneLab XRD 模块 —— 由矿物组成计算理论合成衍射谱。

重要说明
--------
本模块生成的是**理论计算谱**，不是实测谱。做法是：
  1. 从 data/xrd_peaks.csv 取每种矿物的参考峰位与相对强度；
  2. 按该矿物在样品中的含量百分比给峰高加权；
  3. 每个峰用高斯函数展宽，叠加得到总谱；
  4. 归一到最高峰 = 100。

它能回答「这块石材的谱图大概长什么样」「某个 2θ 位置的峰属于谁」这类问题，
但不能替代实际测试：真实谱图的峰形、背景、择优取向、仪器漂移都无法由此反映。
拿到实测数据后，只需替换 synthesize_pattern() 的数据来源，界面与字段无需改动。
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
XRD_PEAK_FILE = DATA_DIR / "xrd_peaks.csv"

# Cu 靶 Kα1 波长（Å）。若改用 Co 靶需同步修改，否则 d 值全部偏移。
CU_KA_WAVELENGTH = 1.5406

REQUIRED_PEAK_COLUMNS = [
    "mineral_name",
    "two_theta",
    "relative_intensity",
    "hkl",
    "reference",
]

# 这些组分在 XRD 中不产生明锐衍射峰，按非晶质/未指定处理，只贡献宽背景
AMORPHOUS_COMPONENTS = {"黏土矿物", "岩屑", "玻璃质", "副矿物"}


def load_peaks() -> pd.DataFrame:
    """读取矿物参考峰位表。"""
    if not XRD_PEAK_FILE.exists():
        raise FileNotFoundError(f"XRD 峰位表不存在：{XRD_PEAK_FILE}")

    df = pd.read_csv(XRD_PEAK_FILE, encoding="utf-8")

    missing = [c for c in REQUIRED_PEAK_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"xrd_peaks.csv 缺少必需字段：{missing}")

    df["mineral_name"] = df["mineral_name"].astype(str).str.strip()
    df["hkl"] = df["hkl"].astype(str).str.strip()
    df["two_theta"] = pd.to_numeric(df["two_theta"], errors="coerce")
    df["relative_intensity"] = pd.to_numeric(df["relative_intensity"], errors="coerce")

    return df.dropna(subset=["two_theta", "relative_intensity"]).reset_index(drop=True)


def d_spacing(two_theta_deg: float) -> float:
    """由 2θ（度）计算晶面间距 d（Å）。布拉格方程：d = λ / (2 sinθ)。"""
    theta = math.radians(two_theta_deg / 2.0)
    if theta <= 0:
        return float("nan")
    return CU_KA_WAVELENGTH / (2.0 * math.sin(theta))


def _gaussian(x: np.ndarray, center: float, fwhm: float) -> np.ndarray:
    """归一化高斯峰形，半高宽为 fwhm（度）。"""
    return np.exp(-4.0 * math.log(2.0) * ((x - center) / fwhm) ** 2)


def available_minerals(peaks: pd.DataFrame) -> set[str]:
    """峰位表里已收录的矿物名集合。"""
    return set(peaks["mineral_name"].unique())


def synthesize_pattern(
    minerals: pd.DataFrame,
    peaks: pd.DataFrame,
    two_theta_range: tuple[float, float] = (5.0, 65.0),
    step: float = 0.02,
    fwhm: float = 0.15,
) -> pd.DataFrame:
    """由样品矿物组成计算理论衍射谱。

    参数
    ----
    minerals : 单个样品的矿物组成（需含 mineral_name 与 content_pct）
    peaks    : load_peaks() 的结果
    two_theta_range : 扫描区间（度）
    step     : 步长（度）
    fwhm     : 半高宽（度），越大峰越宽越矮

    返回
    ----
    DataFrame，列为 two_theta 与 intensity（归一化到 100）
    """
    low, high = two_theta_range
    x = np.arange(low, high + step / 2, step)
    y = np.zeros_like(x)

    if minerals.empty:
        return pd.DataFrame({"two_theta": x, "intensity": y})

    known = available_minerals(peaks)

    for _, row in minerals.iterrows():
        name = row["mineral_name"]
        content = float(row["content_pct"]) if pd.notna(row["content_pct"]) else 0.0
        if content <= 0:
            continue

        weight = content / 100.0

        if name in AMORPHOUS_COMPONENTS or name not in known:
            # 无明锐峰：用一个很宽的包在 25° 附近模拟弥散背景
            y += weight * 0.18 * _gaussian(x, 25.0, 12.0)
            continue

        subset = peaks[peaks["mineral_name"] == name]
        for _, peak in subset.iterrows():
            amplitude = weight * (float(peak["relative_intensity"]) / 100.0)
            y += amplitude * _gaussian(x, float(peak["two_theta"]), fwhm)

    peak_max = float(np.max(y)) if y.size else 0.0
    if peak_max > 0:
        y = y / peak_max * 100.0

    return pd.DataFrame({"two_theta": x, "intensity": y})


def assign_peaks(
    minerals: pd.DataFrame,
    peaks: pd.DataFrame,
    min_relative: float = 8.0,
    limit: int = 24,
) -> pd.DataFrame:
    """列出样品中最强的若干衍射峰，并标注归属矿物与 d 值。

    只保留该样品中实际存在的矿物；相对强度按含量加权后排序，
    因此含量低的矿物即使参考峰很强，也可能排不到前面 —— 这正是定量分析的基本逻辑。
    """
    if minerals.empty:
        return pd.DataFrame(
            columns=["two_theta", "d_angstrom", "hkl", "mineral_name", "relative_intensity", "content_pct", "reference"]
        )

    known = available_minerals(peaks)
    present = minerals[minerals["mineral_name"].isin(known)].copy()
    if present.empty:
        return pd.DataFrame(
            columns=["two_theta", "d_angstrom", "hkl", "mineral_name", "relative_intensity", "content_pct", "reference"]
        )

    merged = present.merge(peaks, on="mineral_name", how="inner")
    merged["scaled_intensity"] = (
        merged["relative_intensity"] * merged["content_pct"] / 100.0
    )

    merged = merged[merged["relative_intensity"] >= min_relative]
    merged = merged.sort_values("scaled_intensity", ascending=False).head(limit)

    merged["d_angstrom"] = merged["two_theta"].apply(d_spacing)

    return merged[
        [
            "two_theta",
            "d_angstrom",
            "hkl",
            "mineral_name",
            "relative_intensity",
            "content_pct",
            "reference",
        ]
    ].reset_index(drop=True)


def amorphous_fraction(minerals: pd.DataFrame) -> float:
    """非晶质或未指定组分的含量合计（%）。"""
    if minerals.empty:
        return 0.0
    mask = minerals["mineral_name"].isin(AMORPHOUS_COMPONENTS)
    return float(minerals.loc[mask, "content_pct"].sum())


def phase_summary(minerals: pd.DataFrame, peaks: pd.DataFrame) -> pd.DataFrame:
    """按矿物汇总：是否有参考峰、含量多少。用于说明谱图覆盖了哪些物相。"""
    if minerals.empty:
        return pd.DataFrame(columns=["mineral_name", "content_pct", "has_reference_peaks"])

    known = available_minerals(peaks)
    out = minerals[["mineral_name", "content_pct"]].copy()
    out["has_reference_peaks"] = out["mineral_name"].isin(known)
    return out.sort_values("content_pct", ascending=False).reset_index(drop=True)
