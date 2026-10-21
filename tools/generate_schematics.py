"""生成石材标本照与显微照片的**示意图**。

用法：
    .venv/bin/python tools/generate_schematics.py

重要：本脚本产出的是程序生成的示意图，不是摄影照片，也不是显微摄影。
每张图都烙有「示意图」水印，且 data/images.csv 中 is_measured 一律记为「否」。
目的是先把图像字段、目录结构与展示逻辑跑通，真实照片到位后直接替换文件即可。

设计依据
--------
- 标本照：底色取自 stones.csv 的 color 字段，矿物斑点的颗粒大小与配色
  按 rock_type 与矿物含量决定
- 显微照片：模拟正交偏光下的薄片，颗粒边界清晰、干涉色按矿物种类赋色，
  颗粒粒径按岩石类型区分（玄武岩隐晶质、花岗岩中粗粒、大理岩等粒）
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from stonelab.data import load_minerals, load_stones, minerals_of  # noqa: E402

OUT_DIR = ROOT / "data" / "images"
FONT_PATH = "/System/Library/Fonts/PingFang.ttc"

HAND_SIZE = 420
MICRO_W, MICRO_H = 540, 400

# 商品名颜色 → 近似 RGB
COLOR_MAP = {
    "灰白": (186, 184, 176),
    "深灰": (92, 92, 96),
    "纯黑": (38, 38, 40),
    "紫红": (126, 58, 62),
    "白底灰纹": (226, 224, 220),
    "乳白": (236, 232, 222),
    "米黄": (212, 190, 148),
    "青灰": (126, 134, 132),
    "浅红": (174, 106, 94),
    "黑灰": (76, 78, 80),
    "纯白": (240, 239, 234),
}

# 矿物 → 手标本上的自然色
HAND_MINERAL_COLOR = {
    "石英": (206, 206, 208),
    "钾长石": (216, 196, 178),
    "斜长石": (222, 220, 212),
    "长石": (218, 208, 194),
    "方解石": (238, 236, 230),
    "白云石": (232, 228, 220),
    "黑云母": (58, 46, 38),
    "白云母": (206, 200, 186),
    "绢云母": (198, 194, 180),
    "角闪石": (64, 74, 58),
    "辉石": (78, 84, 70),
    "橄榄石": (128, 136, 86),
    "磁铁矿": (30, 30, 32),
    "赤铁矿": (96, 48, 42),
    "石墨": (24, 24, 26),
    "绿泥石": (86, 108, 84),
    "蛇纹石": (140, 152, 110),
    "黏土矿物": (124, 118, 104),
    "岩屑": (136, 132, 122),
    "玻璃质": (58, 54, 50),
    "副矿物": (72, 72, 72),
}

# 矿物 → 正交偏光下的干涉色（模拟）
XPOL_MINERAL_COLOR = {
    "石英": (198, 198, 204),
    "钾长石": (226, 218, 200),
    "斜长石": (236, 233, 226),
    "长石": (230, 225, 212),
    "方解石": (246, 238, 228),
    "白云石": (240, 232, 222),
    "黑云母": (92, 66, 42),
    "白云母": (216, 206, 176),
    "绢云母": (202, 196, 170),
    "角闪石": (82, 104, 70),
    "辉石": (108, 120, 94),
    "橄榄石": (168, 180, 108),
    "磁铁矿": (22, 22, 25),
    "赤铁矿": (88, 42, 38),
    "石墨": (18, 18, 20),
    "绿泥石": (78, 110, 86),
    "蛇纹石": (148, 170, 118),
    "黏土矿物": (118, 112, 96),
    "岩屑": (138, 132, 120),
    "玻璃质": (56, 52, 48),
    "副矿物": (68, 68, 68),
}

# 岩石类型 → 典型颗粒粒径（像素，显微视域内）
GRAIN_SIZE = {
    "花岗岩": 30,
    "辉长岩": 15,
    "辉绿岩": 9,
    "大理岩": 34,
    "石灰岩（重结晶）": 24,
    "石灰岩": 12,
    "砂岩": 17,
    "玄武岩": 6,
    "板岩": 9,
}

# 岩石类型 → 显微视域实际宽度（mm），用于比例尺
FIELD_WIDTH_MM = {
    "花岗岩": 5.0,
    "辉长岩": 3.0,
    "辉绿岩": 2.5,
    "大理岩": 5.0,
    "石灰岩（重结晶）": 4.0,
    "石灰岩": 2.5,
    "砂岩": 3.5,
    "玄武岩": 2.0,
    "板岩": 2.5,
}


def _font(size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(FONT_PATH, size)
    except OSError:
        return ImageFont.load_default()


def _weighted_palette(minerals) -> tuple[list[tuple], list[float]]:
    """按矿物含量生成抽签用的颜色池与权重。"""
    colors, weights = [], []
    for _, row in minerals.iterrows():
        name = row["mineral_name"]
        pct = float(row["content_pct"]) if row["content_pct"] == row["content_pct"] else 0.0
        if pct <= 0:
            continue
        colors.append((name, pct))
        weights.append(pct)
    return colors, weights


def _irregular_polygon(cx, cy, radius, rng, vertices=9, jitter=0.34):
    angles = np.sort(rng.uniform(0, 2 * math.pi, vertices))
    points = []
    for angle in angles:
        r = radius * rng.uniform(1 - jitter, 1 + jitter)
        points.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
    return points


def _watermark(img: Image.Image, text: str, size: int = 15) -> None:
    """在右下角烙上水印，确保图像脱离数据库后也不会被误认为实测照片。"""
    draw = ImageDraw.Draw(img, "RGBA")
    font = _font(size)
    bbox = draw.textbbox((0, 0), text, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad = 8
    x = img.width - w - pad * 2
    y = img.height - h - pad * 2

    draw.rectangle([x - pad, y - pad, img.width - 2, img.height - 2], fill=(0, 0, 0, 110))
    draw.text((x, y - 2), text, font=font, fill=(255, 255, 255, 225))


def make_hand_sample(row, minerals, out_path: Path) -> None:
    """生成手标本（板材）示意图。"""
    rng = np.random.default_rng(abs(hash(row["stone_id"])) % (2**32))
    base = COLOR_MAP.get(row["color"], (180, 178, 172))
    grain = GRAIN_SIZE.get(row["rock_type"], 18)

    img = Image.new("RGB", (HAND_SIZE, HAND_SIZE), base)
    draw = ImageDraw.Draw(img, "RGBA")

    palette, weights = _weighted_palette(minerals)
    if not palette:
        palette, weights = [("副矿物", 100.0)], [100.0]

    names = [p[0] for p in palette]
    probs = np.array(weights, dtype=float)
    probs = probs / probs.sum()

    # 手标本视域约相当于 100 mm 板材，3 mm 的矿物颗粒只占十几个像素，
    # 因此斑点必须比显微颗粒小得多，否则会画成拼贴画
    spot = max(grain * 0.48, 4.0)
    count = int((HAND_SIZE / spot) ** 2 * 1.9)

    for _ in range(count):
        cx = rng.uniform(0, HAND_SIZE)
        cy = rng.uniform(0, HAND_SIZE)
        idx = int(rng.choice(len(names), p=probs))
        name = names[idx]
        rgb = HAND_MINERAL_COLOR.get(name, (150, 148, 142))

        radius = spot * rng.uniform(0.55, 1.2)
        jitter = int(rng.integers(-12, 13))
        color = tuple(max(0, min(255, c + jitter)) for c in rgb)
        alpha = int(rng.integers(205, 255))

        draw.polygon(_irregular_polygon(cx, cy, radius, rng, vertices=7), fill=color + (alpha,))

    # 大理石类加几道细脉
    if "大理岩" in str(row["rock_type"]):
        for _ in range(int(rng.integers(3, 7))):
            pts = []
            x = rng.uniform(0, HAND_SIZE)
            y = 0.0
            while y < HAND_SIZE:
                x += rng.normal(0, 6)
                pts.append((x, y))
                y += rng.uniform(14, 28)
            draw.line(pts, fill=(120, 118, 112, 70), width=int(rng.integers(1, 3)))

    img = img.filter(ImageFilter.GaussianBlur(0.6))

    # 整体色调校正：红砂岩的红来自铁质胶结物对颗粒的染色，黑色石材来自暗色矿物弥漫，
    # 单纯按矿物本色叠加会显得偏灰白。深色石材需要更强的混合比例，
    # 否则「山西黑」（斜长石占 52%）会渲染成中灰色。
    tint = COLOR_MAP.get(row["color"], base)
    tint_lum = 0.299 * tint[0] + 0.587 * tint[1] + 0.114 * tint[2]
    strength = 0.30 if tint_lum >= 120 else 0.30 + 0.55 * (120 - tint_lum) / 120
    img = Image.blend(img, Image.new("RGB", img.size, tint), strength)

    # 细微噪点，模拟石材表面的颗粒感
    noise = rng.normal(0, 5, (HAND_SIZE, HAND_SIZE, 3))
    arr = np.clip(np.array(img, dtype=float) + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)

    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, HAND_SIZE - 1, HAND_SIZE - 1], outline=(90, 90, 90), width=2)

    _watermark(img, "示意图 · 非实物照片")
    img.save(out_path, optimize=True)


def make_micrograph(row, minerals, out_path: Path) -> None:
    """生成正交偏光薄片示意图。"""
    rng = np.random.default_rng((abs(hash(row["stone_id"])) + 7) % (2**32))
    grain = GRAIN_SIZE.get(row["rock_type"], 15)
    field_mm = FIELD_WIDTH_MM.get(row["rock_type"], 3.0)

    palette, weights = _weighted_palette(minerals)
    if not palette:
        palette, weights = [("副矿物", 100.0)], [100.0]

    names = [p[0] for p in palette]
    probs = np.array(weights, dtype=float)
    probs = probs / probs.sum()

    # 背景取主导矿物干涉色的暗调。若用纯黑打底，颗粒间的缝隙会露出黑三角，
    # 看起来像孔隙，而真实薄片是紧密镶嵌的
    dominant = names[int(np.argmax(probs))]
    dom_rgb = XPOL_MINERAL_COLOR.get(dominant, (150, 148, 142))
    bg = tuple(int(c * 0.62) for c in dom_rgb)

    img = Image.new("RGB", (MICRO_W, MICRO_H), bg)
    draw = ImageDraw.Draw(img, "RGBA")

    step = max(grain * 0.85, 4.0)
    cols = int(MICRO_W / step) + 2
    rows = int(MICRO_H / step) + 2

    for gy in range(rows):
        for gx in range(cols):
            cx = gx * step + rng.uniform(-step * 0.45, step * 0.45)
            cy = gy * step + rng.uniform(-step * 0.45, step * 0.45)
            idx = int(rng.choice(len(names), p=probs))
            name = names[idx]
            rgb = XPOL_MINERAL_COLOR.get(name, (140, 138, 132))

            radius = grain * rng.uniform(0.62, 1.12)
            jitter = int(rng.integers(-18, 19))
            color = tuple(max(0, min(255, c + jitter)) for c in rgb)

            poly = _irregular_polygon(cx, cy, radius, rng, vertices=int(rng.integers(6, 11)))
            draw.polygon(poly, fill=color + (255,))
            edge = tuple(int(c * 0.72) for c in color)
            draw.line(list(poly) + [poly[0]], fill=edge + (200,), width=1)

            # 斜长石的聚片双晶：在颗粒内部画平行细条
            if name in {"斜长石", "长石"} and radius > 9:
                stripes = int(rng.integers(3, 7))
                for s in range(1, stripes):
                    offset = -radius + 2 * radius * s / stripes
                    draw.line(
                        [(cx + offset, cy - radius), (cx + offset + radius * 0.35, cy + radius)],
                        fill=(60, 58, 54, 90),
                        width=1,
                    )

    img = img.filter(ImageFilter.GaussianBlur(0.4))

    draw = ImageDraw.Draw(img)
    # 视域边缘压暗，模拟显微照明的渐晕
    for i in range(24):
        alpha = int(70 * (1 - i / 24))
        draw.rectangle([i, i, MICRO_W - 1 - i, MICRO_H - 1 - i], outline=(0, 0, 0, alpha))

    # 比例尺：取视域宽度的 1/5，标注实际长度
    bar_mm = round(field_mm / 5, 1)
    bar_px = int(MICRO_W * (bar_mm / field_mm))
    bx, by = 18, MICRO_H - 34
    draw.rectangle([bx - 6, by - 20, bx + bar_px + 8, by + 20], fill=(0, 0, 0, 130))
    draw.line([(bx, by), (bx + bar_px, by)], fill=(255, 255, 255, 240), width=3)
    draw.line([(bx, by - 5), (bx, by + 5)], fill=(255, 255, 255, 240), width=2)
    draw.line([(bx + bar_px, by - 5), (bx + bar_px, by + 5)], fill=(255, 255, 255, 240), width=2)
    draw.text((bx, by - 19), f"{bar_mm} mm", font=_font(14), fill=(255, 255, 255, 240))

    draw.text((MICRO_W - 190, 12), "正交偏光（模拟）", font=_font(13), fill=(255, 255, 255, 200))

    _watermark(img, "示意图 · 非显微摄影")
    img.save(out_path, optimize=True)


def write_index(stones) -> Path:
    """写出图像索引表。索引与文件由同一脚本生成，避免两者脱节。"""
    index_path = ROOT / "data" / "images.csv"
    rows = []

    for _, row in stones.iterrows():
        sid = row["stone_id"]
        field_mm = FIELD_WIDTH_MM.get(row["rock_type"], 3.0)

        rows.append(
            {
                "image_id": f"{sid}-H",
                "stone_id": sid,
                "image_type": "标本照",
                "file_name": f"{sid}_hand.png",
                "caption": f"{row['name_cn']} 手标本示意图（{row['rock_type']}，{row['origin']}）",
                "field_of_view_mm": 100.0,
                "polarization": "",
                "source": "StoneLab 程序生成",
                "license": "本项目生成，可自由使用",
                "is_measured": "否",
            }
        )
        rows.append(
            {
                "image_id": f"{sid}-M",
                "stone_id": sid,
                "image_type": "显微照片",
                "file_name": f"{sid}_micro.png",
                "caption": f"{row['name_cn']} 薄片显微示意图（正交偏光，视域 {field_mm} mm）",
                "field_of_view_mm": field_mm,
                "polarization": "正交偏光（模拟）",
                "source": "StoneLab 程序生成",
                "license": "本项目生成，可自由使用",
                "is_measured": "否",
            }
        )

    pd.DataFrame(rows).to_csv(index_path, index=False, encoding="utf-8")
    return index_path


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    stones = load_stones()
    minerals_all = load_minerals()

    count = 0
    for _, row in stones.iterrows():
        sid = row["stone_id"]
        mins = minerals_of(minerals_all, sid)

        hand_path = OUT_DIR / f"{sid}_hand.png"
        micro_path = OUT_DIR / f"{sid}_micro.png"

        make_hand_sample(row, mins, hand_path)
        make_micrograph(row, mins, micro_path)
        count += 2

        print(f"{sid} {row['name_cn']:<10} {row['rock_type']:<10} -> {hand_path.name}, {micro_path.name}")

    index_path = write_index(stones)
    print(f"\n完成：{count} 张图，输出目录 {OUT_DIR}")
    print(f"索引表：{index_path}")


if __name__ == "__main__":
    main()
