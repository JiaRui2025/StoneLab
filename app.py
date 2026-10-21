"""StoneLab · 建筑石材研究数据库 — Streamlit 界面层。

运行方式：
    .venv/bin/python -m streamlit run app.py
"""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from stonelab.data import (
    composition_total,
    load_minerals,
    load_stones,
    minerals_of,
    search_stones,
)
from stonelab.images import (
    count_measured,
    image_path,
    images_of,
    is_measured,
    load_images,
    missing_files,
)
from stonelab.xrd import (
    amorphous_fraction,
    assign_peaks,
    load_peaks,
    phase_summary,
    synthesize_pattern,
)

st.set_page_config(
    page_title="StoneLab · 建筑石材研究数据库",
    page_icon="🪨",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def _stones() -> pd.DataFrame:
    return load_stones()


@st.cache_data(show_spinner=False)
def _minerals() -> pd.DataFrame:
    return load_minerals()


@st.cache_data(show_spinner=False)
def _peaks() -> pd.DataFrame:
    return load_peaks()


@st.cache_data(show_spinner=False)
def _images() -> pd.DataFrame:
    return load_images()


def _fmt(value: float | None, unit: str = "", digits: int = 2) -> str:
    """把可能为空的数值格式化成人能读的字符串。"""
    if value is None or pd.isna(value):
        return "—"
    return f"{value:.{digits}f}{unit}"


def render_header() -> None:
    st.title("StoneLab")
    st.caption("建筑石材研究数据库 · v0.2 · 本地运行")


def render_sidebar(stones: pd.DataFrame) -> tuple[str, list[str], tuple[float, float]]:
    st.sidebar.header("检索条件")

    query = st.sidebar.text_input(
        "石材名称 / 别名 / 产地",
        placeholder="例如：白麻、汉白玉、山东",
        help="在中文名、英文名、别名、岩石类型、产地中做模糊匹配",
    )

    all_classes = sorted(stones["rock_class"].dropna().unique().tolist())
    rock_classes = st.sidebar.multiselect(
        "岩石大类",
        options=all_classes,
        default=all_classes,
    )

    p_min = float(stones["porosity_pct"].min())
    p_max = float(stones["porosity_pct"].max())
    porosity_range = st.sidebar.slider(
        "孔隙率区间 (%)",
        min_value=0.0,
        max_value=float(p_max) + 0.5,
        value=(0.0, float(p_max) + 0.5),
        step=0.1,
    )

    st.sidebar.divider()
    st.sidebar.metric("收录样品", f"{len(stones)} 块")
    st.sidebar.metric("矿物记录", f"{len(_minerals())} 条")
    st.sidebar.metric("衍射峰位", f"{len(_peaks())} 条")

    images = _images()
    measured, total = count_measured(images)
    st.sidebar.metric("图像", f"{total} 张", delta=f"实测 {measured} 张", delta_color="off")

    st.sidebar.caption(
        "物理性能为文献参考值，XRD 谱与图像均为理论计算/程序生成，非实测数据。"
    )

    return query, rock_classes, porosity_range


def render_results(results: pd.DataFrame, query: str) -> str | None:
    st.subheader(f"检索结果 · {len(results)} 条")

    if results.empty:
        st.warning(
            f"没有匹配「{query}」的记录。可以试试只输入两个字，或放宽岩石大类与孔隙率筛选。"
        )
        return None

    table = results[
        [
            "stone_id",
            "name_cn",
            "rock_type",
            "rock_class",
            "origin",
            "porosity_pct",
            "water_absorption_pct",
            "bulk_density_gcm3",
        ]
    ].rename(
        columns={
            "stone_id": "编号",
            "name_cn": "名称",
            "rock_type": "岩石类型",
            "rock_class": "大类",
            "origin": "产地",
            "porosity_pct": "孔隙率 (%)",
            "water_absorption_pct": "吸水率 (%)",
            "bulk_density_gcm3": "体积密度 (g/cm³)",
        }
    )

    st.dataframe(table, hide_index=True)

    labels = results.apply(
        lambda r: f"{r['name_cn']}　·　{r['stone_id']}　·　{r['rock_type']}",
        axis=1,
    ).tolist()

    chosen = st.selectbox("选择样品查看详情", options=labels, index=0)
    return results.iloc[labels.index(chosen)]["stone_id"]


def render_metrics(row: pd.Series) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("孔隙率", _fmt(row["porosity_pct"], " %"))
    c2.metric("吸水率", _fmt(row["water_absorption_pct"], " %"))
    c3.metric("体积密度", _fmt(row["bulk_density_gcm3"], " g/cm³"))
    c4.metric("抗压强度", _fmt(row["compressive_strength_mpa"], " MPa", 0))


def render_overview(row: pd.Series) -> None:
    left, right = st.columns([1, 1])

    with left:
        st.subheader("基本信息")
        profile = pd.DataFrame(
            {
                "字段": [
                    "样品编号", "中文名", "英文名", "别名", "岩石大类", "岩石类型",
                    "产地", "颜色", "抗弯强度", "冻融质量损失", "数据来源",
                ],
                "内容": [
                    row["stone_id"], row["name_cn"], row["name_en"],
                    row["aliases"].replace("；", "、") or "—",
                    row["rock_class"], row["rock_type"], row["origin"], row["color"],
                    _fmt(row["flexural_strength_mpa"], " MPa"),
                    _fmt(row["freeze_thaw_loss_pct"], " %"),
                    row["data_source"],
                ],
            }
        )
        st.dataframe(profile, hide_index=True)

        if row["notes"]:
            st.caption(f"备注：{row['notes']}")

    with right:
        st.subheader("岩石学分类")
        st.markdown(f"**{row['rock_class']}** → {row['rock_type']}")
        st.caption(f"产地：{row['origin']}　|　颜色：{row['color']}")

        st.markdown("**耐候性参考**")
        st.write(
            f"冻融质量损失 {_fmt(row['freeze_thaw_loss_pct'], ' %')}　"
            f"抗弯强度 {_fmt(row['flexural_strength_mpa'], ' MPa')}"
        )


def render_minerals(stone_id: str) -> None:
    minerals = minerals_of(_minerals(), stone_id)

    if minerals.empty:
        st.info("该样品尚未录入矿物组成数据。")
        return

    total = composition_total(_minerals(), stone_id)
    st.caption(f"共 {len(minerals)} 种矿物，含量合计 {total:.1f}%")

    chart = (
        alt.Chart(minerals)
        .mark_bar()
        .encode(
            x=alt.X("content_pct:Q", title="含量 (%)", scale=alt.Scale(domain=[0, 100])),
            y=alt.Y("mineral_name:N", sort="-x", title=""),
            color=alt.Color("mineral_group:N", title="矿物类别"),
            tooltip=[
                alt.Tooltip("mineral_name:N", title="矿物"),
                alt.Tooltip("mineral_formula:N", title="化学式"),
                alt.Tooltip("mineral_group:N", title="类别"),
                alt.Tooltip("content_pct:Q", title="含量 (%)"),
                alt.Tooltip("determination_method:N", title="测定方法"),
            ],
        )
        .properties(height=max(180, 28 * len(minerals)))
    )
    st.altair_chart(chart)

    detail = minerals.rename(
        columns={
            "mineral_name": "矿物",
            "mineral_formula": "化学式",
            "mineral_group": "类别",
            "content_pct": "含量 (%)",
            "determination_method": "测定方法",
        }
    )[["矿物", "化学式", "类别", "含量 (%)", "测定方法"]]
    st.dataframe(detail, hide_index=True)


def _per_mineral_patterns(
    minerals: pd.DataFrame,
    peaks: pd.DataFrame,
    two_theta_range: tuple[float, float],
    fwhm: float,
) -> pd.DataFrame:
    """分别计算每种矿物单独的理论谱，各自归一化，便于比较峰位。"""
    frames = []
    for name in minerals["mineral_name"]:
        single = minerals[minerals["mineral_name"] == name]
        if single.empty:
            continue
        pattern = synthesize_pattern(single, peaks, two_theta_range=two_theta_range, fwhm=fwhm)
        pattern["mineral_name"] = name
        frames.append(pattern)

    if not frames:
        return pd.DataFrame(columns=["two_theta", "intensity", "mineral_name"])

    return pd.concat(frames, ignore_index=True)


def render_images(stone_id: str, row: pd.Series) -> None:
    images = images_of(_images(), stone_id)

    if images.empty:
        st.info(
            "该样品尚未登记图像。把照片放入 `data/images/`，"
            "并在 `data/images.csv` 中登记一行即可。"
        )
        return

    if any(not is_measured(r) for _, r in images.iterrows()):
        st.warning(
            "以下图像为**程序生成的示意图**：按矿物组成与岩石类型渲染，"
            "用于说明结构特征，**不是该样品的实物照片，也不是显微摄影**。"
            "图中已烙印水印，索引表 `is_measured` 记为「否」。",
            icon="⚠️",
        )

    cols = st.columns(len(images))
    for col, (_, img_row) in zip(cols, images.iterrows()):
        with col:
            path = image_path(img_row["file_name"])
            if path.exists():
                st.image(str(path), caption=img_row["caption"])
            else:
                st.info(
                    f"缺少文件 `{img_row['file_name']}`\n\n"
                    f"预期路径：`data/images/{img_row['file_name']}`"
                )

            meta_bits = [f"来源：{img_row['source']}"]
            if pd.notna(img_row["field_of_view_mm"]):
                meta_bits.append(f"视域：{img_row['field_of_view_mm']:g} mm")
            if img_row["polarization"]:
                meta_bits.append(img_row["polarization"])
            meta_bits.append(
                "实测影像" if is_measured(img_row) else "示意图（非实测）"
            )
            st.caption("　|　".join(meta_bits))

    with st.expander("图像元数据"):
        table = images[
            ["image_id", "image_type", "file_name", "field_of_view_mm",
             "polarization", "source", "license", "is_measured"]
        ].rename(
            columns={
                "image_id": "图像编号",
                "image_type": "类型",
                "file_name": "文件名",
                "field_of_view_mm": "视域 (mm)",
                "polarization": "偏光条件",
                "source": "来源",
                "license": "许可",
                "is_measured": "是否实测",
            }
        )
        st.dataframe(table, hide_index=True)

    with st.expander("替换为真实照片的方法"):
        st.markdown(
            f"""
            1. 把实物照片命名为 `{stone_id}_hand.png`，显微照片命名为 `{stone_id}_micro.png`，
               放入 `data/images/`（覆盖同名示意图即可，无需改代码）
            2. 在 `data/images.csv` 中把对应行的 `is_measured` 改为「是」，
               `source` 改为实际来源（如「本项目显微摄影」），`license` 填写授权方式
            3. 显微照片建议在 `polarization` 字段注明「单偏光」或「正交偏光」，
               并在 `field_of_view_mm` 填写实际视域宽度

            界面会自动撤掉示意图警告，因为判断依据是索引表字段，不是文件名。
            """
        )


def render_xrd(stone_id: str, row: pd.Series) -> None:
    minerals = minerals_of(_minerals(), stone_id)
    peaks = _peaks()

    if minerals.empty:
        st.info("该样品尚未录入矿物组成，无法计算衍射谱。")
        return

    st.warning(
        "以下为**理论计算谱**：按矿物组成与参考峰位合成，非实测数据。"
        "真实谱图还需考虑仪器漂移、择优取向与样品制备影响。",
        icon="⚠️",
    )

    c1, c2 = st.columns(2)
    with c1:
        two_theta_range = st.slider(
            "2θ 扫描范围 (°)",
            min_value=5,
            max_value=70,
            value=(5, 65),
            step=1,
            key=f"range_{stone_id}",
        )
    with c2:
        fwhm = st.slider(
            "半高宽 FWHM (°)",
            min_value=0.05,
            max_value=0.60,
            value=0.15,
            step=0.01,
            key=f"fwhm_{stone_id}",
            help="峰宽越大，代表晶粒越细或结晶度越低",
        )

    pattern = synthesize_pattern(
        minerals, peaks, two_theta_range=two_theta_range, fwhm=fwhm
    )

    assigned = assign_peaks(minerals, peaks)
    top = assigned.head(5).copy()
    top["label"] = top["hkl"] + " " + top["mineral_name"]

    base = alt.Chart(pattern)
    area = base.mark_area(opacity=0.22, color="#4c78a8")
    line = base.mark_line(color="#1f4e79", strokeWidth=1.4)
    curve = (area + line).encode(
        x=alt.X("two_theta:Q", title="2θ (°, Cu Kα λ=1.5406 Å)"),
        y=alt.Y("intensity:Q", title="相对强度 (归一化)"),
        tooltip=[
            alt.Tooltip("two_theta:Q", title="2θ (°)", format=".2f"),
            alt.Tooltip("intensity:Q", title="强度", format=".1f"),
        ],
    )

    rules = (
        alt.Chart(top)
        .mark_rule(strokeDash=[3, 3], color="#b23b3b", opacity=0.55)
        .encode(x="two_theta:Q")
    )
    labels = (
        alt.Chart(top)
        .mark_text(angle=90, dx=-4, dy=8, align="right", fontSize=9, color="#8a2b2b")
        .encode(x="two_theta:Q", y=alt.value(4), text="label:N")
    )

    st.altair_chart((curve + rules + labels).properties(height=320))

    st.subheader("主要衍射峰归属")
    st.caption(
        f"按「参考强度 × 矿物含量」排序，取前 {len(assigned)} 个峰。"
        f"非晶质或未指定组分合计 {amorphous_fraction(minerals):.1f}%，不产生明锐峰。"
    )

    if assigned.empty:
        st.info("该样品的矿物均无参考峰位记录，无法归属。")
    else:
        table = assigned.copy()
        table["two_theta"] = table["two_theta"].map(lambda v: f"{v:.2f}")
        table["d_angstrom"] = table["d_angstrom"].map(lambda v: f"{v:.3f}")
        table["content_pct"] = table["content_pct"].map(lambda v: f"{v:.1f}")
        table = table.rename(
            columns={
                "two_theta": "2θ (°)",
                "d_angstrom": "d (Å)",
                "hkl": "晶面 (hkl)",
                "mineral_name": "归属矿物",
                "relative_intensity": "参考强度",
                "content_pct": "含量 (%)",
                "reference": "参考卡片",
            }
        )
        st.dataframe(table, hide_index=True)

    with st.expander("按矿物分解查看"):
        st.caption("每种矿物单独归一化，便于看清各自的峰位分布。")
        per = _per_mineral_patterns(minerals, peaks, two_theta_range, fwhm)
        if per.empty:
            st.info("无可分解的物相。")
        else:
            split_chart = (
                alt.Chart(per)
                .mark_line(opacity=0.75, strokeWidth=1.1)
                .encode(
                    x=alt.X("two_theta:Q", title="2θ (°)"),
                    y=alt.Y("intensity:Q", title="相对强度"),
                    color=alt.Color("mineral_name:N", title="矿物"),
                    tooltip=[
                        alt.Tooltip("mineral_name:N", title="矿物"),
                        alt.Tooltip("two_theta:Q", title="2θ (°)", format=".2f"),
                        alt.Tooltip("intensity:Q", title="强度", format=".1f"),
                    ],
                )
                .properties(height=280)
            )
            st.altair_chart(split_chart)

    with st.expander("物相覆盖情况"):
        coverage = phase_summary(minerals, peaks)
        coverage["has_reference_peaks"] = coverage["has_reference_peaks"].map(
            {True: "有参考峰位", False: "无参考峰位"}
        )
        coverage = coverage.rename(
            columns={
                "mineral_name": "矿物",
                "content_pct": "含量 (%)",
                "has_reference_peaks": "峰位状态",
            }
        )
        st.dataframe(coverage, hide_index=True)
        st.caption(
            "「无参考峰位」的组分（黏土矿物、岩屑、玻璃质、副矿物）在谱图中只表现为弥散背景。"
        )

    with st.expander("实测数据接入说明"):
        st.markdown(
            """
            当前谱图由 `stonelab/xrd.py` 的 `synthesize_pattern()` 计算得出。接入实测数据时：

            1. 将原始谱图按样品编号存入 `data/xrd/<stone_id>.xy`（两列：2θ 与强度）
            2. 在 `stones.csv` 的 `xrd_status` 字段标记为「已测」
            3. 把 `render_xrd()` 的数据来源从 `synthesize_pattern()` 换为读取实测文件

            界面、峰位归属表与物相覆盖逻辑均可复用，不需要重写。
            """
        )


def render_detail(stones: pd.DataFrame, stone_id: str) -> None:
    row = stones.loc[stones["stone_id"] == stone_id].iloc[0]

    st.divider()
    st.header(f"{row['name_cn']}　{row['name_en']}")
    render_metrics(row)

    tab_overview, tab_images, tab_minerals, tab_xrd = st.tabs(
        ["概览", "图像", "矿物组成", "XRD 衍射"]
    )

    with tab_overview:
        render_overview(row)

    with tab_images:
        render_images(stone_id, row)

    with tab_minerals:
        render_minerals(stone_id)

    with tab_xrd:
        render_xrd(stone_id, row)


def main() -> None:
    render_header()

    stones = _stones()
    query, rock_classes, porosity_range = render_sidebar(stones)

    results = search_stones(
        stones,
        query=query,
        rock_classes=rock_classes,
        porosity_range=porosity_range,
    )

    selected = render_results(results, query)
    if selected:
        render_detail(stones, selected)


if __name__ == "__main__":
    main()
