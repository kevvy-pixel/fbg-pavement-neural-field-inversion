from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
DATA = ROOT / "data"
FIG.mkdir(exist_ok=True)

FONT_REG = ROOT / "fonts" / "NotoSansCJKsc-Regular.otf"
FONT_BOLD = ROOT / "fonts" / "NotoSansCJKsc-Bold.otf"
CN = FontProperties(fname=str(FONT_REG))
CN_B = FontProperties(fname=str(FONT_BOLD))

mpl.rcParams.update(
    {
        "axes.unicode_minus": False,
        "figure.dpi": 160,
        "savefig.dpi": 220,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.22,
        "grid.linewidth": 0.7,
    }
)

BLUE = "#2563EB"
TEAL = "#0F766E"
ORANGE = "#EA580C"
PURPLE = "#7C3AED"
RED = "#DC2626"
GRAY = "#64748B"
LIGHT = "#EFF6FF"


def set_cn(ax, title=None, xlabel=None, ylabel=None):
    if title:
        ax.set_title(title, fontproperties=CN_B, fontsize=12, pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, fontproperties=CN, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, fontproperties=CN, fontsize=10)
    for lab in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        lab.set_fontproperties(CN)
        lab.set_fontsize(9)


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def evolution_flow():
    fig, ax = plt.subplots(figsize=(11.3, 4.1))
    ax.axis("off")
    boxes = [
        (0.02, 0.54, 0.18, 0.32, "Step19 基线\nFiLM + 自动微分", "完整共享神经场"),
        (0.22, 0.54, 0.18, 0.32, "结构探索\nResidual / MoE", "快速筛选后严格复核"),
        (0.42, 0.54, 0.18, 0.32, "公平全训练\nSOTA vs Residual", "恢复同规格与同预算"),
        (0.62, 0.54, 0.16, 0.32, "稀疏校准\nRidge-4", "低成本工况适配"),
        (0.80, 0.54, 0.18, 0.32, "稳健性验证\nRidge-6 + Gate", "20 次嵌套空间划分"),
    ]
    colors = [BLUE, ORANGE, PURPLE, TEAL, "#0369A1"]
    for i, (x, y, w, h, title, sub) in enumerate(boxes):
        box = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            linewidth=1.5,
            edgecolor=colors[i],
            facecolor=LIGHT if i != 4 else "#ECFEFF",
        )
        ax.add_patch(box)
        ax.text(
            x + w / 2,
            y + h * 0.64,
            title,
            ha="center",
            va="center",
            fontproperties=CN_B,
            fontsize=10.5,
            color=colors[i],
        )
        ax.text(
            x + w / 2,
            y + h * 0.22,
            sub,
            ha="center",
            va="center",
            fontproperties=CN,
            fontsize=8.5,
            color="#334155",
        )
        if i < len(boxes) - 1:
            ax.annotate(
                "",
                xy=(boxes[i + 1][0] - 0.006, y + h / 2),
                xytext=(x + w + 0.006, y + h / 2),
                arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=1.5),
            )
    decisions = [
        (0.11, "发现 95 cm 极边界短板"),
        (0.31, "快速实验提示轻微增益"),
        (0.51, "严格对照否定残差主干"),
        (0.70, "4 点有均值收益但波动大"),
        (0.89, "敲定 SOTA + Ridge-6 / 门控"),
    ]
    for x, txt in decisions:
        ax.text(
            x,
            0.32,
            txt,
            ha="center",
            va="center",
            fontproperties=CN,
            fontsize=8.5,
            color="#0F172A",
        )
        ax.plot([x, x], [0.47, 0.39], color="#CBD5E1", lw=1.2)
    ax.text(
        0.5,
        0.08,
        "原则：先检验网络结构，再把工况特异误差交给可解释、闭式求解的稀疏校准层",
        ha="center",
        fontproperties=CN_B,
        fontsize=10.5,
        color="#0F172A",
    )
    save(fig, "experiment_evolution")


def fair_comparison():
    zero = pd.read_csv(DATA / "full_zero_shot_summary_v7.csv")
    fixed = pd.read_csv(DATA / "common_fixed_test_summary_v7.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.1))
    labels = ["Step19-SOTA", "Residual-FiLM"]
    colors = [BLUE, ORANGE]
    z = zero.set_index("method").loc[
        ["Step19-SOTA-FiLM", "Step19-Residual-FiLM"]
    ]
    f = fixed.set_index("method").loc[
        ["Step19-SOTA-FiLM", "Step19-Residual-FiLM"]
    ]
    for ax, df, title in zip(
        axes,
        [z, f],
        ["完整零样本测试（616 行）", "共同固定测试集（528 行）"],
    ):
        vals = df["micro_RMSE_microstrain"].to_numpy()
        bars = ax.bar(labels, vals, color=colors, width=0.58)
        ax.set_ylim(0, max(vals) * 1.26)
        set_cn(ax, title, ylabel="RMSE（με）")
        ax.set_xticklabels(labels, fontproperties=CN, rotation=0)
        for b, v in zip(bars, vals):
            ax.text(
                b.get_x() + b.get_width() / 2,
                v + max(vals) * 0.035,
                f"{v:.1f}",
                ha="center",
                fontproperties=CN_B,
                fontsize=10,
            )
    fig.suptitle(
        "Step19 同规格、同训练预算的公平对照",
        fontproperties=CN_B,
        fontsize=14,
        y=1.02,
    )
    fig.tight_layout()
    save(fig, "fair_sota_residual")


def repeated_global():
    df = pd.read_csv(DATA / "method_summary_v9.csv")
    order = [
        "Step19-SOTA-FiLM",
        "Step19-SOTA+Ridge-4",
        "Step19-SOTA+Ridge-6",
        "Step19-SOTA+Gated-Ridge-6",
    ]
    labels = ["SOTA", "Ridge-4", "Ridge-6", "Gated Ridge-6"]
    colors = [BLUE, GRAY, TEAL, PURPLE]
    df = df.set_index("method").loc[order]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    x = np.arange(4)
    axes[0].bar(
        x,
        df["micro_RMSE_mean_microstrain"],
        yerr=df["micro_RMSE_std_microstrain"],
        capsize=4,
        color=colors,
    )
    axes[1].bar(
        x,
        df["micro_R2_mean"],
        yerr=df["micro_R2_std"],
        capsize=4,
        color=colors,
    )
    for ax in axes:
        ax.set_xticks(x, labels, rotation=12, ha="right")
    set_cn(axes[0], "20 次重复空间划分：总体 RMSE", ylabel="RMSE（με，均值±标准差）")
    set_cn(axes[1], "20 次重复空间划分：总体 $R^2$", ylabel="$R^2$（均值±标准差）")
    axes[0].set_ylim(65, 94)
    axes[1].set_ylim(0.80, 0.90)
    for ax, col in [
        (axes[0], "micro_RMSE_mean_microstrain"),
        (axes[1], "micro_R2_mean"),
    ]:
        for i, v in enumerate(df[col]):
            ax.text(
                i,
                v + (1.2 if "RMSE" in col else 0.004),
                f"{v:.2f}" if "RMSE" in col else f"{v:.3f}",
                ha="center",
                fontproperties=CN_B,
                fontsize=8.5,
            )
    fig.tight_layout()
    save(fig, "repeated_global_performance")


def target_95cm():
    df = pd.read_csv(DATA / "target_95cm_summary_v9.csv")
    baseline = float(
        (df["RMSE_mean_microstrain"] + df["RMSE_gain_mean_microstrain"]).mean()
    )
    labels = ["SOTA\n基线", "Ridge-4", "Ridge-6", "Gated\nRidge-6"]
    means = [baseline] + df["RMSE_mean_microstrain"].tolist()
    errs = [0.0] + df["RMSE_std_microstrain"].tolist()
    colors = [BLUE, GRAY, TEAL, PURPLE]
    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    x = np.arange(4)
    bars = ax.bar(x, means, yerr=errs, capsize=4, color=colors, width=0.62)
    ax.set_xticks(x, labels)
    set_cn(
        ax,
        "40 kg 纵中心线 95 cm：20 次重复划分",
        ylabel="固定测试集 RMSE（με）",
    )
    ax.set_ylim(80, 175)
    for i, (b, v) in enumerate(zip(bars, means)):
        ax.text(
            b.get_x() + b.get_width() / 2,
            v + (3 if i == 0 else errs[i] + 3),
            f"{v:.1f}",
            ha="center",
            fontproperties=CN_B,
            fontsize=9.5,
        )
    ax.text(
        0.98,
        0.96,
        "Ridge-6：平均降低 15.8 με，胜率 85%",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontproperties=CN_B,
        fontsize=9.5,
        color=TEAL,
    )
    save(fig, "target_95cm")


def case_gain():
    df = pd.read_csv(DATA / "case_robustness_summary_v9.csv")
    df = df[df["method"] == "Step19-SOTA+Ridge-6"].copy()
    df = df.sort_values("RMSE_gain_mean_microstrain")
    labels = (
        df["case_id"]
        .str.replace("加载", "", regex=False)
        .str.replace("纵中心线", "纵线", regex=False)
        .tolist()
    )
    vals = df["RMSE_gain_mean_microstrain"].to_numpy()
    wins = df["win_rate_percent"].to_numpy()
    colors = [TEAL if v >= 0 else RED for v in vals]
    fig, ax = plt.subplots(figsize=(8.4, 8.1))
    y = np.arange(len(df))
    ax.barh(y, vals, color=colors, alpha=0.9)
    ax.axvline(0, color="#0F172A", lw=0.9)
    ax.set_yticks(y, labels)
    set_cn(
        ax,
        "Ridge-6 分工况 RMSE 改善（20 次重复均值）",
        xlabel="相对 SOTA 的 RMSE 降低量（με；正值为改善）",
    )
    for i, (v, win) in enumerate(zip(vals, wins)):
        ax.text(
            v + (0.7 if v >= 0 else -0.7),
            i,
            f"{v:.1f} / {win:.0f}%",
            va="center",
            ha="left" if v >= 0 else "right",
            fontproperties=CN,
            fontsize=7.5,
        )
    ax.text(
        0.99,
        0.015,
        "标注：平均 RMSE 改善 / 胜率",
        transform=ax.transAxes,
        ha="right",
        fontproperties=CN,
        fontsize=8,
        color=GRAY,
    )
    fig.tight_layout()
    save(fig, "casewise_ridge6_gain")


def main():
    evolution_flow()
    fair_comparison()
    repeated_global()
    target_95cm()
    case_gain()
    print(f"Figures written to {FIG}")


if __name__ == "__main__":
    main()
