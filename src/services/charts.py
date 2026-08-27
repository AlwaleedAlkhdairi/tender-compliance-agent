"""Altair chart builders for the Streamlit result views.

Palette: validated categorical slots (CVD-checked); identity keeps a fixed
hue per supplier, disqualification is shown by a neutral bar + an explicit
label (never color alone). Every value is also visible in the adjacent
tables, so charts never carry information alone.
"""

import altair as alt
import pandas as pd

# Fixed categorical order (validated light-mode slots).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
NEUTRAL = "#8a8a85"
GRID = "#e8e8e4"
INK = "#3d3d3a"


def supplier_colors(suppliers: list[str]) -> dict[str, str]:
    """Stable supplier -> hue assignment (alphabetical, fixed slot order)."""
    return {s: SERIES[i % len(SERIES)] for i, s in enumerate(sorted(suppliers))}


def _base_axis(**kwargs):
    return alt.Axis(gridColor=GRID, domainColor=GRID, tickColor=GRID,
                    labelColor=INK, titleColor=INK, **kwargs)


def weighted_total_chart(ranking: list[dict]) -> alt.Chart:
    """Horizontal magnitude bars of the weighted totals, ranked."""
    rows = [
        {
            "supplier": r["supplier"] + ("  (disqualified)" if r["disqualified"] else ""),
            "total": r["weighted_total"],
            "status": "Disqualified" if r["disqualified"] else "Qualified",
            "rank": r["rank"],
        }
        for r in ranking
    ]
    df = pd.DataFrame(rows)
    base = alt.Chart(df).encode(
        y=alt.Y("supplier:N", sort=alt.EncodingSortField("rank"),
                axis=_base_axis(title=None, labelLimit=260)),
        x=alt.X("total:Q", scale=alt.Scale(domain=[0, 100]),
                axis=_base_axis(title="Weighted total (0-100)")),
    )
    bars = base.mark_bar(cornerRadiusEnd=4, height=26).encode(
        color=alt.Color(
            "status:N",
            scale=alt.Scale(domain=["Qualified", "Disqualified"],
                            range=[SERIES[0], NEUTRAL]),
            legend=alt.Legend(title=None, orient="top", labelColor=INK),
        ),
        tooltip=["supplier", "total", "status", "rank"],
    )
    labels = base.mark_text(align="left", dx=6, color=INK).encode(
        text=alt.Text("total:Q", format=".1f")
    )
    return (bars + labels).properties(height=max(90, 44 * len(rows)))


def criterion_breakdown_chart(ranking: list[dict]) -> alt.Chart:
    """Grouped bars: per-criterion scores, one fixed hue per supplier."""
    rows = []
    for r in ranking:
        for criterion, cell in r["breakdown"].items():
            rows.append({
                "supplier": r["supplier"],
                "criterion": criterion,
                "score": cell["score"],
                "weight": f"{cell['weight']}%",
            })
    df = pd.DataFrame(rows)
    colors = supplier_colors(sorted({r["supplier"] for r in ranking}))
    return (
        alt.Chart(df)
        .mark_bar(cornerRadiusEnd=4)
        .encode(
            x=alt.X("criterion:N", axis=_base_axis(title=None, labelAngle=0, labelLimit=140)),
            xOffset=alt.XOffset("supplier:N", sort=sorted(colors)),
            y=alt.Y("score:Q", scale=alt.Scale(domain=[0, 100]),
                    axis=_base_axis(title="Score (0-100)")),
            color=alt.Color(
                "supplier:N",
                scale=alt.Scale(domain=sorted(colors), range=[colors[s] for s in sorted(colors)]),
                legend=alt.Legend(title=None, orient="top", labelColor=INK),
            ),
            tooltip=["supplier", "criterion", "score", "weight"],
        )
        .properties(height=280)
    )
