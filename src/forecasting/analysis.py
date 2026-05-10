"""결정론 분석 함수.

다차원 분석을 위한 패턴별 함수:
- get_event_window_sales / format_event_windows: 이벤트 ±N일 윈도우.
- get_weekday_pattern_md: 요일별 평균·분산.
- get_snap_effect_md: SNAP일 vs 비-SNAP일.

각 함수는 raw 데이터 + 자주 쓰이는 reference 값만 제공. 분류·결론은 LLM의 일.
"""
from __future__ import annotations

import duckdb
import pandas as pd

from src.config import DUCKDB_PATH, TEST_START_DATE

_con: duckdb.DuckDBPyConnection | None = None


def _get_con() -> duckdb.DuckDBPyConnection:
    global _con
    if _con is None:
        _con = duckdb.connect(str(DUCKDB_PATH), read_only=True)
    return _con


def _weekday_baseline_mean(
    sku_id: str,
    cutoff_date: str = TEST_START_DATE,
    last_n_days: int = 365,
) -> dict[str, float]:
    """이 SKU의 요일별 평균 sales (이벤트일 포함된 raw 평균).

    cutoff_date 이전 last_n_days 의 데이터로 계산. event lift 비교 baseline.
    """
    end_dt = pd.Timestamp(cutoff_date) - pd.Timedelta(days=1)
    start_dt = end_dt - pd.Timedelta(days=last_n_days - 1)
    sql = """
    SELECT c.weekday, AVG(s.sales) AS mean_sales
    FROM sales_train AS s
    JOIN calendar AS c ON s.date = c.date
    WHERE s.sku_id = ? AND s.date BETWEEN ? AND ?
    GROUP BY c.weekday
    """
    df = _get_con().execute(sql, [sku_id, str(start_dt.date()), str(end_dt.date())]).df()
    if df.empty:
        return {}
    return {row["weekday"]: float(row["mean_sales"]) for _, row in df.iterrows()}


def _seasonal_weekday_baseline(
    sku_id: str,
    ev_date,
    baseline_days: int = 30,
    exclude_days: int = 5,
) -> dict[str, float]:
    """이벤트 일자 기준 그 해 같은 시기(±baseline_days) 의 요일별 평균 sales.

    baseline_days = ±30 → 60일 윈도우 (계절성 + 그 해 전반적 수준 반영)
    exclude_days = ±5 → 이벤트 직접 영향 구간 제외 (이벤트 효과 contamination 방지)

    이게 "그 해 시계열 모델이 학습한 weekly seasonality + level" 의 proxy.
    이벤트 일자 sales / 이 baseline = *baseline 위 추가* 효과 = 진짜 이벤트 lift.
    """
    ev_pd = pd.Timestamp(ev_date)
    win_start = (ev_pd - pd.Timedelta(days=baseline_days)).date()
    win_end = (ev_pd + pd.Timedelta(days=baseline_days)).date()
    excl_start = (ev_pd - pd.Timedelta(days=exclude_days)).date()
    excl_end = (ev_pd + pd.Timedelta(days=exclude_days)).date()
    sql = """
    SELECT c.weekday, AVG(s.sales) AS mean_sales
    FROM sales_train AS s
    JOIN calendar AS c ON s.date = c.date
    WHERE s.sku_id = ?
      AND s.date BETWEEN ? AND ?
      AND s.date NOT BETWEEN ? AND ?
    GROUP BY c.weekday
    """
    df = _get_con().execute(
        sql, [sku_id, str(win_start), str(win_end), str(excl_start), str(excl_end)]
    ).df()
    if df.empty:
        return {}
    return {
        row["weekday"]: float(row["mean_sales"])
        for _, row in df.iterrows()
        if pd.notna(row["mean_sales"]) and row["mean_sales"] > 0
    }


def get_event_window_sales(
    sku_id: str,
    event_name: str,
    days: int = 7,
    cutoff_date: str = TEST_START_DATE,
) -> pd.DataFrame:
    """과거 동일 이벤트 일자 ± `days`일 윈도우의 sales 조회.

    Args:
        sku_id: 분석 대상 SKU.
        event_name: calendar.event_name_1 또는 event_name_2의 정확 표기 (예: 'SuperBowl').
        days: 이벤트일 기준 전후 일수 (기본 7).
        cutoff_date: 이 날짜 이전(< cutoff_date)의 이벤트만 조회 (기본 test 시작일).

    Returns:
        DataFrame columns:
          - ev_date: 과거 이벤트 일자
          - ev_year: 그 해
          - date: 윈도우 안의 일자
          - sales: 그날 sales
          - weekday: 요일
          - is_event_day: True면 이벤트 당일, False면 주변일
    """
    sql = f"""
    WITH past_events AS (
        SELECT date AS ev_date,
               EXTRACT('year' FROM date)::INT AS ev_year
        FROM calendar
        WHERE (event_name_1 = ? OR event_name_2 = ?)
          AND date < ?
    )
    SELECT pe.ev_date,
           pe.ev_year,
           s.date,
           s.sales,
           c.weekday,
           (s.date = pe.ev_date) AS is_event_day
    FROM past_events pe
    JOIN sales_train s
      ON s.date BETWEEN pe.ev_date - INTERVAL '{days}' DAY
                    AND pe.ev_date + INTERVAL '{days}' DAY
    JOIN calendar c ON s.date = c.date
    WHERE s.sku_id = ?
    ORDER BY pe.ev_date, s.date
    """
    df = _get_con().execute(sql, [event_name, event_name, cutoff_date, sku_id]).df()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["ev_date"] = pd.to_datetime(df["ev_date"]).dt.date
    return df


def format_event_windows(
    sku_id: str,
    event_name: str,
    days: int = 5,
    cutoff_date: str = TEST_START_DATE,
) -> str:
    """이벤트의 모든 과거 occurrences ±days일 윈도우 sales 를 *baseline 분리* 형태로
    markdown 반환.

    핵심 분리:
    - 각 occurrence 의 "그 해 같은 시기 (±30일) 같은 요일 평균" 을 *그 해 baseline*
      으로 사용 (시계열 모델이 학습한 weekly seasonality + 그 해 level 의 proxy).
    - lift = sales / 그 해 같은 요일 평균 = **baseline 위 추가 효과** (= 진짜 이벤트
      효과, baseline 이 못 본 부분).
    - 매년 ratio 를 *연도별로 비교* — 일관 / 추이 / outlier year 식별.

    이걸로 LLM 이 "토요일 +50% lift" 를 *주말 효과 + 이벤트 효과* 로 합성된 신호로
    오해하지 않게.
    """
    df = get_event_window_sales(sku_id, event_name, days=days, cutoff_date=cutoff_date)
    if df.empty:
        return f"## {event_name}\n\n과거 occurrence 없음."

    df["offset"] = (
        pd.to_datetime(df["date"]) - pd.to_datetime(df["ev_date"])
    ).dt.days

    # occurrence (year) 별 seasonal weekday baseline 계산
    occurrences = sorted(df["ev_date"].unique())
    seasonal_baselines: dict = {}
    for ev_date in occurrences:
        seasonal_baselines[ev_date] = _seasonal_weekday_baseline(
            sku_id, ev_date, baseline_days=30, exclude_days=days
        )

    # 각 row 에 그 해 weekday baseline 과 ratio 부여
    def _baseline_for(row):
        bl = seasonal_baselines.get(row["ev_date"], {})
        return bl.get(row["weekday"], None)

    df["seasonal_baseline"] = df.apply(_baseline_for, axis=1)
    df["ratio"] = df.apply(
        lambda r: (r["sales"] / r["seasonal_baseline"])
        if r["seasonal_baseline"] and r["seasonal_baseline"] > 0
        else None,
        axis=1,
    )

    parts = [f"## {event_name} — {df['ev_date'].nunique()} occurrences\n"]
    parts.append(
        "**ratio = 그 해 같은 시기(±30일) 같은 요일 평균 대비 비율**. "
        "= baseline 위 *추가* 효과 (시계열 모델이 못 본 진짜 이벤트 효과). "
        "1.0× = 평소 수준, 1.5× = +50% 더, 0.7× = -30%.\n"
    )

    # ============================================================
    # 1. 연도별 ratio 매트릭스 (event day + 주변 offset)
    # ============================================================
    parts.append("### 연도별 ratio 매트릭스 (offset × year)\n")
    parts.append(
        "각 셀 = sales / 그 해 같은 요일 평균. 연도별로 비교해 *매년 일관 인지, "
        "추이 인지, outlier year 인지* 판단.\n"
    )

    # offset → year → ratio matrix
    years = sorted({pd.Timestamp(d).year for d in occurrences})
    NEAR = min(days, 3)
    near_offsets = list(range(-NEAR, NEAR + 1))

    # 연도별 baseline-quality 평가: baseline 이 거의 0 (<0.5) 이면 ratio 의미 없음.
    # 낮은 볼륨 SKU 도 정상 처리되도록 threshold 는 절대값 0.5 만.
    sparse_years: set = set()
    for ev_date, bl in seasonal_baselines.items():
        if not bl:
            sparse_years.add(pd.Timestamp(ev_date).year)
            continue
        bl_avg = sum(bl.values()) / max(1, len(bl))
        if bl_avg < 0.5:
            sparse_years.add(pd.Timestamp(ev_date).year)

    # 표 header
    header = "| offset | 요일      |"
    sep = "|-------:|-----------|"
    for y in years:
        flag = " (sparse)" if y in sparse_years else ""
        header += f" {y}{flag}    |"
        sep += "-" * (10 + len(flag)) + ":|"
    header += " median (sparse 제외) |"
    sep += "------------:|"
    parts.append(header)
    parts.append(sep)

    # offset 별 median ratio (sparse 연도 제외) 저장
    offset_median_ratios: list[tuple[int, str, float, int]] = []  # (offset, wd, median, n_used)

    for off in near_offsets:
        off_rows = df[df["offset"] == off]
        if off_rows.empty:
            continue
        wd = str(off_rows["weekday"].iloc[0])
        mark = " ★" if off == 0 else "  "
        line = f"| {off:>+3}{mark} | {wd:<9} |"
        ratios_for_median = []
        for y in years:
            row = off_rows[pd.to_datetime(off_rows["ev_date"]).dt.year == y]
            if row.empty or pd.isna(row["ratio"].iloc[0]):
                line += "      —    |"
            else:
                r = float(row["ratio"].iloc[0])
                line += f" {r:>5.2f}× |"
                if y not in sparse_years:
                    ratios_for_median.append(r)
        if ratios_for_median:
            ratios_for_median.sort()
            n = len(ratios_for_median)
            median_r = (
                ratios_for_median[n // 2]
                if n % 2
                else (ratios_for_median[n // 2 - 1] + ratios_for_median[n // 2]) / 2
            )
            line += f" {median_r:>10.2f}× (n={n}) |"
            offset_median_ratios.append((off, wd, median_r, n))
        else:
            line += "          — |"
        parts.append(line)
    parts.append("")
    if sparse_years:
        parts.append(
            f"※ sparse 연도 ({sorted(sparse_years)}): 그 해 baseline 매출 자체가 매우 낮아 "
            f"(평균 < 5) ratio 가 부풀어짐. median 계산에서 제외함.\n"
        )

    # ============================================================
    # 2. 자동 신호 요약 (median 기준 — sparse year 제외)
    # ============================================================
    UPPER_THRESHOLD = 1.20  # +20% 이상 = 신호
    LOWER_THRESHOLD = 0.80  # -20% 이하 = 신호
    signals = []
    for off, wd, mr, n in offset_median_ratios:
        if mr >= UPPER_THRESHOLD or mr <= LOWER_THRESHOLD:
            if off < 0:
                kind = "build-up" if mr > 1.0 else "직전 dip"
            elif off == 0:
                kind = "이벤트 당일 spike" if mr > 1.0 else "이벤트 당일 dip"
            else:
                kind = "post-event lift" if mr > 1.0 else "anti-spike"
            signals.append((off, wd, mr, kind, n))

    parts.append("### 자동 신호 요약 (median ratio 기준, sparse year 제외)\n")
    if not signals:
        parts.append(
            f"- 모든 offset 의 median ratio 가 {LOWER_THRESHOLD:.2f}× ~ "
            f"{UPPER_THRESHOLD:.2f}× 범위 — 다일 보정 신호 없음. 이벤트 효과 미미.\n"
        )
    else:
        parts.append(
            f"median ratio (sparse 연도 제외) 가 baseline (1.0×) 에서 ±20% 이상 벗어난 offset. "
            "**이게 진짜 이벤트 효과 — Agent 는 forecast 의 그 일자 baseline × ratio 로 적용**.\n"
        )
        for off, wd, mr, kind, n in signals:
            sign_arrow = "직전" if off < 0 else ("당일" if off == 0 else "직후")
            lift_pct = (mr - 1) * 100
            parts.append(
                f"- offset {off:+d} ({wd}, {sign_arrow}): "
                f"median ratio **{mr:.2f}×** ({lift_pct:+.0f}%, n={n}) — {kind}"
            )
        parts.append("")

    # ============================================================
    # 3. 각 occurrence 상세 (이벤트 일자 sales + ratio + 그 해 baseline)
    # ============================================================
    for ev_date, group in df.groupby("ev_date", sort=True):
        group = group.sort_values("date").reset_index(drop=True)
        ev_row = group[group["is_event_day"]]
        if ev_row.empty:
            continue
        ev_weekday = str(ev_row["weekday"].iloc[0])
        ev_sales = int(ev_row["sales"].iloc[0])
        ev_ratio = ev_row["ratio"].iloc[0]
        ev_baseline = ev_row["seasonal_baseline"].iloc[0]

        parts.append(f"### {ev_date} ({ev_weekday}) ← event day")
        if ev_baseline:
            parts.append(
                f"- 이벤트 sales: **{ev_sales}**. 그 해 같은 시기 {ev_weekday} 평균: "
                f"**{ev_baseline:.1f}**. → ratio: **{ev_ratio:.2f}×** "
                f"({(ev_ratio-1)*100:+.0f}%)"
            )
        else:
            parts.append(
                f"- 이벤트 sales: **{ev_sales}**. (seasonal baseline 데이터 부족)"
            )

        # Daily 표 — 각 일자 sales + 그 해 같은 요일 평균 + ratio
        rows = ["| date       | weekday   | sales | seasonal Sat/Sun등 평균 | ratio |",
                "|------------|-----------|------:|------------------------:|------:|"]
        for _, r in group.iterrows():
            mark = "★" if r["is_event_day"] else " "
            bl = r["seasonal_baseline"]
            ratio_str = (
                f"{r['ratio']:>5.2f}×" if bl and pd.notna(r["ratio"]) else "    —"
            )
            bl_str = f"{bl:>22.1f}" if bl else "                     —"
            rows.append(
                f"| {r['date']} | {str(r['weekday']):<9} | {int(r['sales']):>5} | "
                f"{bl_str} | {ratio_str} |  {mark}"
            )
        parts.append("\n".join(rows))
        parts.append("")

    return "\n".join(parts)


WEEKDAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def get_weekday_pattern_md(
    sku_id: str,
    last_n_days: int = 365,
    cutoff_date: str = TEST_START_DATE,
) -> str:
    """이 SKU의 요일별 sales 패턴을 markdown으로 반환.

    최근 last_n_days 기간의 요일별 평균·중앙·표준편차·관측 수.
    이벤트일도 포함됨 (이벤트는 별도 분석 도구가 따로 처리).

    cutoff_date 이전 데이터만 사용 (test 누수 방지).
    """
    end_dt = pd.Timestamp(cutoff_date) - pd.Timedelta(days=1)
    start_dt = end_dt - pd.Timedelta(days=last_n_days - 1)
    sql = """
    SELECT s.date, s.sales, c.weekday
    FROM sales_train AS s
    JOIN calendar AS c ON s.date = c.date
    WHERE s.sku_id = ? AND s.date BETWEEN ? AND ?
    """
    df = _get_con().execute(sql, [sku_id, str(start_dt.date()), str(end_dt.date())]).df()
    if df.empty:
        return f"## 요일 패턴 (최근 {last_n_days}일)\n\n데이터 없음."

    parts = [
        f"## 요일 패턴 (최근 {last_n_days}일: {start_dt.date()} ~ {end_dt.date()})",
        "",
        "| weekday   |   n |   mean | median |    std |   min |   max | zero_count |",
        "|-----------|----:|-------:|-------:|-------:|------:|------:|-----------:|",
    ]
    for wd in WEEKDAY_ORDER:
        sub = df[df["weekday"] == wd]["sales"]
        if sub.empty:
            continue
        zeros = int((sub == 0).sum())
        parts.append(
            f"| {wd:<9} | {len(sub):>3} | {sub.mean():>6.1f} | {sub.median():>6.1f} | "
            f"{sub.std():>6.1f} | {int(sub.min()):>5} | {int(sub.max()):>5} | {zeros:>10} |"
        )

    weekend = df[df["weekday"].isin(["Saturday", "Sunday"])]["sales"]
    weekday = df[~df["weekday"].isin(["Saturday", "Sunday"])]["sales"]
    parts.append("")
    parts.append(
        f"- weekday 평균(Mon-Fri): {weekday.mean():.1f} (n={len(weekday)})"
    )
    parts.append(
        f"- weekend 평균(Sat-Sun): {weekend.mean():.1f} (n={len(weekend)})"
    )
    if weekday.mean() > 0:
        wkdratio = weekend.mean() / weekday.mean() - 1
        parts.append(f"- weekend 대비 weekday: {wkdratio:+.1%}")
    return "\n".join(parts)


def get_same_period_history_md(
    sku_id: str,
    forecast_start: str,
    forecast_end: str,
    n_years: int = 3,
) -> str:
    """forecast 기간의 과거 같은 시기 sales — anomaly 식별 + robust median.

    각 과거 연도별 detail + flag (anomaly·sparse) + 전체 robust median 제공.
    LLM 이 *아무 평균이나* 쓰지 않고 *anomaly 분리한 합리적 reference* 사용하게.
    """
    fs = pd.Timestamp(forecast_start)
    fe = pd.Timestamp(forecast_end)
    parts = [
        f"## 과거 같은 시기 ({fs.month}/{fs.day} ~ {fe.month}/{fe.day}) 매출",
        "",
        "*anomaly* = 다른 연도와 매우 다른 수준 (평균이 다른 해의 2배 이상 등). "
        "*sparse* = 0 일자 다수 (공급 단절·신제품 가능). robust median 은 anomaly·sparse 제외.\n",
    ]

    rows: list[dict] = []
    for year_offset in range(1, n_years + 1):
        y_start = fs - pd.DateOffset(years=year_offset)
        y_end = fe - pd.DateOffset(years=year_offset)
        if y_end >= pd.Timestamp(TEST_START_DATE):
            continue
        sql = """
        SELECT s.date, s.sales, c.weekday
        FROM sales_train AS s
        JOIN calendar AS c ON s.date = c.date
        WHERE s.sku_id = ? AND s.date BETWEEN ? AND ?
        ORDER BY s.date
        """
        df = _get_con().execute(
            sql, [sku_id, str(y_start.date()), str(y_end.date())]
        ).df()
        if df.empty:
            parts.append(f"- **{y_start.year}**: 데이터 없음")
            continue
        total = int(df["sales"].sum())
        avg = float(df["sales"].mean())
        n = len(df)
        weekend_part = df[df["weekday"].isin(["Saturday", "Sunday"])]["sales"]
        weekday_part = df[~df["weekday"].isin(["Saturday", "Sunday"])]["sales"]
        weekend_avg = float(weekend_part.mean()) if len(weekend_part) else 0.0
        weekday_avg = float(weekday_part.mean()) if len(weekday_part) else 0.0
        zero_count = int((df["sales"] == 0).sum())
        sparse_pct = zero_count / n
        rows.append(
            {
                "year": y_start.year,
                "start": y_start.date(),
                "end": y_end.date(),
                "n": n,
                "total": total,
                "avg": avg,
                "weekday_avg": weekday_avg,
                "weekend_avg": weekend_avg,
                "zero_count": zero_count,
                "sparse_pct": sparse_pct,
            }
        )

    if not rows:
        parts.append("- (과거 데이터 없음)")
        return "\n".join(parts)

    # anomaly·sparse 자동 식별
    avgs = [r["avg"] for r in rows]
    avgs_sorted = sorted(avgs)
    median_of_avgs = (
        avgs_sorted[len(avgs_sorted) // 2]
        if len(avgs_sorted) % 2
        else (avgs_sorted[len(avgs_sorted) // 2 - 1] + avgs_sorted[len(avgs_sorted) // 2]) / 2
    )
    for r in rows:
        flags = []
        if r["sparse_pct"] >= 0.20:
            flags.append("sparse")
        # anomaly: 다른 해 median 의 2배 이상 또는 0.5배 이하
        if median_of_avgs > 0:
            if r["avg"] >= 2.0 * median_of_avgs:
                flags.append("anomaly-high")
            elif r["avg"] <= 0.5 * median_of_avgs:
                flags.append("anomaly-low")
        r["flags"] = flags

    # 표 출력
    parts.append("| year | n | 평균 | 평일 평균 | 주말 평균 | 0 일자 | flag |")
    parts.append("|-----:|--:|-----:|---------:|---------:|-------:|------|")
    for r in rows:
        flag_str = ", ".join(r["flags"]) if r["flags"] else "-"
        parts.append(
            f"| {r['year']} | {r['n']} | {r['avg']:.1f} | {r['weekday_avg']:.1f} | "
            f"{r['weekend_avg']:.1f} | {r['zero_count']} | {flag_str} |"
        )
    parts.append("")

    # robust reference (anomaly·sparse 제외)
    clean_rows = [r for r in rows if not r["flags"]]
    if clean_rows:
        clean_avgs = [r["avg"] for r in clean_rows]
        clean_wd = [r["weekday_avg"] for r in clean_rows]
        clean_we = [r["weekend_avg"] for r in clean_rows]
        parts.append(
            f"**robust reference** (flag 없는 {len(clean_rows)}개 연도 median):"
        )
        parts.append(
            f"- 평균: {sorted(clean_avgs)[len(clean_avgs)//2]:.1f}, "
            f"평일: {sorted(clean_wd)[len(clean_wd)//2]:.1f}, "
            f"주말: {sorted(clean_we)[len(clean_we)//2]:.1f}"
        )
    else:
        parts.append(
            "**robust reference**: 모든 연도가 anomaly/sparse 로 분류됨 — "
            "과거 같은 시기 reference 신뢰 낮음. *최근 input 또는 365일 요일 패턴* 활용 권장."
        )
    return "\n".join(parts)


def get_snap_effect_md(
    sku_id: str,
    state_id: str,
    last_n_days: int = 365,
    cutoff_date: str = TEST_START_DATE,
) -> str:
    """이 SKU·이 state의 SNAP일 vs 비-SNAP일 sales 평균을 markdown으로 반환.

    최근 last_n_days 기간 기준. 이벤트일도 포함.
    """
    snap_col = f"snap_{state_id}"
    if state_id not in ("CA", "TX", "WI"):
        return f"## SNAP 효과\n\n알 수 없는 state_id: {state_id}"

    end_dt = pd.Timestamp(cutoff_date) - pd.Timedelta(days=1)
    start_dt = end_dt - pd.Timedelta(days=last_n_days - 1)
    sql = f"""
    SELECT s.sales, c.{snap_col} AS is_snap
    FROM sales_train AS s
    JOIN calendar AS c ON s.date = c.date
    WHERE s.sku_id = ? AND s.date BETWEEN ? AND ?
    """
    df = _get_con().execute(sql, [sku_id, str(start_dt.date()), str(end_dt.date())]).df()
    if df.empty:
        return f"## SNAP 효과 ({state_id}, 최근 {last_n_days}일)\n\n데이터 없음."

    snap = df[df["is_snap"] == 1]["sales"]
    nosnap = df[df["is_snap"] == 0]["sales"]
    parts = [
        f"## SNAP 효과 ({state_id}, 최근 {last_n_days}일: {start_dt.date()} ~ {end_dt.date()})",
        "",
        f"- SNAP 일자 평균: {snap.mean():.2f} (n={len(snap)}, median {snap.median():.1f})",
        f"- 비-SNAP 일자 평균: {nosnap.mean():.2f} (n={len(nosnap)}, median {nosnap.median():.1f})",
    ]
    if nosnap.mean() > 0:
        diff = snap.mean() / nosnap.mean() - 1
        parts.append(f"- SNAP 효과: {diff:+.1%} (mean SNAP / mean non-SNAP)")
    return "\n".join(parts)
