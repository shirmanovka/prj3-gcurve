import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
from datetime import date, timedelta
from decimal import Decimal, ROUND_HALF_UP
from scipy.interpolate import CubicSpline

# ─────────────────────────────────────────────
# КОНФИГУРАЦИЯ
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="G-curve | MOEX",
    page_icon="📈",
    layout="wide",
)

MOEX_ZCYC_URL = "https://iss.moex.com/iss/engines/stock/zcyc/securities.json"
PLOT_MATURITIES = np.arange(0.25, 20.25, 0.25)   # до 20 лет — крайняя точка MOEX


def math_round(value: float, decimals: int = 2) -> float:
    """Математическое округление (0.5 всегда вверх)."""
    d = Decimal(str(value))
    quant = Decimal('0.' + '0' * decimals) if decimals > 0 else Decimal('0')
    return float(d.quantize(quant, rounding=ROUND_HALF_UP))

# ─────────────────────────────────────────────
# ПОЛУЧЕНИЕ ДАННЫХ С MOEX
# ─────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_zcyc(query_date: str) -> dict | None:
    """
    Возвращает словарь с кубическим сплайном G-curve.
    MOEX отдаёт доходности в yearyields (11 точек: 0.25…20 лет).
    Мы строим CubicSpline для произвольного срока.
    """
    params = {"date": query_date} if query_date else {}
    try:
        resp = requests.get(MOEX_ZCYC_URL, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return {"error": str(e)}

    # --- yearyields: готовые доходности по стандартным срокам ---
    try:
        cols = data["yearyields"]["columns"]   # ['tradedate','tradetime','period','value']
        rows = data["yearyields"]["data"]
    except KeyError:
        return {"error": "Неожиданная структура ответа MOEX API"}

    if not rows:
        return {"error": f"Нет данных G-curve за дату {query_date}. Выберите торговый день."}

    periods = np.array([r[cols.index("period")] for r in rows], dtype=float)
    yields  = np.array([r[cols.index("value")]  for r in rows], dtype=float)

    # --- params: коэффициенты модели (для информации) ---
    pcols = data["params"]["columns"]
    prows = data["params"]["data"]
    model_params = {}
    if prows:
        row = dict(zip(pcols, prows[0]))
        model_params = {k: row[k] for k in ("B1", "B2", "B3", "T1", "G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G9") if k in row}

    trade_date = rows[0][cols.index("tradedate")]

    # Возвращаем только сериализуемые данные — сплайн строится отдельно
    return {
        "periods":      periods.tolist(),
        "yields":       yields.tolist(),
        "date":         trade_date,
        "model_params": model_params,
    }


def gcurve_value(result: dict, m: float) -> float:
    """
    Значение G-curve (%) для срока m лет.
    Сплайн строится из кешированных списков — операция быстрая (11 точек).
    """
    periods = np.array(result["periods"])
    yields  = np.array(result["yields"])

    if m <= periods[0]:
        # Линейная экстраполяция влево
        slope = (yields[1] - yields[0]) / (periods[1] - periods[0])
        return float(yields[0] + slope * (m - periods[0]))
    if m >= periods[-1]:
        # Линейная экстраполяция вправо
        slope = (yields[-1] - yields[-2]) / (periods[-1] - periods[-2])
        return float(yields[-1] + slope * (m - periods[-1]))

    cs = CubicSpline(periods, yields)
    return float(cs(m))


# ─────────────────────────────────────────────
# РАСЧЁТ СПРЕДА
# ─────────────────────────────────────────────
def effective_yield(coupon_pct: float, periods: int) -> float:
    """Эффективная доходность: (1 + c/n)^n - 1, результат в %."""
    return ((1 + coupon_pct / 100 / periods) ** periods - 1) * 100


def build_spread_table(base_coupon: float, maturity: float,
                       periods: int, g_val: float) -> pd.DataFrame:
    """
    Таблица 11 строк: базовый купон ± 5/10/15/20/25 б.п.
    Шаг 5 б.п. = 0.05 % изменение купона.
    """
    offsets_bp = [-25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25]
    rows = []
    for bp in offsets_bp:
        c = base_coupon + bp * 0.01     # 1 б.п. = 0.01%
        ey = effective_yield(c, periods)
        spread = (ey-g_val) * 100     # % → б.п.
        rows.append({
            "Купон, %":                    math_round(c, 2),
            "Периодичность":               periods,
            "Эффективная доходность, %":   math_round(ey, 2),
            "G-curve, %":                  math_round(g_val, 2),
            "Спред, б.п.":                 math_round(spread, 2),
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
st.sidebar.title("Параметры")

if st.sidebar.button("🔄 Обновить данные"):
    st.cache_data.clear()
    st.rerun()

# Основная дата
main_date: date = st.sidebar.date_input(
    "Основная дата",
    value=date.today(),
    max_value=date.today(),
)

# Дополнительные даты для сравнения
st.sidebar.markdown("---")
st.sidebar.markdown("**Дополнительные даты для сравнения**")

if "extra_dates" not in st.session_state:
    st.session_state.extra_dates: list[date] = []

col_inp, col_btn = st.sidebar.columns([3, 1])
with col_inp:
    extra_date_input: date = st.date_input(
        "Дата",
        value=date.today() - timedelta(days=7),
        max_value=date.today(),
        label_visibility="collapsed",
        key="extra_date_picker",
    )
with col_btn:
    if st.button("＋", help="Добавить дату на график"):
        if extra_date_input not in st.session_state.extra_dates and extra_date_input != main_date:
            st.session_state.extra_dates.append(extra_date_input)

for i, d in enumerate(list(st.session_state.extra_dates)):
    c1, c2 = st.sidebar.columns([4, 1])
    c1.write(str(d))
    if c2.button("✕", key=f"del_{i}"):
        st.session_state.extra_dates.pop(i)
        st.rerun()

# ─────────────────────────────────────────────
# ПАРАМЕТРЫ ОБЛИГАЦИИ (sidebar)
# ─────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown("**Параметры облигации**")

coupon_input = st.sidebar.number_input(
    "Купон (%)", min_value=0.0, max_value=100.0,
    value=13.0, step=0.05, format="%.2f",
)
maturity_input = st.sidebar.number_input(
    "Срок до погашения (лет)", min_value=0.25, max_value=20.0,
    value=3.0, step=0.25, format="%.2f",
)
periods_input = st.sidebar.selectbox(
    "Периодичность выплат", options=[1, 2, 4, 12], index=3,
)

# ─────────────────────────────────────────────
# ЗАГРУЗКА ОСНОВНЫХ ДАННЫХ
# ─────────────────────────────────────────────
st.title("Кривая бескупонной доходности (G-curve)")
st.caption("Данные MOEX ISS | Кубическая интерполяция по 11 точкам ОФЗ")

with st.spinner(f"Загрузка G-curve за {main_date}…"):
    main_res = fetch_zcyc(str(main_date))

if "error" in main_res:
    st.error(f"**Ошибка:** {main_res['error']}")
    st.stop()

# Значение G-curve для выбранного срока — показываем в sidebar сразу
g_main = gcurve_value(main_res, maturity_input)

st.sidebar.markdown("---")
st.sidebar.markdown("**Результат расчёта**")
st.sidebar.metric(
    label=f"G-curve на {maturity_input} лет (дата {main_res['date']})",
    value=f"{math_round(g_main):.2f} %",
)
eff_base = effective_yield(coupon_input, periods_input)
spread_base = (eff_base - g_main) * 100
st.sidebar.metric(
    label=f"Эфф. доходность купона {coupon_input}%",
    value=f"{math_round(eff_base):.2f} %",
)
st.sidebar.metric(
    label="Спред к G-curve",
    value=f"{math_round(spread_base):.2f} б.п.",
)

# ─────────────────────────────────────────────
# ГРАФИК G-CURVE
# ─────────────────────────────────────────────
fig = go.Figure()

# Основная кривая (жирная)
y_main = [gcurve_value(main_res, m) for m in PLOT_MATURITIES]
fig.add_trace(go.Scatter(
    x=PLOT_MATURITIES, y=y_main,
    mode="lines",
    name=f"{main_res['date']} (основная)",
    line=dict(width=3, color="#1f77b4"),
))
# Реальные точки MOEX поверх кривой
fig.add_trace(go.Scatter(
    x=main_res["periods"], y=main_res["yields"],
    mode="markers",
    name="Точки MOEX",
    marker=dict(size=7, color="#1f77b4", symbol="circle"),
    showlegend=False,
))

# Маркер выбранного срока
fig.add_trace(go.Scatter(
    x=[maturity_input], y=[g_main],
    mode="markers+text",
    name=f"Срок {maturity_input} лет",
    marker=dict(size=12, color="red", symbol="x"),
    text=[f"{math_round(g_main):.2f}%"],
    textposition="top center",
    showlegend=False,
))

# Дополнительные кривые
palette = ["#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]
for idx, extra_d in enumerate(st.session_state.extra_dates):
    with st.spinner(f"Загрузка G-curve за {extra_d}…"):
        ep = fetch_zcyc(str(extra_d))
    if "error" in ep:
        st.warning(f"Дата {extra_d}: {ep['error']}")
        continue
    y_extra = [gcurve_value(ep, m) for m in PLOT_MATURITIES]
    fig.add_trace(go.Scatter(
        x=PLOT_MATURITIES, y=y_extra,
        mode="lines",
        name=str(extra_d),
        line=dict(width=1.5, dash="dash", color=palette[idx % len(palette)]),
    ))

fig.update_layout(
    title=f"G-curve MOEX ОФЗ | {main_res['date']}" +
          (f" + {len(st.session_state.extra_dates)} доп." if st.session_state.extra_dates else ""),
    xaxis_title="Срок (годы)",
    yaxis_title="Доходность (%)",
    legend=dict(orientation="h", y=-0.2),
    hovermode="x unified",
    height=480,
    template="plotly_white",
)
st.plotly_chart(fig, use_container_width=True)

# Коэффициенты модели
with st.expander(f"Параметры модели MOEX за {main_res['date']}"):
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Точки кривой (yearyields)**")
        pts_df = pd.DataFrame({"Срок (лет)": main_res["periods"], "Доходность (%)": main_res["yields"]})
        st.dataframe(pts_df, hide_index=True, use_container_width=True)
    with col_b:
        st.markdown("**Коэффициенты модели**")
        st.json(main_res["model_params"])

# ─────────────────────────────────────────────
# РУЧНОЙ РАСЧЁТ G-CURVE
# ─────────────────────────────────────────────
st.markdown("---")

left_block, right_block = st.columns(2)

with left_block:
    st.subheader("Расчёт G-curve по сроку")
    st.caption(f"Дата: **{main_res['date']}**")
    rc1, rc2, rc3 = st.columns([2, 1, 3])
    with rc1:
        manual_m = st.number_input(
            "Срок (лет)", min_value=0.25, max_value=20.0,
            value=3.0, step=0.25, format="%.2f", key="manual_m",
        )
    with rc2:
        calc_btn = st.button("Рассчитать", use_container_width=True)
    with rc3:
        if calc_btn:
            val = gcurve_value(main_res, manual_m)
            st.success(f"G-curve на **{manual_m}** лет = **{math_round(val):.2f}%**")

with right_block:
    st.subheader("Расчёт доходности по спреду")
    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        spread_bp_input = st.number_input(
            "Спред (б.п.)", min_value=-1000.0, max_value=5000.0,
            value=100.0, step=5.0, format="%.0f", key="spread_bp",
        )
    with sc2:
        spread_m_input = st.number_input(
            "Срок (лет)", min_value=0.25, max_value=20.0,
            value=3.0, step=0.25, format="%.2f", key="spread_m",
        )
    with sc3:
        spread_periods_input = st.selectbox(
            "Периодичность", options=[1, 2, 4, 12], index=3, key="spread_periods",
        )

    spread_yield = gcurve_value(main_res, spread_m_input) + spread_bp_input / 100
    spread_coupon = (
        ((1 + spread_yield / 100) ** (1 / spread_periods_input) - 1)
        * 100 * spread_periods_input
    )

    sr1, sr2 = st.columns(2)
    sr1.metric("Доходность", f"{math_round(spread_yield):.2f}%")
    sr2.metric("Купон", f"{math_round(spread_coupon):.2f}%")

# ─────────────────────────────────────────────
# ТАБЛИЦА РАСЧЁТА СПРЕДА
# ─────────────────────────────────────────────
st.markdown("---")
st.subheader("Расчёт спреда для размещения")

info_cols = st.columns(3)
info_cols[0].metric("Дата расчёта", main_res["date"])
info_cols[1].metric("Срок", f"{maturity_input} лет")
info_cols[2].metric("G-curve на этот срок", f"{math_round(g_main):.2f}%")

spread_df = build_spread_table(coupon_input, maturity_input, periods_input, g_main)

# Подсветка базовой строки (индекс 5, смещение = 0)
def highlight_base(row):
    color = "background-color: #dbeafe" if row.name == 5 else ""
    return [color] * len(row)

styled = (
    spread_df.style
    .apply(highlight_base, axis=1)
    .format({
        "Купон, %":                  "{:.2f}",
        "Эффективная доходность, %": "{:.2f}",
        "G-curve, %":                "{:.2f}",
        "Спред, б.п.":               "{:.2f}",
    })
    .bar(subset=["Спред, б.п."], color=["#f4a9a8", "#a8d5b5"], align="zero")
)

st.dataframe(styled, use_container_width=True, hide_index=True, height=423)

# Скачивание CSV
csv_bytes = spread_df.to_csv(index=False).encode("utf-8-sig")
st.download_button(
    label="⬇ Скачать CSV",
    data=csv_bytes,
    file_name=f"gcurve_spread_{main_res['date']}_{coupon_input}pct_{maturity_input}y.csv",
    mime="text/csv",
)

st.markdown("---")
st.caption("Автор проекта: Ширманов К.А. | tg: [shirman7](tg://resolve?domain=shirman7)")
