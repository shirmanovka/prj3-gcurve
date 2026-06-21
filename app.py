import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
from datetime import date, timedelta

# ─────────────────────────────────────────────
# КОНФИГУРАЦИЯ
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="G-curve | MOEX",
    page_icon="📈",
    layout="wide",
)

MOEX_ZCYC_URL = "https://iss.moex.com/iss/engines/stock/zcyc/securities.json"
MATURITIES = np.arange(0.25, 30.25, 0.25)  # 0.25 … 30 лет

# ─────────────────────────────────────────────
# ПОЛУЧЕНИЕ ДАННЫХ С MOEX
# ─────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_zcyc(query_date: str) -> dict | None:
    """
    Запрашивает коэффициенты ZCYC с MOEX ISS.
    Возвращает словарь {beta0, beta1, beta2, tau} или None при ошибке.
    query_date — строка 'YYYY-MM-DD'.
    """
    params = {"date": query_date}
    try:
        resp = requests.get(MOEX_ZCYC_URL, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        st.error(f"Ошибка при обращении к MOEX API: {e}")
        return None

    # Структура ответа: data["params"]["columns"] + data["params"]["data"]
    try:
        cols = data["params"]["columns"]
        rows = data["params"]["data"]
    except KeyError:
        st.error("Неожиданная структура ответа MOEX API.")
        return None

    if not rows:
        return None  # данных за эту дату нет

    # Берём первую строку (обычно одна запись на дату)
    row = dict(zip(cols, rows[0]))
    try:
        return {
            "beta0": float(row["BETA0"]),
            "beta1": float(row["BETA1"]),
            "beta2": float(row["BETA2"]),
            "tau":   float(row["TAU"]),
            "date":  row.get("TRADEDATE", query_date),
        }
    except (KeyError, TypeError, ValueError):
        st.error("Не удалось извлечь коэффициенты из ответа API.")
        return None


# ─────────────────────────────────────────────
# ФОРМУЛА НЕЛЬСОНА-СИГЕЛЯ
# ─────────────────────────────────────────────
def nelson_siegel(m: float | np.ndarray,
                  beta0: float, beta1: float,
                  beta2: float, tau: float) -> float | np.ndarray:
    """
    y(m) = beta0
           + beta1 * (1 - exp(-m/tau)) / (m/tau)
           + beta2 * ((1 - exp(-m/tau)) / (m/tau) - exp(-m/tau))
    Возвращает доходность в % (коэффициенты MOEX уже в %).
    """
    t = m / tau
    factor = (1 - np.exp(-t)) / t
    return beta0 + beta1 * factor + beta2 * (factor - np.exp(-t))


# ─────────────────────────────────────────────
# РАСЧЁТ СПРЕДА
# ─────────────────────────────────────────────
def effective_yield(coupon_pct: float, periods: int) -> float:
    """Эффективная доходность: (1 + c/n)^n - 1, результат в %."""
    return ((1 + coupon_pct / 100 / periods) ** periods - 1) * 100


def build_spread_table(base_coupon: float, maturity: float,
                       periods: int, gcurve_val: float) -> pd.DataFrame:
    """
    Строит таблицу 11 строк:
    купон ±25/20/15/10/5 б.п. от base_coupon (б.п. = 0.01 %).
    """
    offsets_bp = [-25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25]
    rows = []
    for bp in offsets_bp:
        c = base_coupon + bp * 0.01
        ey = effective_yield(c, periods)
        spread = (ey - gcurve_val) * 100  # уже в б.п. (% → б.п. = *100)
        rows.append({
            "Купон, %": round(c, 4),
            "Периодичность": periods,
            "Эффективная доходность, %": round(ey, 4),
            "G-curve, %": round(gcurve_val, 4),
            "Спред, б.п.": round(spread, 2),
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
    value=date.today() - timedelta(days=1),
    max_value=date.today(),
)

# Дополнительные даты для сравнения
st.sidebar.markdown("---")
st.sidebar.markdown("**Дополнительные даты для сравнения**")

# Храним список доп. дат в session_state
if "extra_dates" not in st.session_state:
    st.session_state.extra_dates: list[date] = []

col_add, col_clear = st.sidebar.columns([3, 1])
with col_add:
    extra_date_input: date = st.date_input(
        "Добавить дату",
        value=date.today() - timedelta(days=7),
        max_value=date.today(),
        label_visibility="collapsed",
    )
with col_clear:
    if st.button("＋", help="Добавить дату на график"):
        if extra_date_input not in st.session_state.extra_dates and extra_date_input != main_date:
            st.session_state.extra_dates.append(extra_date_input)

# Отображаем добавленные даты с кнопкой удаления
for i, d in enumerate(list(st.session_state.extra_dates)):
    c1, c2 = st.sidebar.columns([4, 1])
    c1.write(str(d))
    if c2.button("✕", key=f"del_{i}"):
        st.session_state.extra_dates.pop(i)
        st.rerun()

# Параметры облигации
st.sidebar.markdown("---")
st.sidebar.markdown("**Параметры облигации**")

coupon_input = st.sidebar.number_input(
    "Купон (%)", min_value=0.0, max_value=100.0,
    value=13.0, step=0.05, format="%.2f",
)
maturity_input = st.sidebar.number_input(
    "Срок до погашения (лет)", min_value=0.25, max_value=30.0,
    value=3.0, step=0.25, format="%.2f",
)
periods_input = st.sidebar.selectbox(
    "Периодичность выплат", options=[1, 2, 4, 12], index=1,
)

# ─────────────────────────────────────────────
# ЗАГРУЗКА ДАННЫХ ДЛЯ ОСНОВНОЙ ДАТЫ
# ─────────────────────────────────────────────
st.title("Кривая бескупонной доходности (G-curve)")
st.caption("Модель Нельсона–Сигеля | Данные MOEX ISS")

with st.spinner(f"Загрузка данных MOEX за {main_date}..."):
    main_params = fetch_zcyc(str(main_date))

if main_params is None:
    st.error(
        f"Данные за **{main_date}** недоступны. "
        "Попробуйте другую дату (торговые дни) или нажмите «Обновить данные»."
    )
    st.stop()

# ─────────────────────────────────────────────
# ПОСТРОЕНИЕ ГРАФИКА
# ─────────────────────────────────────────────
fig = go.Figure()

# Основная кривая
y_main = nelson_siegel(MATURITIES, **{k: main_params[k] for k in ("beta0", "beta1", "beta2", "tau")})
fig.add_trace(go.Scatter(
    x=MATURITIES, y=y_main,
    mode="lines",
    name=f"Основная дата: {main_date}",
    line=dict(width=3, color="#1f77b4"),
))

# Дополнительные кривые
palette = ["#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]
for idx, extra_d in enumerate(st.session_state.extra_dates):
    with st.spinner(f"Загрузка данных MOEX за {extra_d}..."):
        ep = fetch_zcyc(str(extra_d))
    if ep is None:
        st.warning(f"Данные за {extra_d} недоступны, дата пропущена.")
        continue
    y_extra = nelson_siegel(MATURITIES, **{k: ep[k] for k in ("beta0", "beta1", "beta2", "tau")})
    fig.add_trace(go.Scatter(
        x=MATURITIES, y=y_extra,
        mode="lines",
        name=str(extra_d),
        line=dict(width=1.5, dash="dash", color=palette[idx % len(palette)]),
    ))

fig.update_layout(
    title=dict(
        text=f"G-curve | Основная дата: {main_date}" +
             (f" + {len(st.session_state.extra_dates)} доп." if st.session_state.extra_dates else ""),
        font=dict(size=16),
    ),
    xaxis_title="Срок (годы)",
    yaxis_title="Доходность (%)",
    legend=dict(orientation="h", y=-0.15),
    hovermode="x unified",
    height=480,
    template="plotly_white",
)

st.plotly_chart(fig, use_container_width=True)

# Коэффициенты в expander
with st.expander(f"Коэффициенты модели за {main_date}"):
    st.json({
        "beta0": main_params["beta0"],
        "beta1": main_params["beta1"],
        "beta2": main_params["beta2"],
        "tau":   main_params["tau"],
        "дата":  main_params["date"],
    })

# ─────────────────────────────────────────────
# РУЧНОЙ РАСЧЁТ G-CURVE
# ─────────────────────────────────────────────
st.markdown("---")
st.subheader("Ручной расчёт G-curve")
st.caption(f"Используется основная дата: **{main_date}**")

col1, col2, col3 = st.columns([2, 1, 3])
with col1:
    manual_m = st.number_input(
        "Срок (лет)", min_value=0.25, max_value=30.0,
        value=3.0, step=0.25, format="%.2f", key="manual_m",
    )
with col2:
    calc_btn = st.button("Рассчитать", use_container_width=True)
with col3:
    if calc_btn:
        val = nelson_siegel(
            manual_m,
            main_params["beta0"], main_params["beta1"],
            main_params["beta2"], main_params["tau"],
        )
        st.success(f"Доходность на срок **{manual_m}** лет: **{val:.4f}%**")

# ─────────────────────────────────────────────
# ТАБЛИЦА РАСЧЁТА СПРЕДА
# ─────────────────────────────────────────────
st.markdown("---")
st.subheader("Расчёт спреда для размещения")
st.caption(
    f"Основная дата: **{main_date}** | "
    f"Срок: **{maturity_input} лет** | "
    f"Периодичность: **{periods_input}** раз/год"
)

# Значение G-curve для выбранного срока
gcurve_for_maturity = nelson_siegel(
    maturity_input,
    main_params["beta0"], main_params["beta1"],
    main_params["beta2"], main_params["tau"],
)
st.info(f"G-curve на срок {maturity_input} лет: **{gcurve_for_maturity:.4f}%**")

spread_df = build_spread_table(
    coupon_input, maturity_input, periods_input, gcurve_for_maturity
)

# Подсветка базовой строки (середина, индекс 5)
def highlight_base(row):
    if row.name == 5:
        return ["background-color: #dbeafe"] * len(row)
    return [""] * len(row)

styled = spread_df.style.apply(highlight_base, axis=1).format({
    "Купон, %": "{:.4f}",
    "Эффективная доходность, %": "{:.4f}",
    "G-curve, %": "{:.4f}",
    "Спред, б.п.": "{:.2f}",
})

st.dataframe(styled, use_container_width=True, hide_index=True)

# Кнопка скачивания CSV
csv_bytes = spread_df.to_csv(index=False).encode("utf-8-sig")
st.download_button(
    label="Скачать таблицу CSV",
    data=csv_bytes,
    file_name=f"gcurve_spread_{main_date}.csv",
    mime="text/csv",
)

st.markdown("---")
st.caption("Источник данных: MOEX ISS | Модель: Nelson–Siegel")
