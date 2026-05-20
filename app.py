# =============================================================================
#  PRODUCTIVIDAD DE PERFORACIÓN MINERA
#  Jumbo Hidráulico · Jack Leg · DTH · Simba
#  Simulación Monte Carlo + Análisis de Costos + Comparación de Equipos
#  Autor: Ing. de Minas — Universidad Nacional del Altiplano Puno
# =============================================================================

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.stats import truncnorm, gaussian_kde

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIGURACIÓN DE PÁGINA
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Productividad de Perforación Minera",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
#  CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main-header {
    background: linear-gradient(135deg, #0a0a1a 0%, #12122a 40%, #0f3460 100%);
    padding: 2.2rem 2rem; border-radius: 16px; margin-bottom: 1.8rem;
    text-align: center; border: 1px solid #e9456033;
    box-shadow: 0 8px 32px rgba(233,69,96,0.2);
}
.main-header h1 { color: #e94560; font-size: 2.4rem; margin: 0; font-weight: 700; letter-spacing: -0.5px; }
.main-header p  { color: #a8b2d8; font-size: 1rem; margin-top: 0.6rem; }
.main-header .badge {
    display: inline-block; background: #e9456022; color: #e94560;
    border: 1px solid #e9456044; border-radius: 20px;
    padding: 0.2rem 0.8rem; font-size: 0.78rem; margin: 0.2rem;
}

.kpi-card {
    background: linear-gradient(135deg, #16213e, #1a1a2e);
    border: 1px solid #e9456033; border-radius: 12px; padding: 1.3rem;
    text-align: center; box-shadow: 0 4px 16px rgba(233,69,96,0.1);
    transition: transform .2s;
}
.kpi-card:hover { transform: translateY(-2px); border-color: #e94560aa; }
.kpi-value { font-size: 2.1rem; font-weight: 700; color: #e94560; line-height: 1; }
.kpi-label { font-size: 0.82rem; color: #a8b2d8; margin-top: 0.4rem; font-weight: 500; }
.kpi-unit  { font-size: 0.72rem; color: #6b7faa; margin-top: 0.2rem; }

.section-title {
    color: #e94560; font-size: 1.25rem; font-weight: 600;
    border-left: 4px solid #e94560; padding-left: 0.9rem;
    margin: 1.8rem 0 1rem 0;
}
.info-box {
    background: #0f346022; border-left: 3px solid #e94560;
    padding: 0.9rem 1.1rem; border-radius: 0 10px 10px 0;
    color: #a8b2d8; font-size: 0.88rem; margin-bottom: 1.2rem;
}
.equipo-card {
    background: linear-gradient(135deg, #16213e, #0d0d1a);
    border-radius: 12px; padding: 1rem 1.2rem;
    border: 1px solid #2a3a5e; margin-bottom: 0.5rem;
}
[data-testid="stSidebar"] { background: #0a0a1a; }
[data-testid="stSidebar"] .block-container { padding: 1rem; }
.stTabs [data-baseweb="tab-list"] { background: #12122a; border-radius: 8px; padding: 4px; }
.stTabs [data-baseweb="tab"] { color: #a8b2d8; border-radius: 6px; padding: 0.5rem 1rem; }
.stTabs [aria-selected="true"] { background: #e9456022 !important; color: #e94560 !important; font-weight: 600; }
div[data-testid="metric-container"] { background: #16213e; border-radius: 8px; padding: 0.8rem; border: 1px solid #2a3a5e; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIGURACIÓN DE EQUIPOS (todo inline)
# ─────────────────────────────────────────────────────────────────────────────
EQUIPOS = {
    "⛏️ Jumbo Hidráulico (Boomer)": {
        "desc": "Equipo electrohidráulico de 2 brazos para avances y galerías en minería subterránea.",
        "tipo": "Rotopercutivo", "n_brazos": 2, "diam_mm": 45,
        "vp": 45.0,  "dm": 90, "u": 80,  "ucs": 100, "cai": 2.0, "t_pos": 3.0,
        "c_eq": 75.0, "c_mo": 18.0, "c_mant": 12.0, "c_cons": 6.0,
        "p_broca": 230.0, "vu_broca": 250.0, "p_barra": 800.0, "vu_barra": 1000.0,
        "color": "#e94560", "rango_vp": (20, 80), "rango_dm": (85, 95),
        "app": "Galerías de desarrollo, túneles, avances",
    },
    "🔩 Jack Leg (Neumático)": {
        "desc": "Perforadora neumática manual. Un solo operador. Bajo costo de inversión.",
        "tipo": "Rotopercutivo", "n_brazos": 1, "diam_mm": 38,
        "vp": 18.0,  "dm": 80, "u": 70,  "ucs": 80,  "cai": 1.5, "t_pos": 8.0,
        "c_eq": 12.0, "c_mo": 8.0,  "c_mant": 3.0,  "c_cons": 2.0,
        "p_broca": 45.0,  "vu_broca": 150.0, "p_barra": 120.0, "vu_barra": 400.0,
        "color": "#00d4aa", "rango_vp": (8, 35),  "rango_dm": (70, 88),
        "app": "Labores estrechas, pequeña minería, artesanal",
    },
    "💥 DTH (Martillo en Fondo)": {
        "desc": "Down-The-Hole. Percusión directa en la broca. Ideal para diámetros grandes y rocas duras.",
        "tipo": "DTH", "n_brazos": 1, "diam_mm": 115,
        "vp": 30.0,  "dm": 85, "u": 75,  "ucs": 150, "cai": 3.0, "t_pos": 5.0,
        "c_eq": 55.0, "c_mo": 15.0, "c_mant": 10.0, "c_cons": 8.0,
        "p_broca": 450.0, "vu_broca": 400.0, "p_barra": 600.0, "vu_barra": 800.0,
        "color": "#f39c12", "rango_vp": (15, 55), "rango_dm": (80, 92),
        "app": "Taladros largos, chimeneas, piques, cielo abierto",
    },
    "🎯 Simba (Perforadora Radial)": {
        "desc": "Perforadora radial electrohidráulica de producción. Taladros largos y stope drilling.",
        "tipo": "Rotopercutivo", "n_brazos": 1, "diam_mm": 64,
        "vp": 35.0,  "dm": 88, "u": 82,  "ucs": 120, "cai": 2.5, "t_pos": 4.0,
        "c_eq": 60.0, "c_mo": 16.0, "c_mant": 9.0,  "c_cons": 5.0,
        "p_broca": 320.0, "vu_broca": 300.0, "p_barra": 500.0, "vu_barra": 700.0,
        "color": "#9b59b6", "rango_vp": (18, 60), "rango_dm": (82, 93),
        "app": "Tajeos largos, stope drilling, banco de producción",
    },
}

# ─────────────────────────────────────────────────────────────────────────────
#  LAYOUT BASE PLOTLY
# ─────────────────────────────────────────────────────────────────────────────
DARK  = "#0a0a1a"
CARD  = "#16213e"
GRID  = "#1e2a4a"
TEXT  = "#a8b2d8"
R     = "#e94560"
G     = "#00d4aa"
Y     = "#f39c12"
P     = "#9b59b6"
B     = "#3498db"

def base_layout(title="", h=440):
    return dict(
        paper_bgcolor=DARK, plot_bgcolor=DARK,
        font=dict(color=TEXT, family="Inter, Arial", size=12),
        title=dict(text=title, font=dict(color="#ffffff", size=14), x=0.01),
        margin=dict(l=50, r=20, t=50, b=45), height=h,
        legend=dict(bgcolor=CARD, bordercolor=GRID, borderwidth=1, font=dict(size=11)),
        xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, color=TEXT),
        yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, color=TEXT),
    )

# ─────────────────────────────────────────────────────────────────────────────
#  FUNCIONES DE CÁLCULO (todo inline)
# ─────────────────────────────────────────────────────────────────────────────
def calcular(vp_campo, dm, u, t_guardia, t_pos,
             c_eq, c_mo, c_mant, c_cons_h,
             p_broca, vu_broca, p_barra, vu_barra,
             ucs, cai, rqd, n_brazos):
    """
    Fórmulas base de productividad:
      VPef   = VP × U × DM
      Tef    = TG × U × DM
      MP/g   = VPef × Tef × N_brazos
      Ch     = Ceq + Cmo + Cmant + Ccons
      Cm     = Ch / VPef
      TDC    = (P_broca/VU_broca) + Cm
    """
    vp_ef    = vp_campo * u * dm
    t_ef     = t_guardia * u * dm
    mp_g     = vp_ef * t_ef * n_brazos
    ch       = c_eq + c_mo + c_mant + c_cons_h
    cm       = ch / vp_ef if vp_ef > 0 else 0
    c_br_m   = p_broca  / vu_broca
    c_ba_m   = p_barra  / vu_barra
    c_cons_m = c_br_m + c_ba_m
    tdc      = c_br_m + cm
    c_dir    = cm + c_cons_m
    # Corrección roca (Bauer-Calder simplificado)
    f_roca   = max(0.45, min(1.35, 1.0 - (ucs - 80)*0.002 - (cai - 2.0)*0.05))
    vp_corr  = vp_ef * f_roca
    return {
        "vp_ef": round(vp_ef, 3),   "vp_corr": round(vp_corr, 3),
        "t_ef":  round(t_ef, 3),    "mp_g":    round(mp_g, 3),
        "ch":    round(ch, 3),      "cm":      round(cm, 4),
        "c_br_m":round(c_br_m,4),   "c_ba_m":  round(c_ba_m, 4),
        "c_cons_m":round(c_cons_m,4),"tdc":    round(tdc, 4),
        "c_dir": round(c_dir, 4),
    }


def monte_carlo(vp_campo, dm, u, t_guardia, c_eq, c_mo, c_mant, c_cons_h,
                p_broca, vu_broca, p_barra, vu_barra, n_brazos,
                n_sim, cv_vp, cv_dm, cv_u, cv_cost, ucs, cai, rqd):
    """
    Simulación Monte Carlo con distribución normal truncada.
    Varía VP, DM, U y todos los costos operativos.
    """
    np.random.seed(42)

    def tnorm(mu, cv, n, lo=0.3, hi=2.5):
        s = mu * cv
        if s < 1e-9:
            return np.full(n, mu)
        a, b = (mu*lo - mu)/s, (mu*hi - mu)/s
        return truncnorm.rvs(a, b, loc=mu, scale=s, size=n)

    vp_s  = tnorm(vp_campo, cv_vp,  n_sim)
    dm_s  = np.clip(tnorm(dm,       cv_dm,  n_sim, 0.6, 1.15), 0.40, 0.99)
    u_s   = np.clip(tnorm(u,        cv_u,   n_sim, 0.6, 1.15), 0.40, 0.99)
    ceq_s = tnorm(c_eq,   cv_cost, n_sim)
    cmo_s = tnorm(c_mo,   cv_cost, n_sim)
    cmt_s = tnorm(c_mant, cv_cost, n_sim)
    ccs_s = tnorm(c_cons_h, cv_cost, n_sim)

    vp_ef_s = vp_s * u_s * dm_s
    t_ef_s  = t_guardia * u_s * dm_s
    mp_g_s  = vp_ef_s * t_ef_s * n_brazos
    ch_s    = ceq_s + cmo_s + cmt_s + ccs_s
    cm_s    = np.where(vp_ef_s > 0, ch_s / vp_ef_s, 0)
    tdc_s   = (p_broca / vu_broca) + cm_s

    return {"vp_s": vp_ef_s, "mp_s": mp_g_s, "cm_s": cm_s, "tdc_s": tdc_s}

# ─────────────────────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Parámetros de Entrada")
    equipo_sel = st.selectbox("🔩 Equipo", list(EQUIPOS.keys()))
    cfg = EQUIPOS[equipo_sel]

    st.markdown("### 🪨 Roca")
    ucs    = st.slider("UCS (MPa)", 20, 300, cfg["ucs"], 5)
    cai    = st.slider("CAI – Índice Cerchar", 0.1, 5.0, cfg["cai"], 0.1)
    rqd    = st.slider("RQD (%)", 10, 100, 75, 5)
    f_prot = round(ucs / 10, 1)
    st.caption(f"f Protodyakonov = **{f_prot}**")

    st.markdown("### ⏱️ Operativo")
    vp_campo  = st.number_input("VP campo (m/h)", 3.0, 150.0, float(cfg["vp"]), 0.5)
    dm        = st.slider("Disponib. Mecánica (%)", 60, 99, cfg["dm"])
    u_eq      = st.slider("Utilización (%)", 50, 99, cfg["u"])
    t_guard   = st.selectbox("Guardia (h)", [8, 10, 12])
    t_pos     = st.number_input("Posicionam. (min/tal)", 1.0, 30.0, float(cfg["t_pos"]), 0.5)

    st.markdown("### 💰 Costos (USD/h)")
    c_eq   = st.number_input("Equipo",         5.0,  500.0, float(cfg["c_eq"]),   1.0)
    c_mo   = st.number_input("Mano de obra",   3.0,  100.0, float(cfg["c_mo"]),   0.5)
    c_mant = st.number_input("Mantenimiento",  1.0,   80.0, float(cfg["c_mant"]), 0.5)
    c_cons = st.number_input("Consumibles/h",  0.5,   50.0, float(cfg["c_cons"]), 0.5)

    st.markdown("### 🔧 Aceros")
    p_broca  = st.number_input("Precio broca ($)",    10.0,  800.0, float(cfg["p_broca"]),  5.0)
    vu_broca = st.number_input("Vida broca (m)",       30.0,  600.0, float(cfg["vu_broca"]), 10.0)
    p_barra  = st.number_input("Precio barra ($)",    30.0, 2000.0, float(cfg["p_barra"]),  10.0)
    vu_barra = st.number_input("Vida barra (m)",      100.0, 2000.0, float(cfg["vu_barra"]), 50.0)

    st.markdown("### 🎲 Monte Carlo")
    n_sim   = st.selectbox("N° Simulaciones", [1000, 5000, 10000, 50000], index=2)
    cv_vp   = st.slider("CV Velocidad (%)",   2, 30, 10)
    cv_dm   = st.slider("CV Disponib. (%)",   2, 20,  5)
    cv_u    = st.slider("CV Utilización (%)", 2, 20,  5)
    cv_cost = st.slider("CV Costos (%)",      2, 25,  8)

    st.markdown("---")
    calcular_btn = st.button("🚀 CALCULAR TODO", use_container_width=True, type="primary")

# ─────────────────────────────────────────────────────────────────────────────
#  CABECERA PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="main-header">
    <h1>⛏️ Productividad de Perforación Minera</h1>
    <p>Simulación Monte Carlo · Análisis de Costos · Comparación de Equipos</p>
    <span class="badge">Jumbo Hidráulico</span>
    <span class="badge">Jack Leg</span>
    <span class="badge">DTH</span>
    <span class="badge">Simba</span>
    <span class="badge">Monte Carlo {n_sim:,} iter.</span>
    <span class="badge">UCS = {ucs} MPa · f = {f_prot}</span>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  CÁLCULOS DETERMINISTAS
# ─────────────────────────────────────────────────────────────────────────────
res = calcular(
    vp_campo=vp_campo, dm=dm/100, u=u_eq/100,
    t_guardia=t_guard, t_pos=t_pos,
    c_eq=c_eq, c_mo=c_mo, c_mant=c_mant, c_cons_h=c_cons,
    p_broca=p_broca, vu_broca=vu_broca,
    p_barra=p_barra, vu_barra=vu_barra,
    ucs=ucs, cai=cai, rqd=rqd,
    n_brazos=cfg["n_brazos"],
)

# ─────────────────────────────────────────────────────────────────────────────
#  KPIs
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f'<p class="section-title">📊 Resultados — {equipo_sel}</p>', unsafe_allow_html=True)
k1, k2, k3, k4, k5, k6 = st.columns(6)
kpis = [
    (k1, res["vp_ef"],    "VP Efectiva",        "m/h"),
    (k2, res["t_ef"],     "Tiempo Efectivo",    "h/guardia"),
    (k3, res["mp_g"],     "Metros/Guardia",     "m"),
    (k4, res["ch"],       "Costo Horario",      "USD/h"),
    (k5, res["cm"],       "Costo/Metro",        "USD/m"),
    (k6, res["tdc"],      "TDC",                "USD/m"),
]
for col, val, lbl, unit in kpis:
    col.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value">{val:.2f}</div>
        <div class="kpi-label">{lbl}</div>
        <div class="kpi-unit">{unit}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🎲 Monte Carlo",
    "📈 Sensibilidad",
    "⚙️ Comparación de Equipos",
    "💰 Costos",
    "📉 VP vs Roca",
    "📋 Memoria de Cálculo",
])

# ═══════════════════════════════════════════════════════════════════════
#  TAB 1 — MONTE CARLO
# ═══════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<p class="section-title">🎲 Simulación Monte Carlo</p>', unsafe_allow_html=True)
    st.markdown(f"""<div class="info-box">
    <b>{n_sim:,} iteraciones</b> con distribución normal truncada. Variables aleatorias:
    VP (CV={cv_vp}%), DM (CV={cv_dm}%), U (CV={cv_u}%), Costos (CV={cv_cost}%).
    La curva blanca es la KDE ajustada. Las líneas muestran P10 · P50 · P90.
    </div>""", unsafe_allow_html=True)

    mc = monte_carlo(
        vp_campo, dm/100, u_eq/100, t_guard,
        c_eq, c_mo, c_mant, c_cons,
        p_broca, vu_broca, p_barra, vu_barra,
        cfg["n_brazos"], n_sim,
        cv_vp/100, cv_dm/100, cv_u/100, cv_cost/100,
        ucs, cai, rqd,
    )

    def hist_kde(data, nombre, unidad, color):
        p10, p50, p90 = np.percentile(data, [10, 50, 90])
        mu, sig = np.mean(data), np.std(data)
        nbins = 60
        bw = (data.max() - data.min()) / nbins
        kde_x = np.linspace(data.min(), data.max(), 400)
        kde_y = gaussian_kde(data)(kde_x) * len(data) * bw
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=data, nbinsx=nbins, name="Frecuencia",
                                   marker=dict(color=color, opacity=0.75, line=dict(color=DARK, width=0.3))))
        fig.add_trace(go.Scatter(x=kde_x, y=kde_y, mode="lines", name="KDE",
                                 line=dict(color="#ffffff", width=2.2)))
        for pv, pl, pc in [(p10,"P10",Y),(p50,"P50","#ffffff"),(p90,"P90",G)]:
            fig.add_vline(x=pv, line=dict(color=pc, dash="dash", width=1.5),
                          annotation=dict(text=f"{pl}={pv:.2f}", font=dict(color=pc, size=10), y=1.06, yref="paper"))
        fig.add_annotation(x=0.97, y=0.97, xref="paper", yref="paper",
                           text=f"μ={mu:.2f}  σ={sig:.2f}  CV={(sig/mu*100 if mu else 0):.1f}%",
                           showarrow=False, font=dict(color=TEXT, size=11),
                           bgcolor=CARD, bordercolor=GRID, borderwidth=1)
        fig.update_layout(**base_layout(f"Monte Carlo — {nombre}"))
        fig.update_xaxes(title_text=unidad); fig.update_yaxes(title_text="Frecuencia")
        return fig

    c1, c2 = st.columns(2)
    with c1: st.plotly_chart(hist_kde(mc["mp_s"], "Metros / Guardia", "m/guardia", R), use_container_width=True)
    with c2: st.plotly_chart(hist_kde(mc["cm_s"], "Costo por Metro",  "USD/m",     G), use_container_width=True)
    c3, c4 = st.columns(2)
    with c3: st.plotly_chart(hist_kde(mc["vp_s"], "VP Efectiva",      "m/h",       Y), use_container_width=True)
    with c4: st.plotly_chart(hist_kde(mc["tdc_s"],"TDC",              "USD/m",     P), use_container_width=True)

    # Scatter VP vs Costo
    st.markdown('<p class="section-title">🔀 Dispersión VP vs Costo/Metro (muestra 3,000 pts)</p>', unsafe_allow_html=True)
    idx = np.random.choice(len(mc["vp_s"]), min(3000, n_sim), replace=False)
    fig_sc = go.Figure(go.Scatter(
        x=mc["vp_s"][idx], y=mc["cm_s"][idx], mode="markers",
        marker=dict(color=mc["mp_s"][idx], colorscale="RdYlGn", size=4, opacity=0.6,
                    colorbar=dict(title="MP/g", thickness=12, x=1.01)),
    ))
    fig_sc.update_layout(**base_layout("Dispersión VP Efectiva vs Costo/m (color = MP/guardia)", h=380))
    fig_sc.update_xaxes(title_text="VP Efectiva (m/h)")
    fig_sc.update_yaxes(title_text="Costo/Metro (USD/m)")
    st.plotly_chart(fig_sc, use_container_width=True)

    # Tabla percentiles
    st.markdown('<p class="section-title">📊 Tabla de Percentiles</p>', unsafe_allow_html=True)
    pcts = [5, 10, 25, 50, 75, 90, 95]
    df_pct = pd.DataFrame({
        "Percentil (%)":      pcts,
        "MP/Guardia (m)":     np.round([np.percentile(mc["mp_s"], p) for p in pcts], 2),
        "Costo/m (USD)":      np.round([np.percentile(mc["cm_s"], p) for p in pcts], 3),
        "VP Efectiva (m/h)":  np.round([np.percentile(mc["vp_s"], p) for p in pcts], 2),
        "TDC (USD/m)":        np.round([np.percentile(mc["tdc_s"],p) for p in pcts], 3),
    })
    st.dataframe(
        df_pct.style
              .background_gradient(cmap="RdYlGn",   subset=["MP/Guardia (m)"])
              .background_gradient(cmap="RdYlGn_r", subset=["Costo/m (USD)", "TDC (USD/m)"]),
        use_container_width=True, hide_index=True,
    )

# ═══════════════════════════════════════════════════════════════════════
#  TAB 2 — SENSIBILIDAD
# ═══════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<p class="section-title">📈 Análisis de Sensibilidad ±30%</p>', unsafe_allow_html=True)
    st.markdown("""<div class="info-box">
    Cada variable se varía ±30% respecto a su valor base mientras las demás permanecen constantes.
    El diagrama de Tornado muestra el swing total (impacto) ordenado de mayor a menor.
    </div>""", unsafe_allow_html=True)

    VARS_SENS = {
        "VP Campo":          ("vp_campo",  vp_campo),
        "Disp. Mecánica":    ("dm",        dm/100),
        "Utilización":       ("u",         u_eq/100),
        "Costo Equipo":      ("c_eq",      c_eq),
        "Costo M.O.":        ("c_mo",      c_mo),
        "Costo Mant.":       ("c_mant",    c_mant),
        "Precio Broca":      ("p_broca",   p_broca),
        "Vida Útil Broca":   ("vu_broca",  vu_broca),
        "Precio Barra":      ("p_barra",   p_barra),
    }
    PARAMS_BASE = dict(vp_campo=vp_campo, dm=dm/100, u=u_eq/100, t_guardia=t_guard, t_pos=t_pos,
                       c_eq=c_eq, c_mo=c_mo, c_mant=c_mant, c_cons_h=c_cons,
                       p_broca=p_broca, vu_broca=vu_broca, p_barra=p_barra, vu_barra=vu_barra,
                       ucs=ucs, cai=cai, rqd=rqd, n_brazos=cfg["n_brazos"])
    pct_range = np.linspace(-0.30, 0.30, 13)
    colors_s  = [R, G, Y, P, B, "#e67e22", "#1abc9c", "#e74c3c", "#27ae60"]

    # Spider chart
    fig_spider = go.Figure()
    for i, (lbl, (key, base)) in enumerate(VARS_SENS.items()):
        ys = []
        for pct in pct_range:
            p = dict(PARAMS_BASE); p[key] = base * (1 + pct)
            ys.append(calcular(**p)["mp_g"])
        fig_spider.add_trace(go.Scatter(x=pct_range*100, y=ys, mode="lines+markers",
                                        name=lbl, line=dict(color=colors_s[i%len(colors_s)], width=2),
                                        marker=dict(size=4)))
    base_mp = res["mp_g"]
    fig_spider.add_hline(y=base_mp, line=dict(color="#ffffff", dash="dot", width=1.2),
                         annotation=dict(text=f"Base={base_mp:.2f} m", font=dict(color="#ffffff", size=10)))
    fig_spider.update_layout(**base_layout("Análisis de Sensibilidad — Metros/Guardia"))
    fig_spider.update_xaxes(title_text="Variación (%)", ticksuffix="%")
    fig_spider.update_yaxes(title_text="Metros / Guardia")

    # Tornado
    impactos = []
    for lbl, (key, base) in VARS_SENS.items():
        p_lo = dict(PARAMS_BASE); p_lo[key]  = base * 0.80
        p_hi = dict(PARAMS_BASE); p_hi[key]  = base * 1.20
        lo   = calcular(**p_lo)["mp_g"]
        hi   = calcular(**p_hi)["mp_g"]
        impactos.append(dict(lbl=lbl, lo=lo, hi=hi, swing=hi-lo))
    impactos.sort(key=lambda x: x["swing"])
    lbls_t = [d["lbl"] for d in impactos]
    lo_t   = [d["lo"]  for d in impactos]
    hi_t   = [d["hi"]  for d in impactos]

    fig_torn = go.Figure()
    fig_torn.add_trace(go.Bar(y=lbls_t, x=[h - base_mp for h in hi_t],
                              orientation="h", name="+20%", marker_color=G, base=base_mp))
    fig_torn.add_trace(go.Bar(y=lbls_t, x=[l - base_mp for l in lo_t],
                              orientation="h", name="-20%", marker_color=R, base=base_mp))
    fig_torn.add_vline(x=base_mp, line=dict(color="#ffffff", width=1.5, dash="dash"))
    fig_torn.update_layout(**base_layout("Diagrama de Tornado — Impacto en MP/Guardia"), barmode="overlay")
    fig_torn.update_xaxes(title_text="Metros / Guardia")

    c1, c2 = st.columns(2)
    with c1: st.plotly_chart(fig_spider, use_container_width=True)
    with c2: st.plotly_chart(fig_torn,   use_container_width=True)

    # Sensibilidad sobre Costo/m
    st.markdown('<p class="section-title">📉 Sensibilidad — Costo por Metro</p>', unsafe_allow_html=True)
    fig_cm_sens = go.Figure()
    for i, (lbl, (key, base)) in enumerate(VARS_SENS.items()):
        ys = []
        for pct in pct_range:
            p = dict(PARAMS_BASE); p[key] = base * (1 + pct)
            ys.append(calcular(**p)["cm"])
        fig_cm_sens.add_trace(go.Scatter(x=pct_range*100, y=ys, mode="lines+markers",
                                          name=lbl, line=dict(color=colors_s[i%len(colors_s)], width=2),
                                          marker=dict(size=4)))
    fig_cm_sens.add_hline(y=res["cm"], line=dict(color="#ffffff", dash="dot", width=1.2))
    fig_cm_sens.update_layout(**base_layout("Sensibilidad — Costo/Metro (USD/m)"))
    fig_cm_sens.update_xaxes(title_text="Variación (%)", ticksuffix="%")
    fig_cm_sens.update_yaxes(title_text="USD/m")
    st.plotly_chart(fig_cm_sens, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════
#  TAB 3 — COMPARACIÓN DE EQUIPOS
# ═══════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<p class="section-title">⚙️ Comparación de los 4 Equipos</p>', unsafe_allow_html=True)
    st.markdown(f"""<div class="info-box">
    Misma condición de roca: UCS={ucs} MPa · CAI={cai} · RQD={rqd}%.
    Se usan los parámetros típicos de cada equipo. Guardia de {t_guard} h.
    </div>""", unsafe_allow_html=True)

    rows = []
    for nombre, c in EQUIPOS.items():
        r = calcular(
            vp_campo=c["vp"], dm=c["dm"]/100, u=c["u"]/100,
            t_guardia=t_guard, t_pos=c["t_pos"],
            c_eq=c["c_eq"], c_mo=c["c_mo"], c_mant=c["c_mant"], c_cons_h=c["c_cons"],
            p_broca=c["p_broca"], vu_broca=c["vu_broca"],
            p_barra=c["p_barra"], vu_barra=c["vu_barra"],
            ucs=ucs, cai=cai, rqd=rqd, n_brazos=c["n_brazos"],
        )
        rows.append({**r, "equipo": nombre, "color": c["color"], "app": c["app"], "tipo": c["tipo"],
                     "diam": c["diam_mm"], "dm_cfg": c["dm"], "u_cfg": c["u"]})

    nombres   = [r["equipo"].split("(")[0].strip() for r in rows]
    mp_vals   = [r["mp_g"] for r in rows]
    cm_vals   = [r["cm"]   for r in rows]
    vp_vals   = [r["vp_ef"]for r in rows]
    tdc_vals  = [r["tdc"]  for r in rows]
    colors_eq = [r["color"] for r in rows]

    # Gráfica barras 2x2
    fig_comp = make_subplots(rows=2, cols=2,
                             subplot_titles=("Metros/Guardia (m)", "VP Efectiva (m/h)",
                                             "Costo/Metro (USD/m)", "TDC (USD/m)"),
                             vertical_spacing=0.16, horizontal_spacing=0.12)
    for vals, row, col, fmt in [(mp_vals,1,1,"{:.1f}"), (vp_vals,1,2,"{:.2f}"),
                                 (cm_vals,2,1,"{:.3f}"), (tdc_vals,2,2,"{:.3f}")]:
        fig_comp.add_trace(go.Bar(x=nombres, y=vals, marker=dict(color=colors_eq),
                                  text=[fmt.format(v) for v in vals], textposition="outside",
                                  showlegend=False), row=row, col=col)
    fig_comp.update_layout(**base_layout("Comparación de Equipos", h=560))
    fig_comp.update_layout(paper_bgcolor=DARK, plot_bgcolor=DARK,
                           font=dict(color=TEXT), margin=dict(l=50,r=20,t=60,b=40))
    for ax in ["xaxis","xaxis2","xaxis3","xaxis4"]:
        fig_comp.update_layout(**{ax: dict(gridcolor=GRID, color=TEXT)})
    for ax in ["yaxis","yaxis2","yaxis3","yaxis4"]:
        fig_comp.update_layout(**{ax: dict(gridcolor=GRID, color=TEXT)})
    st.plotly_chart(fig_comp, use_container_width=True)

    # Scatter costo vs MP/g
    fig_sc2 = go.Figure()
    for r in rows:
        eq_short = r["equipo"].split("(")[0].strip().replace("⛏️","").replace("🔩","").replace("💥","").replace("🎯","").strip()
        fig_sc2.add_trace(go.Scatter(
            x=[r["cm"]], y=[r["mp_g"]], mode="markers+text",
            name=eq_short, text=[eq_short], textposition="top center",
            marker=dict(color=r["color"], size=24, symbol="circle",
                        line=dict(color="#ffffff", width=2)),
            textfont=dict(color="#ffffff", size=11),
        ))
    fig_sc2.update_layout(**base_layout("Costo/m vs Productividad (MP/Guardia)", h=420))
    fig_sc2.update_xaxes(title_text="Costo por Metro (USD/m)")
    fig_sc2.update_yaxes(title_text="Metros / Guardia")
    fig_sc2.update_layout(showlegend=False)
    st.plotly_chart(fig_sc2, use_container_width=True)

    # Radar
    cats = ["MP/Guardia", "VP Efectiva", "DM (%)", "Utiliz. (%)", "Costo bajo"]
    fig_rad = go.Figure()
    max_mp = max(mp_vals); max_vp = max(vp_vals)
    for r in rows:
        eq_short = r["equipo"].split("(")[0].strip()
        vals_rad = [
            r["mp_g"] / max_mp * 10,
            r["vp_ef"] / max_vp * 10,
            r["dm_cfg"] / 10,
            r["u_cfg"]  / 10,
            10 - r["cm"] / max(cm_vals) * 10,
        ]
        vals_rad.append(vals_rad[0])
        fig_rad.add_trace(go.Scatterpolar(r=vals_rad, theta=cats + [cats[0]],
                                           fill="toself", name=eq_short,
                                           line=dict(color=r["color"], width=2),
                                           fillcolor=r["color"] + "22"))
    fig_rad.update_layout(**base_layout("Radar Comparativo (escala 0–10)", h=500))
    fig_rad.update_layout(polar=dict(
        bgcolor=CARD,
        radialaxis=dict(visible=True, range=[0,10], gridcolor=GRID, color=TEXT),
        angularaxis=dict(gridcolor=GRID, color="#ffffff"),
    ))
    st.plotly_chart(fig_rad, use_container_width=True)

    # Tabla
    st.markdown('<p class="section-title">📋 Tabla Comparativa</p>', unsafe_allow_html=True)
    df_comp = pd.DataFrame({
        "Equipo":            [r["equipo"] for r in rows],
        "Tipo":              [r["tipo"]   for r in rows],
        "Ø (mm)":            [r["diam"]   for r in rows],
        "VP Ef. (m/h)":      np.round([r["vp_ef"] for r in rows], 2),
        "MP/Guardia (m)":    np.round([r["mp_g"]  for r in rows], 2),
        "Costo/m (USD)":     np.round([r["cm"]    for r in rows], 3),
        "TDC (USD/m)":       np.round([r["tdc"]   for r in rows], 3),
        "C.Directo (USD/m)": np.round([r["c_dir"] for r in rows], 3),
        "Aplicación":        [r["app"]    for r in rows],
    })
    st.dataframe(df_comp.style
                        .highlight_max(axis=0, subset=["MP/Guardia (m)","VP Ef. (m/h)"], color="#1a472a")
                        .highlight_min(axis=0, subset=["Costo/m (USD)","TDC (USD/m)"],   color="#1a472a"),
                 use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════════════
#  TAB 4 — COSTOS
# ═══════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<p class="section-title">💰 Desglose de Costos por Metro Perforado</p>', unsafe_allow_html=True)
    vp_ef = res["vp_ef"]
    comp_labels = ["Equipo", "Mano de Obra", "Mantenimiento", "Consumibles/h", "Broca", "Barra"]
    comp_vals   = [
        round(c_eq   / vp_ef, 4),
        round(c_mo   / vp_ef, 4),
        round(c_mant / vp_ef, 4),
        round(c_cons / vp_ef, 4),
        round(res["c_br_m"],  4),
        round(res["c_ba_m"],  4),
    ]
    comp_colors = [R, G, Y, P, B, "#e67e22"]
    total_cm = sum(comp_vals)

    # Pie donut
    fig_pie = go.Figure(go.Pie(
        labels=comp_labels, values=comp_vals, hole=0.48,
        marker=dict(colors=comp_colors, line=dict(color=DARK, width=2.5)),
        textinfo="label+percent+value",
        texttemplate="%{label}<br>%{percent}<br>$%{value:.4f}",
        textfont=dict(size=11),
    ))
    fig_pie.add_annotation(x=0.5, y=0.5, showarrow=False,
                            text=f"Total<br><b>{total_cm:.4f}</b><br>USD/m",
                            font=dict(size=13, color="#ffffff"))
    fig_pie.update_layout(**base_layout("Estructura de Costos (USD/m)", h=460))

    # Waterfall
    fig_wf = go.Figure(go.Waterfall(
        orientation="v", measure=["relative"]*6 + ["total"],
        x=comp_labels + ["TOTAL"],
        y=comp_vals   + [None],
        connector=dict(line=dict(color=GRID, width=1.5)),
        increasing=dict(marker=dict(color=R)),
        totals=dict(marker=dict(color=G)),
        text=[f"${v:.4f}" for v in comp_vals] + [f"${total_cm:.4f}"],
        textposition="outside",
        textfont=dict(color="#ffffff"),
    ))
    fig_wf.update_layout(**base_layout("Cascada de Costos — USD por Metro Perforado", h=460))
    fig_wf.update_yaxes(title_text="USD/m")

    c1, c2 = st.columns(2)
    with c1: st.plotly_chart(fig_pie, use_container_width=True)
    with c2: st.plotly_chart(fig_wf,  use_container_width=True)

    # Curva costo acumulado vs metros (sensibilidad por producción mensual)
    st.markdown('<p class="section-title">📈 Costo Total vs Metros Perforados / Mes</p>', unsafe_allow_html=True)
    metros_mes = np.arange(100, 5001, 100)
    fig_curva = go.Figure()
    for nombre_eq, c in EQUIPOS.items():
        r2 = calcular(c["vp"],c["dm"]/100,c["u"]/100,t_guard,c["t_pos"],
                      c["c_eq"],c["c_mo"],c["c_mant"],c["c_cons"],
                      c["p_broca"],c["vu_broca"],c["p_barra"],c["vu_barra"],
                      ucs,cai,rqd,c["n_brazos"])
        fig_curva.add_trace(go.Scatter(x=metros_mes, y=metros_mes * r2["cm"],
                                        mode="lines", name=nombre_eq.split("(")[0].strip(),
                                        line=dict(color=c["color"], width=2.5)))
    fig_curva.update_layout(**base_layout("Costo Acumulado de Perforación vs Metros/Mes", h=400))
    fig_curva.update_xaxes(title_text="Metros Perforados / Mes")
    fig_curva.update_yaxes(title_text="Costo Total (USD)")
    st.plotly_chart(fig_curva, use_container_width=True)

    # Tabla detallada
    st.markdown('<p class="section-title">📊 Tabla de Costos Detallada</p>', unsafe_allow_html=True)
    df_cost = pd.DataFrame({
        "Componente":          comp_labels,
        "Costo Horario ($/h)": [c_eq, c_mo, c_mant, c_cons, "—", "—"],
        "Costo/m (USD)":       comp_vals,
        "% del Total":         [round(v/total_cm*100, 1) for v in comp_vals],
    })
    st.dataframe(df_cost.style.bar(subset=["% del Total"], color=R+"88"),
                 use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════════════
#  TAB 5 — VP VS ROCA
# ═══════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<p class="section-title">📉 Velocidad de Penetración vs Resistencia de la Roca</p>', unsafe_allow_html=True)
    st.markdown("""<div class="info-box">
    Modelo Bauer-Calder (1967): VP decrece con la resistencia UCS y el índice CAI.
    La banda sombreada representa el rango ±15% del equipo seleccionado.
    El punto rojo marca la condición actual de trabajo.
    </div>""", unsafe_allow_html=True)

    ucs_x = np.linspace(20, 300, 200)

    fig_vpr = go.Figure()
    for nombre_eq, c in EQUIPOS.items():
        vp_base = c["vp"]
        vp_cur  = vp_base * (80 / ucs_x) ** 0.35
        vp_cur  = np.clip(vp_cur, c["rango_vp"][0], c["rango_vp"][1])
        eq_short= nombre_eq.split("(")[0].strip()
        fig_vpr.add_trace(go.Scatter(x=ucs_x, y=vp_cur, mode="lines",
                                      name=eq_short, line=dict(color=c["color"], width=2.5)))
        if nombre_eq == equipo_sel:
            fig_vpr.add_trace(go.Scatter(
                x=np.concatenate([ucs_x, ucs_x[::-1]]),
                y=np.concatenate([vp_cur*1.15, (vp_cur*0.85)[::-1]]),
                fill="toself", fillcolor=c["color"]+"22",
                line=dict(color="rgba(0,0,0,0)"), showlegend=False, name="Rango ±15%",
            ))
            # Punto actual
            vp_punto = vp_campo * (80 / ucs) ** 0.35
            vp_punto = float(np.clip(vp_punto, c["rango_vp"][0], c["rango_vp"][1]))
            fig_vpr.add_trace(go.Scatter(
                x=[ucs], y=[vp_punto], mode="markers+text",
                name="Condición actual",
                text=["▶ Actual"], textposition="top right",
                textfont=dict(color="#ffffff"),
                marker=dict(color=R, size=14, symbol="star",
                            line=dict(color="#ffffff", width=2)),
            ))

    fig_vpr.update_layout(**base_layout("VP vs UCS — Modelo Bauer-Calder (1967)", h=480))
    fig_vpr.update_xaxes(title_text="UCS — Resistencia a Compresión Uniaxial (MPa)")
    fig_vpr.update_yaxes(title_text="Velocidad de Penetración (m/h)")
    st.plotly_chart(fig_vpr, use_container_width=True)

    # VP vs CAI
    st.markdown('<p class="section-title">🔬 VP vs Índice Cerchar (CAI)</p>', unsafe_allow_html=True)
    cai_x = np.linspace(0.1, 5.0, 100)
    fig_cai = go.Figure()
    for nombre_eq, c in EQUIPOS.items():
        vp_cai = c["vp"] * (1 - (cai_x - 2.0) * 0.05)
        vp_cai = np.clip(vp_cai, c["rango_vp"][0], c["rango_vp"][1])
        fig_cai.add_trace(go.Scatter(x=cai_x, y=vp_cai, mode="lines",
                                      name=nombre_eq.split("(")[0].strip(),
                                      line=dict(color=c["color"], width=2.5)))
    fig_cai.add_vline(x=cai, line=dict(color="#ffffff", dash="dash", width=1.5),
                       annotation=dict(text=f"CAI actual = {cai}", font=dict(color="#ffffff")))
    fig_cai.update_layout(**base_layout("VP vs CAI (Índice Cerchar de Abrasividad)", h=400))
    fig_cai.update_xaxes(title_text="CAI — Índice Cerchar")
    fig_cai.update_yaxes(title_text="Velocidad de Penetración (m/h)")
    st.plotly_chart(fig_cai, use_container_width=True)

    # Superficie 3D VP vs UCS vs CAI
    st.markdown('<p class="section-title">🌐 Superficie 3D — VP en función de UCS y CAI</p>', unsafe_allow_html=True)
    ucs_3d  = np.linspace(20, 250, 40)
    cai_3d  = np.linspace(0.1, 5.0, 40)
    UCS, CAI = np.meshgrid(ucs_3d, cai_3d)
    vp_base_sel = cfg["vp"]
    VP_surf = vp_base_sel * (80/UCS)**0.35 * (1 - (CAI - 2.0)*0.05)
    VP_surf = np.clip(VP_surf, cfg["rango_vp"][0], cfg["rango_vp"][1])
    fig_3d = go.Figure(go.Surface(z=VP_surf, x=UCS, y=CAI, colorscale="RdYlGn",
                                   colorbar=dict(title="VP (m/h)", thickness=12)))
    fig_3d.add_trace(go.Scatter3d(x=[ucs], y=[cai], z=[vp_campo],
                                   mode="markers", name="Punto actual",
                                   marker=dict(color=R, size=6, symbol="circle")))
    fig_3d.update_layout(
        paper_bgcolor=DARK, font=dict(color=TEXT), height=520,
        margin=dict(l=0, r=0, t=40, b=0),
        title=dict(text=f"Superficie VP — {equipo_sel.split('(')[0].strip()}", font=dict(color="#ffffff")),
        scene=dict(
            xaxis=dict(title="UCS (MPa)", gridcolor=GRID, backgroundcolor=DARK),
            yaxis=dict(title="CAI",       gridcolor=GRID, backgroundcolor=DARK),
            zaxis=dict(title="VP (m/h)",  gridcolor=GRID, backgroundcolor=DARK),
        ),
    )
    st.plotly_chart(fig_3d, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════
#  TAB 6 — MEMORIA DE CÁLCULO
# ═══════════════════════════════════════════════════════════════════════
with tab6:
    st.markdown('<p class="section-title">📋 Memoria de Cálculo Completa</p>', unsafe_allow_html=True)
    mc_mc = monte_carlo(
        vp_campo, dm/100, u_eq/100, t_guard,
        c_eq, c_mo, c_mant, c_cons,
        p_broca, vu_broca, p_barra, vu_barra,
        cfg["n_brazos"], n_sim,
        cv_vp/100, cv_dm/100, cv_u/100, cv_cost/100,
        ucs, cai, rqd,
    )
    st.markdown(f"""
## Proyecto: Productividad de Perforación — {equipo_sel}

---
### 1. DATOS DE ENTRADA

| Parámetro | Símbolo | Valor | Unidad |
|---|---|---|---|
| Velocidad de Penetración (campo) | VP | {vp_campo} | m/h |
| Disponibilidad Mecánica | DM | {dm} | % |
| Utilización del Equipo | U | {u_eq} | % |
| Duración de Guardia | TG | {t_guard} | h |
| Tiempo Posicionamiento | Tp | {t_pos} | min/taladro |
| Resistencia Compresión Uniaxial | UCS | {ucs} | MPa |
| Coeficiente Protodyakonov | f | {f_prot} | adim |
| Índice Cerchar | CAI | {cai} | — |
| RQD | RQD | {rqd} | % |
| N° de Brazos | n | {cfg['n_brazos']} | — |
| Diámetro de Perforación | Ø | {cfg['diam_mm']} | mm |

---
### 2. FÓRMULAS Y RESULTADOS DETERMINISTAS

#### 2.1 Velocidad de Penetración Efectiva
```
VPef = VP × U × DM
VPef = {vp_campo} × {u_eq/100} × {dm/100}
VPef = {res['vp_ef']} m/h
```

#### 2.2 Tiempo Efectivo de Perforación
```
Tef = TG × U × DM
Tef = {t_guard} × {u_eq/100} × {dm/100}
Tef = {res['t_ef']} h/guardia
```

#### 2.3 Metros Perforados por Guardia
```
MP = VPef × Tef × N_brazos
MP = {res['vp_ef']} × {res['t_ef']} × {cfg['n_brazos']}
MP = {res['mp_g']} m/guardia
```

#### 2.4 Costo Horario Total
```
Ch = Ceq + Cmo + Cmant + Ccons
Ch = {c_eq} + {c_mo} + {c_mant} + {c_cons}
Ch = {res['ch']} USD/h
```

#### 2.5 Costo por Metro Perforado
```
Cm = Ch / VPef
Cm = {res['ch']} / {res['vp_ef']}
Cm = {res['cm']} USD/m
```

#### 2.6 Costo de Aceros por Metro
```
C_broca = P_broca / VU_broca = {p_broca} / {vu_broca} = {res['c_br_m']} USD/m
C_barra = P_barra / VU_barra = {p_barra} / {vu_barra} = {res['c_ba_m']} USD/m
Ccons/m = {res['c_br_m']} + {res['c_ba_m']} = {res['c_cons_m']} USD/m
```

#### 2.7 TDC — Total Drilling Cost
```
TDC = (P_broca / VU_broca) + (Ch / VPef)
TDC = {res['c_br_m']} + {res['cm']}
TDC = {res['tdc']} USD/m
```

#### 2.8 Costo Directo Total
```
C_directo = Cm + Ccons/m
C_directo = {res['cm']} + {res['c_cons_m']}
C_directo = {res['c_dir']} USD/m
```

---
### 3. RESULTADOS MONTE CARLO ({n_sim:,} iteraciones)

| Variable | P10 | P50 (mediana) | P90 | Media | Desv. Est. |
|---|---|---|---|---|---|
| VP Efectiva (m/h) | {np.percentile(mc_mc['vp_s'],10):.3f} | {np.percentile(mc_mc['vp_s'],50):.3f} | {np.percentile(mc_mc['vp_s'],90):.3f} | {np.mean(mc_mc['vp_s']):.3f} | {np.std(mc_mc['vp_s']):.3f} |
| MP/Guardia (m) | {np.percentile(mc_mc['mp_s'],10):.2f} | {np.percentile(mc_mc['mp_s'],50):.2f} | {np.percentile(mc_mc['mp_s'],90):.2f} | {np.mean(mc_mc['mp_s']):.2f} | {np.std(mc_mc['mp_s']):.2f} |
| Costo/m (USD) | {np.percentile(mc_mc['cm_s'],10):.4f} | {np.percentile(mc_mc['cm_s'],50):.4f} | {np.percentile(mc_mc['cm_s'],90):.4f} | {np.mean(mc_mc['cm_s']):.4f} | {np.std(mc_mc['cm_s']):.4f} |
| TDC (USD/m) | {np.percentile(mc_mc['tdc_s'],10):.4f} | {np.percentile(mc_mc['tdc_s'],50):.4f} | {np.percentile(mc_mc['tdc_s'],90):.4f} | {np.mean(mc_mc['tdc_s']):.4f} | {np.std(mc_mc['tdc_s']):.4f} |

---
### 4. BIBLIOGRAFÍA
- Alfredo Camac Torres — *Perforación y Voladura de Rocas en Minería*
- Llaique & Sánchez (2015) — *Memoria de Cálculo de Productividad de Perforación*
- Bernaola (1985) — *Tecnología de Perforación*
- Bauer & Calder (1967) — *Drilling Performance & Rock Properties*
- Protodyakonov — *Clasificación de Rocas por Dureza*

> Aplicación desarrollada con Python · Streamlit · Plotly · NumPy · SciPy
""")

    # Botón exportar CSV
    df_export = pd.DataFrame({
        "Parámetro": [
            "Equipo","VP Campo (m/h)","DM (%)","U (%)","T.Guardia (h)","UCS (MPa)","CAI","RQD (%)",
            "VP Efectiva (m/h)","T.Efectivo (h)","MP/Guardia (m)","Ch (USD/h)",
            "Cm (USD/m)","TDC (USD/m)","C.Directo (USD/m)",
            "MC P10 MP/g","MC P50 MP/g","MC P90 MP/g",
            "MC P10 Cm","MC P50 Cm","MC P90 Cm",
        ],
        "Valor": [
            equipo_sel, vp_campo, dm, u_eq, t_guard, ucs, cai, rqd,
            res["vp_ef"], res["t_ef"], res["mp_g"], res["ch"],
            res["cm"], res["tdc"], res["c_dir"],
            round(np.percentile(mc_mc["mp_s"],10),2),
            round(np.percentile(mc_mc["mp_s"],50),2),
            round(np.percentile(mc_mc["mp_s"],90),2),
            round(np.percentile(mc_mc["cm_s"],10),4),
            round(np.percentile(mc_mc["cm_s"],50),4),
            round(np.percentile(mc_mc["cm_s"],90),4),
        ],
    })
    st.download_button(
        label="📥 Descargar Memoria de Cálculo (CSV)",
        data=df_export.to_csv(index=False, encoding="utf-8-sig"),
        file_name=f"memoria_calculo_{equipo_sel[:10].replace(' ','_')}.csv",
        mime="text/csv",
        use_container_width=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(f"""
<div style="text-align:center; color:#6b7faa; font-size:0.82rem; padding:0.5rem;">
    ⛏️ Productividad de Perforación Minera &nbsp;|&nbsp;
    Python + Streamlit + Plotly + Monte Carlo &nbsp;|&nbsp;
    Ing. de Minas — UNA Puno
</div>
""", unsafe_allow_html=True)
