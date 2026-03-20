import math
import io
import joblib
import numpy as np
import pandas as pd
import networkx as nx

from dash import Dash, dcc, html, Input, Output, State, callback_context, no_update
import plotly.express as px
import plotly.graph_objects as go

# =========================================================
# PDF KÜTÜPHANESİ (opsiyonel)
# =========================================================
PDF_AVAILABLE = True
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
except Exception:
    PDF_AVAILABLE = False

# =========================================================
# DOSYA YOLLARI
# =========================================================
DATA_PATH = "arctic_advanced_dataset.csv"
RISK_MODEL_PATH = "risk_score_model.joblib"
CLASS_MODEL_PATH = "risk_class_model.joblib"

# =========================================================
# VERİYİ YÜKLE
# =========================================================
df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])

risk_model = joblib.load(RISK_MODEL_PATH)
class_model = joblib.load(CLASS_MODEL_PATH)

all_dates = sorted(df["date"].dt.strftime("%Y-%m-%d").unique())

# =========================================================
# RENKLER / STİL
# =========================================================
COLORS = {
    "bg": "#03101c",
    "panel": "#08203a",
    "panel2": "#0b2745",
    "panel3": "#0a1d31",
    "text": "#eaf4ff",
    "muted": "#9fb7d3",
    "accent": "#7dd3fc",
    "line": "#15446d",
    "ok": "#22c55e",
    "warn": "#eab308",
    "danger": "#ef4444",
    "route1": "#38bdf8",
    "route2": "#a78bfa",
}

CARD_STYLE = {
    "background": "linear-gradient(180deg, #08203a 0%, #0b2745 100%)",
    "border": "1px solid #15446d",
    "borderRadius": "16px",
    "padding": "12px 16px",
    "boxShadow": "0 6px 18px rgba(0,0,0,0.22)",
}

PANEL_STYLE = {
    "background": "linear-gradient(180deg, #08203a 0%, #0b2745 100%)",
    "border": "1px solid #15446d",
    "borderRadius": "18px",
    "boxShadow": "0 6px 18px rgba(0,0,0,0.18)",
}

INPUT_STYLE = {
    "width": "100%",
    "marginBottom": "8px",
    "height": "34px",
    "borderRadius": "8px",
}

BUTTON_STYLE = {
    "width": "100%",
    "padding": "10px",
    "borderRadius": "10px",
    "border": "none",
    "fontWeight": "700",
    "cursor": "pointer",
}

# =========================================================
# YARDIMCI FONKSİYONLAR
# =========================================================
def create_card(title, value, subtitle):
    """Üst istatistik kartı üretir."""
    return html.Div(
        style=CARD_STYLE,
        children=[
            html.Div(title, style={"color": COLORS["muted"], "fontSize": "12px", "marginBottom": "6px"}),
            html.Div(value, style={"fontSize": "20px", "fontWeight": "700", "color": COLORS["text"], "lineHeight": "1.1"}),
            html.Div(subtitle, style={"color": COLORS["accent"], "fontSize": "11px", "marginTop": "6px"}),
        ],
    )


def risk_band(score):
    """Sayısal risk skorunu Düşük / Orta / Yüksek sınıfına çevirir."""
    if score < 35:
        return "Düşük"
    elif score < 65:
        return "Orta"
    return "Yüksek"


def prepare_features(data):
    """
    Model eğitiminde kullanılan feature sütunlarını seçer.
    Buradaki sıra, modelin eğitim sırasıyla aynı olmalıdır.
    """
    feature_cols = [
        "lat", "lon", "ice_concentration", "ice_thickness", "wind_speed",
        "temperature", "visibility", "ice_drift_speed", "ice_drift_direction",
        "wave_height", "current_speed", "current_direction", "ice_type",
        "melting_rate", "accident_history_score", "ship_density"
    ]
    return data[feature_cols].copy()


def apply_scenario(data, wind_multiplier, thinning_multiplier):
    """
    Kullanıcının seçtiği senaryo katsayılarını veriye uygular:
    - wind_speed -> wind_speed_adj
    - ice_thickness -> ice_thickness_adj

    Ardından AI modelinden:
    - ai_risk_score
    - ai_risk_class
    tahmini alınır.
    """
    d = data.copy()

    # Rüzgar katsayısı uygulanıyor
    d["wind_speed_adj"] = d["wind_speed"] * wind_multiplier

    # Buz incelme katsayısı uygulanıyor
    safe_thinning = max(thinning_multiplier, 0.1)
    d["ice_thickness_adj"] = np.clip(d["ice_thickness"] / safe_thinning, 0.05, None)

    # Model girdisi için ayarlanmış kolonları kullan
    features = d.copy()
    features["wind_speed"] = d["wind_speed_adj"]
    features["ice_thickness"] = d["ice_thickness_adj"]

    X = prepare_features(features)

    # Sürekli risk skoru tahmini
    d["ai_risk_score"] = risk_model.predict(X)
    d["ai_risk_score"] = np.clip(d["ai_risk_score"], 0, 100)

    # Sınıf tahmini
    d["ai_risk_class"] = class_model.predict(X)

    return d


def nearest_node(nodes_df, lat, lon):
    """Veri içindeki en yakın grid düğümünü bulur."""
    distances = ((nodes_df["lat"] - lat) ** 2 + (nodes_df["lon"] - lon) ** 2)
    return distances.idxmin()


def geo_distance(lat1, lon1, lat2, lon2):
    """
    Basitleştirilmiş grid-benzeri mesafe.
    Gerçek haversine değil; rota maliyeti için yeterli bir yaklaşım.
    """
    return math.sqrt((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2)


def build_graph(data, route_mode="safe"):
    """
    Rota için graph oluşturur.

    route_mode:
    - safe      : riski daha çok önemser
    - shortest  : mesafeyi daha çok önemser
    - balanced  : risk ve mesafeyi dengeler
    """
    G = nx.Graph()

    # Düğümleri ekle
    for idx, row in data.iterrows():
        G.add_node(
            idx,
            lat=row["lat"],
            lon=row["lon"],
            risk=row["ai_risk_score"]
        )

    idx_list = list(data.index)

    # Komşu düğümler arasında kenar oluştur
    for i in range(len(idx_list)):
        idx_i = idx_list[i]
        row_i = data.loc[idx_i]

        for j in range(i + 1, len(idx_list)):
            idx_j = idx_list[j]
            row_j = data.loc[idx_j]

            lat_diff = abs(row_i["lat"] - row_j["lat"])
            lon_diff = abs(row_i["lon"] - row_j["lon"])

            # Veri setinin grid yapısına göre komşuluk tanımı
            is_neighbor = (
                (lat_diff == 1 and lon_diff == 0) or
                (lat_diff == 0 and lon_diff == 2) or
                (lat_diff == 1 and lon_diff == 2)
            )

            if is_neighbor:
                dist = geo_distance(
                    row_i["lat"], row_i["lon"],
                    row_j["lat"], row_j["lon"]
                )

                avg_risk = (row_i["ai_risk_score"] + row_j["ai_risk_score"]) / 2.0

                # Route mode'a göre kenar maliyeti
                if route_mode == "shortest":
                    edge_weight = dist + (avg_risk / 35.0)
                elif route_mode == "balanced":
                    edge_weight = (dist * 1.0) + (avg_risk / 18.0)
                else:  # safe
                    edge_weight = dist + (avg_risk / 8.0)

                G.add_edge(
                    idx_i,
                    idx_j,
                    weight=edge_weight,
                    risk=avg_risk,
                    dist=dist
                )

    return G


def path_to_df(data, path):
    """Bulunan rota path listesini DataFrame'e çevirir."""
    if not path:
        return pd.DataFrame()
    return data.loc[path].copy()


def route_stats(route_df):
    """Bir rota için toplam risk, ortalama risk, uzunluk ve adım sayısı hesaplar."""
    if route_df.empty:
        return {"total_risk": None, "avg_risk": None, "distance": None, "steps": 0}

    total_risk = float(route_df["ai_risk_score"].sum())
    avg_risk = float(route_df["ai_risk_score"].mean())

    distance = 0.0
    rows = route_df.reset_index(drop=True)
    for i in range(len(rows) - 1):
        distance += geo_distance(
            rows.loc[i, "lat"], rows.loc[i, "lon"],
            rows.loc[i + 1, "lat"], rows.loc[i + 1, "lon"]
        )

    return {
        "total_risk": total_risk,
        "avg_risk": avg_risk,
        "distance": float(distance),
        "steps": int(len(route_df)),
    }


def find_routes(data, start_lat, start_lon, end_lat, end_lon, route_mode="safe"):
    """
    Başlangıç ve hedef arasında:
    - birincil rota
    - alternatif rota
    bulur.
    """
    if data.empty:
        return pd.DataFrame(), pd.DataFrame(), None, None

    start_idx = nearest_node(data, start_lat, start_lon)
    end_idx = nearest_node(data, end_lat, end_lon)

    G = build_graph(data, route_mode=route_mode)

    primary_df = pd.DataFrame()
    alternative_df = pd.DataFrame()

    # Ana rota
    try:
        primary_path = nx.shortest_path(G, source=start_idx, target=end_idx, weight="weight")
        primary_df = path_to_df(data, primary_path)
    except Exception:
        primary_path = None

    # Alternatif rota
    try:
        if primary_path and len(primary_path) > 2:
            G_alt = G.copy()
            internal_nodes = primary_path[1:-1]

            remove_count = max(1, int(len(internal_nodes) * 0.35))

            for node in internal_nodes[:remove_count]:
                if G_alt.has_node(node):
                    G_alt.remove_node(node)

            alt_path = nx.shortest_path(G_alt, source=start_idx, target=end_idx, weight="weight")
            alternative_df = path_to_df(data, alt_path)
    except Exception:
        pass

    return primary_df, alternative_df, start_idx, end_idx


def safe_fmt(value, nd=2):
    """None ise '-' döndürür, değilse biçimlendirir."""
    if value is None:
        return "-"
    return f"{value:.{nd}f}"


def make_pdf_bytes(selected_date, start_lat, start_lon, end_lat, end_lon,
                   wind_multiplier, thinning_multiplier, primary_stats, alt_stats,
                   avg_risk, route_mode):
    """PDF raporunu byte olarak üretir."""
    if not PDF_AVAILABLE:
        return None

    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    y = height - 50
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y, "Arktik AI Risk Haritalama ve Rota Raporu")

    y -= 30
    c.setFont("Helvetica", 11)

    lines = [
        f"Tarih: {selected_date}",
        f"Rota Modu: {route_mode}",
        f"Baslangic Noktasi: ({start_lat}, {start_lon})",
        f"Hedef Noktasi: ({end_lat}, {end_lon})",
        f"Ruzgar Katsayisi: {wind_multiplier}",
        f"Buz Incelmesi Katsayisi: {thinning_multiplier}",
        f"Bolgesel Ortalama AI Riski: {safe_fmt(avg_risk, 1)}",
        "",
        "Birincil Rota:",
        f" - Toplam Rota Riski: {safe_fmt(primary_stats['total_risk'], 1)}",
        f" - Ortalama Rota Riski: {safe_fmt(primary_stats['avg_risk'], 1)}",
        f" - Rota Uzunlugu: {safe_fmt(primary_stats['distance'], 2)}",
        f" - Hucresayisi: {primary_stats['steps']}",
        "",
        "Alternatif Rota:",
        f" - Toplam Rota Riski: {safe_fmt(alt_stats['total_risk'], 1)}",
        f" - Ortalama Rota Riski: {safe_fmt(alt_stats['avg_risk'], 1)}",
        f" - Rota Uzunlugu: {safe_fmt(alt_stats['distance'], 2)}",
        f" - Hucresayisi: {alt_stats['steps']}",
    ]

    for line in lines:
        c.drawString(50, y, line)
        y -= 18
        if y < 60:
            c.showPage()
            y = height - 50
            c.setFont("Helvetica", 11)

    c.save()
    buffer.seek(0)
    return buffer.getvalue()


def make_comment(avg_risk, top_factor, primary_stats, alt_stats, route_mode):
    """Sol paneldeki sistem yorumu alanı için metin üretir."""
    mode_text = {
        "safe": "Sistem güvenliği önceliklendiren rota modunda çalışmaktadır.",
        "shortest": "Sistem en kısa mesafeyi önceliklendiren rota modunda çalışmaktadır.",
        "balanced": "Sistem risk ve mesafeyi dengeleyen rota modunda çalışmaktadır."
    }

    lines = [
        html.P(mode_text.get(route_mode, "Rota modu uygulanmıştır."), style={"margin": "0 0 6px 0"}),
        html.P(f"Seçilen senaryoda AI tabanlı ortalama bölgesel risk {avg_risk:.1f} olarak hesaplanmıştır.", style={"margin": "0 0 6px 0"}),
        html.P(f"En baskın etki eden faktör: {top_factor}.", style={"margin": "0 0 6px 0"}),
    ]

    if primary_stats["total_risk"] is not None:
        lines.append(
            html.P(
                f"Birincil rota {primary_stats['steps']} hücreden oluşmakta, toplam rota riski "
                f"{primary_stats['total_risk']:.1f}, ortalama rota riski {primary_stats['avg_risk']:.1f} "
                f"ve rota uzunluğu {primary_stats['distance']:.2f} grid-birimdir.",
                style={"margin": "0 0 6px 0"}
            )
        )

    if alt_stats["total_risk"] is not None:
        lines.append(
            html.P(
                f"Alternatif rota için toplam risk {alt_stats['total_risk']:.1f}, ortalama risk "
                f"{alt_stats['avg_risk']:.1f} ve uzunluk {alt_stats['distance']:.2f} grid-birimdir.",
                style={"margin": "0"}
            )
        )

    return lines

# =========================================================
# DASH APP
# =========================================================
app = Dash(__name__)
app.title = "Arktik Risk ve Güvenli Rota Sistemi"

# =========================================================
# LAYOUT
# =========================================================
app.layout = html.Div(
    style={
        "background": "radial-gradient(circle at top, #0a2740 0%, #04101d 55%, #020a12 100%)",
        "minHeight": "100vh",
        "padding": "12px",
        "fontFamily": "Arial, sans-serif",
        "color": COLORS["text"],
    },
    children=[
        dcc.Store(id="point-store", data={"start_lat": 68, "start_lon": -68, "end_lat": 79, "end_lon": -24}),
        dcc.Store(id="route-stats-store"),
        dcc.Download(id="download-pdf"),

        # Üst başlık
        html.Div(
            style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginBottom": "10px"},
            children=[
                html.Div([
                    html.H1("Arktik AI Risk Haritalama ve Güvenli Rota Sistemi", style={"margin": "0", "fontSize": "22px"}),
                    html.Div(
                        "AI tahmini, risk yüzdeleri, tehlike analizi ve rota optimizasyonu",
                        style={"color": COLORS["muted"], "marginTop": "4px", "fontSize": "12px"},
                    ),
                ]),
                html.Div(
                    "AI + Rota + Analiz",
                    style={
                        "padding": "8px 12px",
                        "borderRadius": "12px",
                        "background": "#0a1d31",
                        "border": "1px solid #15446d",
                        "color": COLORS["accent"],
                        "fontWeight": "700",
                        "fontSize": "12px",
                    },
                ),
            ],
        ),

        # Üst kartlar
        html.Div(
            id="top-cards",
            style={
                "display": "grid",
                "gridTemplateColumns": "repeat(4, 1fr)",
                "gap": "10px",
                "marginBottom": "12px",
            },
        ),

        # Ana 3 sütun
        html.Div(
            style={
                "display": "grid",
                "gridTemplateColumns": "360px minmax(980px, 1fr) 420px",
                "gap": "12px",
                "alignItems": "start",
            },
            children=[
                # =====================================================
                # SOL PANEL
                # =====================================================
                html.Div(
                    style={**PANEL_STYLE, "padding": "14px"},
                    children=[
                        html.H3("Kontrol Paneli", style={"marginTop": "0", "marginBottom": "10px", "fontSize": "18px"}),

                        html.Label("Tarih", style={"color": COLORS["muted"], "fontSize": "12px"}),
                        dcc.Dropdown(
                            id="date-dropdown",
                            options=[{"label": d, "value": d} for d in all_dates],
                            value=all_dates[0],
                            clearable=False,
                            style={"color": "black", "marginBottom": "10px"},
                        ),

                        html.Label("Risk Görüntüleme Eşiği", style={"color": COLORS["muted"], "fontSize": "12px"}),
                        dcc.Slider(
                            id="risk-threshold",
                            min=0, max=100, step=5, value=35,
                            marks={0: "0", 25: "25", 50: "50", 75: "75", 100: "100"},
                        ),

                        html.Br(),
                        html.Label("Rüzgâr Senaryo Katsayısı", style={"color": COLORS["muted"], "fontSize": "12px"}),
                        dcc.Slider(
                            id="wind-multiplier",
                            min=0.5, max=2.0, step=0.1, value=1.0,
                            marks={0.5: "0.5", 1.0: "1.0", 1.5: "1.5", 2.0: "2.0"},
                        ),

                        html.Br(),
                        html.Label("Buz İncelmesi Senaryo Katsayısı", style={"color": COLORS["muted"], "fontSize": "12px"}),
                        dcc.Slider(
                            id="thinning-multiplier",
                            min=0.5, max=2.0, step=0.1, value=1.0,
                            marks={0.5: "0.5", 1.0: "1.0", 1.5: "1.5", 2.0: "2.0"},
                        ),

                        html.Hr(style={"borderColor": COLORS["line"], "margin": "12px 0"}),

                        html.H4("Harita Tıklama Modu", style={"marginBottom": "8px", "fontSize": "15px"}),
                        dcc.RadioItems(
                            id="click-mode",
                            options=[
                                {"label": " Başlangıç seç", "value": "start"},
                                {"label": " Hedef seç", "value": "end"},
                            ],
                            value="start",
                            labelStyle={"display": "block", "marginBottom": "6px", "fontSize": "13px"},
                            inputStyle={"marginRight": "8px"},
                        ),

                        html.Button("Noktaları Sıfırla", id="reset-points-btn", n_clicks=0,
                                    style={**BUTTON_STYLE, "marginTop": "8px", "marginBottom": "10px"}),

                        html.H4("Rota Modu", style={"marginBottom": "8px", "fontSize": "15px"}),
                        dcc.RadioItems(
                            id="route-mode",
                            options=[
                                {"label": " En Güvenli (Safe)", "value": "safe"},
                                {"label": " En Kısa (Shortest)", "value": "shortest"},
                                {"label": " Dengeli (Balanced)", "value": "balanced"},
                            ],
                            value="safe",
                            labelStyle={"display": "block", "marginBottom": "6px", "fontSize": "13px"},
                            inputStyle={"marginRight": "8px"},
                        ),
                        html.Div(
                            "Safe: riski azaltır, Shortest: mesafeyi azaltır, Balanced: ikisini dengeler.",
                            style={"fontSize": "11px", "color": COLORS["muted"], "marginBottom": "10px"}
                        ),

                        html.H4("Rota Girişi", style={"marginBottom": "8px", "fontSize": "15px"}),

                        html.Label("Başlangıç Enlem", style={"fontSize": "12px"}),
                        dcc.Input(id="start-lat", type="number", value=68, style=INPUT_STYLE),

                        html.Label("Başlangıç Boylam", style={"fontSize": "12px"}),
                        dcc.Input(id="start-lon", type="number", value=-68, style=INPUT_STYLE),

                        html.Label("Hedef Enlem", style={"fontSize": "12px"}),
                        dcc.Input(id="end-lat", type="number", value=79, style=INPUT_STYLE),

                        html.Label("Hedef Boylam", style={"fontSize": "12px"}),
                        dcc.Input(id="end-lon", type="number", value=-24, style=INPUT_STYLE),

                        html.Button("Rota Analizi Yap", id="route-btn", n_clicks=0,
                                    style={**BUTTON_STYLE, "marginBottom": "8px"}),
                        html.Button("PDF Rapor İndir", id="download-pdf-btn", n_clicks=0,
                                    style={**BUTTON_STYLE, "marginBottom": "10px"}),

                        html.H4("Animasyon", style={"marginBottom": "8px", "fontSize": "15px"}),
                        html.Div(
                            style={"display": "flex", "gap": "8px", "marginBottom": "10px"},
                            children=[
                                html.Button("▶ Oynat", id="play-btn", n_clicks=0,
                                            style={"flex": "1", "padding": "10px", "borderRadius": "10px", "border": "none"}),
                                html.Button("■ Durdur", id="stop-btn", n_clicks=0,
                                            style={"flex": "1", "padding": "10px", "borderRadius": "10px", "border": "none"}),
                            ],
                        ),
                        dcc.Interval(id="interval", interval=1400, n_intervals=0, disabled=True),

                        html.Hr(style={"borderColor": COLORS["line"], "margin": "12px 0"}),

                        html.H4("AI ve Rota Yorumu", style={"marginBottom": "8px", "fontSize": "15px"}),
                        html.Div(id="system-comment", style={"color": COLORS["muted"], "lineHeight": "1.5", "fontSize": "12px"}),
                    ],
                ),

                # =====================================================
                # ORTA HARİTA PANELİ
                # =====================================================
                html.Div(
                    style={**PANEL_STYLE, "padding": "8px", "height": "860px"},
                    children=[dcc.Graph(id="arctic-map", style={"height": "840px"})],
                ),

                # =====================================================
                # SAĞ PANEL
                # =====================================================
                html.Div(
                    style={
                        "display": "grid",
                        "gridTemplateRows": "220px 220px 220px 320px 320px 300px",
                        "gap": "12px",
                    },
                    children=[
                        html.Div(style={**PANEL_STYLE, "padding": "8px"},
                                 children=[dcc.Graph(id="risk-band-chart", style={"height": "200px"})]),

                        html.Div(style={**PANEL_STYLE, "padding": "8px"},
                                 children=[dcc.Graph(id="risk-percent-chart", style={"height": "200px"})]),

                        html.Div(style={**PANEL_STYLE, "padding": "8px"},
                                 children=[dcc.Graph(id="time-trend-chart", style={"height": "200px"})]),

                        html.Div(
                            style={**PANEL_STYLE, "padding": "8px", "overflowY": "auto"},
                            children=[
                                dcc.Graph(id="factor-chart", style={"height": "240px"}),
                                html.Div(id="danger-summary", style={"fontSize": "12px", "marginTop": "4px"})
                            ],
                        ),

                        html.Div(
                            style={**PANEL_STYLE, "padding": "10px", "overflowY": "auto"},
                            children=[html.Div(id="route-summary", style={"fontSize": "12px"})],
                        ),

                        html.Div(
                            style={**PANEL_STYLE, "padding": "12px", "overflowY": "auto"},
                            children=[
                                html.H4("Kullanılan Veriler", style={"marginTop": "0", "marginBottom": "8px", "fontSize": "15px"}),
                                html.Ul(
                                    style={"paddingLeft": "18px", "margin": "0", "fontSize": "12px", "lineHeight": "1.6"},
                                    children=[
                                        html.Li("lat / lon → konumsal koordinatlar"),
                                        html.Li("ice_concentration → buz yoğunluğu"),
                                        html.Li("ice_thickness → buz kalınlığı"),
                                        html.Li("wind_speed → rüzgâr hızı"),
                                        html.Li("temperature → sıcaklık"),
                                        html.Li("visibility → görüş mesafesi"),
                                        html.Li("ice_drift_speed → buz hareket hızı"),
                                        html.Li("ice_drift_direction → buz hareket yönü"),
                                        html.Li("wave_height → dalga yüksekliği"),
                                        html.Li("current_speed → akıntı hızı"),
                                        html.Li("current_direction → akıntı yönü"),
                                        html.Li("ice_type → buz tipi"),
                                        html.Li("melting_rate → erime oranı"),
                                        html.Li("accident_history_score → geçmiş kaza yoğunluğu"),
                                        html.Li("ship_density → gemi yoğunluğu"),
                                    ],
                                ),
                                html.Hr(style={"borderColor": COLORS["line"], "margin": "10px 0"}),
                                html.Div(
                                    "AI risk skoru ve risk sınıfı tahmini bu değişkenler kullanılarak oluşturulur. "
                                    "Rota optimizasyonu ise bu AI risk skoru ile mesafe bilgisini birlikte kullanır.",
                                    style={"fontSize": "12px", "color": COLORS["muted"], "lineHeight": "1.5"}
                                ),
                            ]
                        ),
                    ],
                ),
            ],
        ),
    ],
)

# =========================================================
# HARİTA TIKLAMA
# =========================================================
@app.callback(
    Output("point-store", "data"),
    Input("arctic-map", "clickData"),
    Input("reset-points-btn", "n_clicks"),
    State("click-mode", "value"),
    State("point-store", "data"),
    prevent_initial_call=True,
)
def update_points(click_data, reset_clicks, click_mode, store):
    """
    Haritaya tıklanınca başlangıç veya hedef noktası güncellenir.
    Reset butonu varsayılan değerlere döner.
    """
    ctx = callback_context
    if not ctx.triggered:
        return store

    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if trigger_id == "reset-points-btn":
        return {"start_lat": 68, "start_lon": -68, "end_lat": 79, "end_lon": -24}

    if trigger_id == "arctic-map" and click_data:
        try:
            lat = click_data["points"][0]["lat"]
            lon = click_data["points"][0]["lon"]

            new_store = dict(store) if store else {}
            if click_mode == "start":
                new_store["start_lat"] = lat
                new_store["start_lon"] = lon
            else:
                new_store["end_lat"] = lat
                new_store["end_lon"] = lon
            return new_store
        except Exception:
            return store

    return store


@app.callback(
    Output("start-lat", "value"),
    Output("start-lon", "value"),
    Output("end-lat", "value"),
    Output("end-lon", "value"),
    Input("point-store", "data"),
)
def sync_inputs_with_store(store):
    """Store içindeki koordinatları inputlara yazar."""
    return (
        store.get("start_lat", 68),
        store.get("start_lon", -68),
        store.get("end_lat", 79),
        store.get("end_lon", -24),
    )

# =========================================================
# ANİMASYON
# =========================================================
@app.callback(
    Output("interval", "disabled"),
    Input("play-btn", "n_clicks"),
    Input("stop-btn", "n_clicks"),
    prevent_initial_call=True,
)
def toggle_animation(play_clicks, stop_clicks):
    """Oynat / Durdur butonları interval'i açıp kapatır."""
    ctx = callback_context
    if not ctx.triggered:
        return True
    btn = ctx.triggered[0]["prop_id"].split(".")[0]
    return btn != "play-btn"


@app.callback(
    Output("date-dropdown", "value"),
    Input("interval", "n_intervals"),
    State("date-dropdown", "value"),
)
def animate_date(n, current_date):
    """Tarih dropdown'unu sıradaki güne taşıyarak animasyon etkisi verir."""
    if current_date not in all_dates:
        return all_dates[0]
    if n == 0:
        return current_date
    idx = all_dates.index(current_date)
    return all_dates[(idx + 1) % len(all_dates)]

# =========================================================
# ANA DASHBOARD CALLBACK
# =========================================================
@app.callback(
    Output("top-cards", "children"),
    Output("arctic-map", "figure"),
    Output("risk-band-chart", "figure"),
    Output("risk-percent-chart", "figure"),
    Output("time-trend-chart", "figure"),
    Output("factor-chart", "figure"),
    Output("system-comment", "children"),
    Output("danger-summary", "children"),
    Output("route-summary", "children"),
    Output("route-stats-store", "data"),
    Input("date-dropdown", "value"),
    Input("risk-threshold", "value"),
    Input("wind-multiplier", "value"),
    Input("thinning-multiplier", "value"),
    Input("route-mode", "value"),
    Input("route-btn", "n_clicks"),
    State("start-lat", "value"),
    State("start-lon", "value"),
    State("end-lat", "value"),
    State("end-lon", "value"),
)
def update_dashboard(selected_date, threshold, wind_multiplier, thinning_multiplier,
                     route_mode, route_clicks, start_lat, start_lon, end_lat, end_lon):
    """
    Tüm dashboard bileşenlerini günceller:
    - seçilen tarih
    - senaryo katsayıları
    - rota modu
    - başlangıç / hedef
    """
    try:
        # -----------------------------------------------------
        # Seçilen tarihteki veri
        # -----------------------------------------------------
        current = df[df["date"].dt.strftime("%Y-%m-%d") == selected_date].copy()
        current = apply_scenario(current, wind_multiplier, thinning_multiplier)
        current["risk_band"] = current["ai_risk_score"].apply(risk_band)

        # -----------------------------------------------------
        # Genel risk istatistikleri
        # -----------------------------------------------------
        total_cells = len(current)
        low_count = int((current["ai_risk_score"] < 35).sum())
        mid_count = int(((current["ai_risk_score"] >= 35) & (current["ai_risk_score"] < 65)).sum())
        high_count = int((current["ai_risk_score"] >= 65).sum())

        low_pct = (low_count / total_cells) * 100 if total_cells else 0
        mid_pct = (mid_count / total_cells) * 100 if total_cells else 0
        high_pct = (high_count / total_cells) * 100 if total_cells else 0

        avg_risk = float(current["ai_risk_score"].mean()) if total_cells else 0.0

        # -----------------------------------------------------
        # Rota hesabı
        # -----------------------------------------------------
        primary_route, alternative_route, start_idx, end_idx = find_routes(
            current, start_lat, start_lon, end_lat, end_lon, route_mode=route_mode
        )

        primary_stats = route_stats(primary_route)
        alt_stats = route_stats(alternative_route)

        best_route_text = "-"
        if primary_stats["total_risk"] is not None and alt_stats["total_risk"] is not None:
            best_route_text = "Birincil" if primary_stats["total_risk"] <= alt_stats["total_risk"] else "Alternatif"
        elif primary_stats["total_risk"] is not None:
            best_route_text = "Birincil"

        # -----------------------------------------------------
        # Üst kartlar
        # -----------------------------------------------------
        cards = [
            create_card("AI Ortalama Risk", f"{avg_risk:.1f}", "Bölgesel risk seviyesi"),
            create_card("Yüksek Riskli Hücre", f"{high_count}", "Kritik hücre sayısı"),
            create_card("Yüksek Risk Yüzdesi", f"%{high_pct:.1f}", "Toplam bölge içindeki oran"),
            create_card("Önerilen Ana Seçenek", best_route_text, "Risk karşılaştırmasına göre"),
        ]

        # -----------------------------------------------------
        # Haritada gösterilecek veri
        # -----------------------------------------------------
        display_df = current[current["ai_risk_score"] >= threshold].copy()
        if display_df.empty:
            display_df = current.copy()

        # -----------------------------------------------------
        # Harita
        # -----------------------------------------------------
        fig_map = px.scatter_geo(
            display_df,
            lat="lat",
            lon="lon",
            color="ai_risk_score",
            size="ai_risk_score",
            hover_data={
                "lat": True,
                "lon": True,
                "ai_risk_score": ":.1f",
                "ai_risk_class": True,
                "ice_concentration": ":.1f",
                "ice_thickness_adj": ":.2f",
                "wind_speed_adj": ":.1f",
                "wave_height": ":.2f",
                "melting_rate": ":.3f",
            },
            projection="stereographic",
            title=f"Arktik AI Risk Haritası — {selected_date}",
            color_continuous_scale=[
                [0.0, "#22c55e"],
                [0.35, "#84cc16"],
                [0.55, "#eab308"],
                [0.75, "#f97316"],
                [1.0, "#ef4444"],
            ],
        )

        fig_map.update_traces(
            marker=dict(
                sizemode="diameter",
                sizeref=2.2,
                opacity=0.82,
                line=dict(width=0.4, color="rgba(255,255,255,0.25)")
            )
        )

        fig_map.update_layout(
            paper_bgcolor="#081a2d",
            plot_bgcolor="#081a2d",
            font_color="#eaf4ff",
            margin=dict(l=0, r=0, t=40, b=0),
            height=820,
            coloraxis_colorbar=dict(
                title="Risk",
                thickness=12,
                len=0.72,
                x=0.96,
                y=0.52
            ),
            legend=dict(
                x=0.01,
                y=0.99,
                bgcolor="rgba(7,25,44,0.78)",
                bordercolor="#1c527f",
                borderwidth=1,
                font=dict(size=10)
            ),
            geo=dict(
                projection_type="stereographic",
                projection_rotation=dict(lat=90, lon=0, roll=0),
                lataxis=dict(range=[60, 90], showgrid=True, gridcolor="#c9d6e2"),
                lonaxis=dict(showgrid=True, gridcolor="#c9d6e2"),
                showocean=True,
                oceancolor="#dce9f5",
                showland=True,
                landcolor="#edf2f7",
                showcoastlines=True,
                coastlinecolor="#97a9bb",
                bgcolor="#081a2d",
            )
        )

        # Birincil rota
        if not primary_route.empty:
            fig_map.add_trace(
                go.Scattergeo(
                    lat=primary_route["lat"],
                    lon=primary_route["lon"],
                    mode="lines+markers",
                    line=dict(width=4, color=COLORS["route1"]),
                    marker=dict(size=5, color=COLORS["route1"]),
                    name="Birincil Güvenli Rota",
                )
            )

        # Alternatif rota
        if not alternative_route.empty:
            fig_map.add_trace(
                go.Scattergeo(
                    lat=alternative_route["lat"],
                    lon=alternative_route["lon"],
                    mode="lines+markers",
                    line=dict(width=3, color=COLORS["route2"], dash="dash"),
                    marker=dict(size=4, color=COLORS["route2"]),
                    name="Alternatif Güvenli Rota",
                )
            )

        # Başlangıç
        fig_map.add_trace(
            go.Scattergeo(
                lat=[start_lat],
                lon=[start_lon],
                mode="markers",
                marker=dict(size=10, color="white", symbol="circle"),
                name="Başlangıç",
            )
        )

        # Hedef
        fig_map.add_trace(
            go.Scattergeo(
                lat=[end_lat],
                lon=[end_lon],
                mode="markers",
                marker=dict(size=10, color="cyan", symbol="star"),
                name="Hedef",
            )
        )

        # -----------------------------------------------------
        # Risk sınıfı grafiği
        # -----------------------------------------------------
        band_counts = current["risk_band"].value_counts().reindex(["Düşük", "Orta", "Yüksek"], fill_value=0)

        fig_band = go.Figure()
        fig_band.add_trace(
            go.Bar(
                x=band_counts.index.tolist(),
                y=band_counts.values.tolist(),
                text=band_counts.values.tolist(),
                textposition="outside",
            )
        )
        fig_band.update_layout(
            title="Risk Sınıfı Dağılımı",
            paper_bgcolor="#08203a",
            plot_bgcolor="#08203a",
            font_color="#eaf4ff",
            margin=dict(l=12, r=12, t=34, b=20),
            title_font_size=13,
            xaxis_title=None,
            yaxis_title=None,
        )

        # -----------------------------------------------------
        # Risk yüzdeleri grafiği
        # -----------------------------------------------------
        fig_percent = go.Figure(
            data=[
                go.Pie(
                    labels=["Düşük", "Orta", "Yüksek"],
                    values=[low_pct, mid_pct, high_pct],
                    hole=0.55,
                    textinfo="label+percent",
                )
            ]
        )
        fig_percent.update_layout(
            title="Risk Yüzdeleri",
            paper_bgcolor="#08203a",
            plot_bgcolor="#08203a",
            font_color="#eaf4ff",
            margin=dict(l=12, r=12, t=34, b=20),
            title_font_size=13,
            showlegend=False,
        )

        # -----------------------------------------------------
        # Zaman trend grafiği
        # -----------------------------------------------------
        trend = df.copy()
        trend = apply_scenario(trend, wind_multiplier, thinning_multiplier)
        trend_daily = trend.groupby(trend["date"].dt.strftime("%Y-%m-%d"))["ai_risk_score"].mean().reset_index()
        trend_daily.columns = ["date", "ai_risk_score"]

        fig_trend = px.line(trend_daily, x="date", y="ai_risk_score", markers=True, title="Zamana Göre Ortalama AI Risk")
        fig_trend.update_layout(
            paper_bgcolor="#08203a",
            plot_bgcolor="#08203a",
            font_color="#eaf4ff",
            margin=dict(l=12, r=12, t=34, b=20),
            title_font_size=13,
            xaxis_title=None,
            yaxis_title=None,
        )

        # -----------------------------------------------------
        # Risk faktörleri
        # -----------------------------------------------------
        factor_values = {
            "Buz Yoğunluğu": ((100 - current["ice_concentration"]) * 0.18).mean(),
            "Buz Kalınlığı": (np.maximum(0, 2.8 - current["ice_thickness_adj"]) * 11).mean(),
            "Rüzgâr": (current["wind_speed_adj"] * 0.9).mean(),
            "Sıcaklık": (np.maximum(0, current["temperature"] + 25) * 1.7).mean(),
            "Görüş": ((100 - current["visibility"]) * 0.25).mean(),
            "Buz Hareketi": (current["ice_drift_speed"] * 16).mean(),
            "Dalga": (current["wave_height"] * 7).mean(),
            "Akıntı": (current["current_speed"] * 10).mean(),
            "Erime Hızı": (current["melting_rate"] * 42).mean(),
            "Geçmiş Kaza": (current["accident_history_score"] * 0.35).mean(),
        }
        factor_df = pd.DataFrame({"Faktör": list(factor_values.keys()), "Katkı": list(factor_values.values())}).sort_values("Katkı", ascending=True)

        fig_factor = px.bar(factor_df, x="Katkı", y="Faktör", orientation="h", title="Riski Etkileyen Faktörler")
        fig_factor.update_layout(
            paper_bgcolor="#08203a",
            plot_bgcolor="#08203a",
            font_color="#eaf4ff",
            margin=dict(l=12, r=12, t=34, b=20),
            title_font_size=13,
            xaxis_title=None,
            yaxis_title=None,
        )

        sorted_factors = sorted(factor_values.items(), key=lambda x: x[1], reverse=True)
        top_3 = sorted_factors[:3]

        factor_explanations = {
            "Buz Yoğunluğu": "Buz yoğunluğunun azalması, açık su ve kırılgan yüzey olasılığını artırarak geçiş güvenliğini azaltır.",
            "Buz Kalınlığı": "Buz kalınlığının düşmesi, taşıma kapasitesini azaltır ve yüzey çökmesi riskini yükseltir.",
            "Rüzgâr": "Yüksek rüzgâr, görüşü düşürür, yüzey dengesini bozar ve buz hareketini artırabilir.",
            "Sıcaklık": "Sıcaklığın yükselmesi, erimeyi hızlandırarak buz yapısının zayıflamasına neden olur.",
            "Görüş": "Düşük görüş, çatlak ve riskli bölgelerin fark edilmesini zorlaştırır.",
            "Buz Hareketi": "Buz hareketinin artması, kısa sürede rota koşullarının değişmesine yol açar.",
            "Dalga": "Dalga yüksekliği, buz parçalanmasını ve kenar bölgelerdeki kararsızlığı artırabilir.",
            "Akıntı": "Akıntılar buz kütlelerinin yer değiştirmesine ve rota güvenliğinin azalmasına neden olabilir.",
            "Erime Hızı": "Yüksek erime oranı, buzun kısa sürede zayıfladığını ve gelecekte riskin artabileceğini gösterir.",
            "Geçmiş Kaza": "Geçmişte yoğun olay görülen alanlar, operasyonel olarak daha dikkatli değerlendirilmelidir.",
        }

        # -----------------------------------------------------
        # Tehlike analizi metni
        # -----------------------------------------------------
        danger_lines = []
        danger_lines.append(html.H4("Tehlike Analizi", style={"marginTop": "0", "marginBottom": "8px", "fontSize": "14px"}))
        danger_lines.append(
            html.P(
                f"Mevcut senaryoda yüksek riskli hücre oranı %{high_pct:.1f} olarak hesaplanmıştır.",
                style={"margin": "0 0 6px 0"}
            )
        )

        if high_pct >= 60:
            danger_lines.append(
                html.P(
                    "Bu durumda bölgenin büyük kısmı tehlikeli kabul edilmektedir ve saha hareketliliği ciddi şekilde sınırlandırılmalıdır.",
                    style={"margin": "0 0 6px 0", "color": "#fca5a5"}
                )
            )
        elif high_pct >= 30:
            danger_lines.append(
                html.P(
                    "Bu durumda orta-yüksek seviyede operasyonel risk bulunmaktadır; rota planlaması dikkatli yapılmalıdır.",
                    style={"margin": "0 0 6px 0", "color": "#fde68a"}
                )
            )
        else:
            danger_lines.append(
                html.P(
                    "Bu durumda genel bölgesel risk daha kontrollü görünmektedir; ancak lokal yüksek risk alanları yine de dikkate alınmalıdır.",
                    style={"margin": "0 0 6px 0", "color": "#86efac"}
                )
            )

        danger_lines.append(
            html.P(
                f"En baskın risk faktörü: {top_3[0][0]}",
                style={"margin": "0 0 6px 0", "fontWeight": "700"}
            )
        )

        for factor_name, factor_value in top_3:
            danger_lines.append(
                html.P(
                    f"• {factor_name}: {factor_explanations.get(factor_name, 'Bu faktör mevcut risk düzeyini artırmaktadır.')}",
                    style={"margin": "0 0 5px 0"}
                )
            )

        # -----------------------------------------------------
        # Sistem yorumu
        # -----------------------------------------------------
        top_factor = factor_df.iloc[-1]["Faktör"]
        comment = make_comment(avg_risk, top_factor, primary_stats, alt_stats, route_mode)

        # -----------------------------------------------------
        # Rota özeti
        # -----------------------------------------------------
        route_summary = [
            html.H4("Rota Özeti", style={"marginTop": "0", "marginBottom": "8px", "fontSize": "15px"}),
            html.P(f"Rota Modu: {route_mode}", style={"margin": "0 0 6px 0", "color": COLORS["accent"]}),
            html.P(f"Başlangıç: ({start_lat}, {start_lon})", style={"margin": "0 0 4px 0"}),
            html.P(f"Hedef: ({end_lat}, {end_lon})", style={"margin": "0 0 8px 0"}),
            html.Hr(style={"borderColor": COLORS["line"]}),
            html.H5("Birincil Güvenli Rota", style={"marginBottom": "6px", "fontSize": "13px"}),
            html.P(f"Toplam Rota Riski: {safe_fmt(primary_stats['total_risk'], 1)}", style={"margin": "0 0 4px 0"}),
            html.P(f"Ortalama Rota Riski: {safe_fmt(primary_stats['avg_risk'], 1)}", style={"margin": "0 0 4px 0"}),
            html.P(f"Rota Uzunluğu: {safe_fmt(primary_stats['distance'], 2)}", style={"margin": "0 0 4px 0"}),
            html.P(f"Hücre Sayısı: {primary_stats['steps']}", style={"margin": "0 0 8px 0"}),
            html.Hr(style={"borderColor": COLORS["line"]}),
            html.H5("Alternatif Güvenli Rota", style={"marginBottom": "6px", "fontSize": "13px"}),
            html.P(f"Toplam Rota Riski: {safe_fmt(alt_stats['total_risk'], 1)}", style={"margin": "0 0 4px 0"}),
            html.P(f"Ortalama Rota Riski: {safe_fmt(alt_stats['avg_risk'], 1)}", style={"margin": "0 0 4px 0"}),
            html.P(f"Rota Uzunluğu: {safe_fmt(alt_stats['distance'], 2)}", style={"margin": "0 0 4px 0"}),
            html.P(f"Hücre Sayısı: {alt_stats['steps']}", style={"margin": "0 0 4px 0"}),
        ]

        route_risk_text = "-"
        route_high_pct = None
        route_mid_pct = None
        route_low_pct = None

        if not primary_route.empty:
            route_total = len(primary_route)
            route_low = int((primary_route["ai_risk_score"] < 35).sum())
            route_mid = int(((primary_route["ai_risk_score"] >= 35) & (primary_route["ai_risk_score"] < 65)).sum())
            route_high = int((primary_route["ai_risk_score"] >= 65).sum())

            route_low_pct = (route_low / route_total) * 100
            route_mid_pct = (route_mid / route_total) * 100
            route_high_pct = (route_high / route_total) * 100

            if route_high_pct >= 50:
                route_risk_text = "Birincil rota üzerinde yüksek riskli hücre oranı çok yüksektir."
            elif route_high_pct >= 20:
                route_risk_text = "Birincil rota üzerinde dikkat gerektiren yüksek riskli bölümler bulunmaktadır."
            else:
                route_risk_text = "Birincil rota görece daha güvenli bölgeler üzerinden geçmektedir."

        if route_high_pct is not None:
            route_summary += [
                html.Hr(style={"borderColor": COLORS["line"]}),
                html.H5("Rota Risk Yüzdeleri", style={"marginBottom": "6px", "fontSize": "13px"}),
                html.P(f"Düşük Riskli Hücre Oranı: %{route_low_pct:.1f}", style={"margin": "0 0 4px 0"}),
                html.P(f"Orta Riskli Hücre Oranı: %{route_mid_pct:.1f}", style={"margin": "0 0 4px 0"}),
                html.P(f"Yüksek Riskli Hücre Oranı: %{route_high_pct:.1f}", style={"margin": "0 0 4px 0"}),
                html.P(route_risk_text, style={"margin": "6px 0 0 0", "color": COLORS["accent"]}),
            ]

        if primary_stats["total_risk"] is not None and alt_stats["total_risk"] is not None:
            better = "Birincil rota" if primary_stats["total_risk"] <= alt_stats["total_risk"] else "Alternatif rota"
            route_summary += [
                html.Hr(style={"borderColor": COLORS["line"]}),
                html.P(f"Daha düşük toplam riskli seçenek: {better}",
                       style={"color": COLORS["accent"], "fontWeight": "700", "margin": "0"})
            ]

        # -----------------------------------------------------
        # PDF için store
        # -----------------------------------------------------
        stats_store = {
            "selected_date": selected_date,
            "start_lat": start_lat,
            "start_lon": start_lon,
            "end_lat": end_lat,
            "end_lon": end_lon,
            "wind_multiplier": wind_multiplier,
            "thinning_multiplier": thinning_multiplier,
            "route_mode": route_mode,
            "avg_risk": avg_risk,
            "primary_stats": primary_stats,
            "alt_stats": alt_stats,
        }

        return (
            cards,
            fig_map,
            fig_band,
            fig_percent,
            fig_trend,
            fig_factor,
            comment,
            danger_lines,
            route_summary,
            stats_store
        )

    except Exception as e:
        error_fig = go.Figure()
        error_fig.update_layout(
            paper_bgcolor="#081a2d",
            plot_bgcolor="#081a2d",
            font_color="white",
            title=f"Hata: {str(e)}",
        )
        fallback_cards = [
            create_card("Hata", "Veri İşlenemedi", "Kod veya model kontrol edilmeli"),
            create_card("Durum", "-", "-"),
            create_card("Durum", "-", "-"),
            create_card("Durum", "-", "-"),
        ]
        return (
            fallback_cards,
            error_fig,
            error_fig,
            error_fig,
            error_fig,
            error_fig,
            [html.P(str(e))],
            [html.P("Tehlike analizi oluşturulamadı.")],
            [html.P("Rota özeti oluşturulamadı.")],
            {}
        )

# =========================================================
# PDF CALLBACK
# =========================================================
@app.callback(
    Output("download-pdf", "data"),
    Input("download-pdf-btn", "n_clicks"),
    State("route-stats-store", "data"),
    prevent_initial_call=True,
)
def download_pdf(n_clicks, stats):
    """PDF indir butonu callback'i."""
    if not stats or not PDF_AVAILABLE:
        return no_update

    pdf_bytes = make_pdf_bytes(
        selected_date=stats.get("selected_date"),
        start_lat=stats.get("start_lat"),
        start_lon=stats.get("start_lon"),
        end_lat=stats.get("end_lat"),
        end_lon=stats.get("end_lon"),
        wind_multiplier=stats.get("wind_multiplier"),
        thinning_multiplier=stats.get("thinning_multiplier"),
        primary_stats=stats.get("primary_stats", {}),
        alt_stats=stats.get("alt_stats", {}),
        avg_risk=stats.get("avg_risk"),
        route_mode=stats.get("route_mode", "safe"),
    )

    if pdf_bytes is None:
        return no_update

    return dcc.send_bytes(pdf_bytes, filename="arktik_risk_rota_raporu.pdf")

# =========================================================
# ÇALIŞTIR
# =========================================================
if __name__ == "__main__":
    app.run(debug=True)