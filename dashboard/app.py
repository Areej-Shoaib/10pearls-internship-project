import json
import subprocess
import sys
from pathlib import Path
import textwrap

import streamlit as st
import plotly.graph_objects as go


# ======================================================
# Page Configuration
# ======================================================

st.set_page_config(
    page_title="CityAQI Forecast",
    layout="wide"
)


# ======================================================
# Paths
# ======================================================

DASHBOARD_DIR = Path(__file__).parent
PROJECT_ROOT = DASHBOARD_DIR.parent

INFERENCE_SCRIPT = (
    PROJECT_ROOT
    / "model_training"
    / "inference.py"
)

INFERENCE_PYTHON = Path(sys.executable)

PREDICTION_FILE = (
    PROJECT_ROOT
    / "model_training"
    / "prediction_result.json"
)


# ======================================================
# Load Custom CSS
# ======================================================

css_path = DASHBOARD_DIR / "style.css"

with open(css_path) as f:
    st.markdown(
        f"<style>{f.read()}</style>",
        unsafe_allow_html=True
    )


# ======================================================
# Cities
# ======================================================

cities = [
    "Faisalabad",
    "Gujranwala",
    "Hyderabad",
    "Islamabad",
    "Karachi",
    "Lahore",
    "Multan",
    "Peshawar",
    "Quetta",
    "Rawalpindi",
    "Sialkot",
    "Sukkur"
]


# ======================================================
# Run Inference
# ======================================================

def run_inference(city):

    if not INFERENCE_PYTHON.exists():
        raise FileNotFoundError(
            f"Hopsworks Python environment not found:\n"
            f"{INFERENCE_PYTHON}"
        )

    if not INFERENCE_SCRIPT.exists():
        raise FileNotFoundError(
            f"Inference script not found:\n"
            f"{INFERENCE_SCRIPT}"
        )

    result = subprocess.run(
        [
            str(INFERENCE_PYTHON),
            str(INFERENCE_SCRIPT),
            city
        ],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(
            result.stderr
            or result.stdout
            or "Inference failed."
        )

    if not PREDICTION_FILE.exists():
        raise FileNotFoundError(
            "prediction_result.json was not created."
        )

    with open(PREDICTION_FILE, "r") as f:
        data = json.load(f)

    return data


# ======================================================
# AQI Category
# ======================================================

def get_aqi_status(aqi):

    if aqi <= 20:
        return (
            "Good",
            "Air quality is good. Air pollution poses little or no risk."
        )

    elif aqi <= 40:
        return (
            "Fair",
            "Air quality is generally acceptable, with low risk to health."
        )

    elif aqi <= 60:
        return (
            "Moderate",
            "Air quality is acceptable, but some pollutants may affect sensitive individuals."
        )

    elif aqi <= 80:
        return (
            "Poor",
            "Air pollution may affect health. Sensitive individuals should consider reducing prolonged outdoor activity."
        )

    elif aqi <= 100:
        return (
            "Very Poor",
            "Health effects may occur. Sensitive individuals should avoid prolonged outdoor activity."
        )

    else:
        return (
            "Extremely Poor",
            "Health alert: everyone may experience significant health effects. Avoid prolonged outdoor exposure."
        )
    

# ======================================================
# Header
# ======================================================

header_col1, header_col2 = st.columns([4, 1])

with header_col1:

    st.title("CityAQI Forecast")

    st.write(
        "AI-powered 3-day AQI forecasting"
    )


with header_col2:

    st.markdown(
        '<div class="live-indicator">'
        '<span>●</span> Live Data'
        '</div>',
        unsafe_allow_html=True
    )


st.divider()


# ======================================================
# AQI Gauge
# ======================================================

def render_aqi_gauge(aqi, category):

    gauge_value = max(0, min(float(aqi), 120))

    # Map AQI 0–120 to gauge angle -90° to +90°
    needle_angle = -90 + (gauge_value / 120) * 180

    gauge_html = (
        '<div class="aqi-gauge-card">'
        '<div class="aqi-gauge-title">Current AQI (European)</div>'

        '<div class="aqi-gauge">'

        '<div class="aqi-gauge-track"></div>'

        '<div class="aqi-gauge-inner"></div>'

        # Needle
        f'<div class="aqi-gauge-needle" '
        f'style="transform: rotate({needle_angle}deg);"></div>'

        # Pivot
        '<div class="aqi-gauge-pivot"></div>'

        # AQI value
        f'<div class="aqi-gauge-value">{aqi:.0f}</div>'

        '</div>'

        f'<div class="aqi-gauge-category">{category}</div>'

        '<div class="aqi-gauge-scale">'
        '<span>0</span>'
        '<span>20</span>'
        '<span>40</span>'
        '<span>60</span>'
        '<span>80</span>'
        '<span>100+</span>'
        '</div>'

        '</div>'
    )

    st.markdown(
        gauge_html,
        unsafe_allow_html=True
    )

# ======================================================
# City Selection + AQI Gauge
# ======================================================

city_col, gauge_col = st.columns(
    [1, 1],
    gap="large"
)

with city_col:

    selected_city = st.selectbox(
        "**Select City**",
        cities
    )

    st.markdown('<div class="last-updated-label">Last Updated</div>', unsafe_allow_html=True)

    last_updated_placeholder = st.empty()

with gauge_col:

    gauge_placeholder = st.empty()


# ======================================================
# Fetch Real Data
# ======================================================

try:

    with st.spinner(
        f"Fetching live AQI data for {selected_city}..."
    ):

        data = run_inference(
            selected_city
        )

except Exception as e:

    st.error(
        "Unable to fetch live AQI data."
    )

    st.exception(e)

    st.stop()


# ======================================================
# Extract Data
# ======================================================

city = data["city"]
timestamp = data["timestamp"]
last_updated_placeholder.markdown(f'<div class="last-updated-value">{timestamp}</div>', unsafe_allow_html=True)

current_conditions = data[
    "current_conditions"
]

current_aqi = current_conditions[
    "aqi"
]

aqi_category, recommendation = get_aqi_status(
    current_aqi
    )

with gauge_placeholder.container():
    render_aqi_gauge(
        current_aqi,
        aqi_category
        )


pm2_5 = current_conditions[
    "pm2_5"
]

temperature = current_conditions[
    "temperature"
]

humidity = current_conditions[
    "humidity"
]

wind_speed = current_conditions[
    "wind_speed"
]

pm10 = current_conditions[
    "pm10"
]

predictions = data[
    "predictions"
]

prediction_24h = predictions[
    "24h"
]

prediction_48h = predictions[
    "48h"
]

prediction_72h = predictions[
    "72h"
]

# ======================================================
# SHAP Explainability
# ======================================================

explanations = data.get(
    "explanations",
    {}
)


FEATURE_LABELS = {
    "european_aqi": "European AQI",
    "pm2_5": "PM2.5",
    "pm10": "PM10",
    "temperature_2m": "Temperature",
    "relative_humidity_2m": "Humidity",
    "pressure_msl": "Atmospheric Pressure",
    "wind_speed_10m": "Wind Speed",
    "carbon_monoxide": "Carbon Monoxide",
    "nitrogen_dioxide": "Nitrogen Dioxide",
    "sulphur_dioxide": "Sulphur Dioxide",
    "ozone": "Ozone",
    "aqi_change_rate": "AQI Change Rate",
    "hour_sin": "Time of Day",
    "hour_cos": "Time of Day",
    "month_sin": "Seasonal Pattern",
    "month_cos": "Seasonal Pattern"
}


def get_feature_label(feature):

    if feature in FEATURE_LABELS:
        return FEATURE_LABELS[feature]

    if feature.startswith("city_"):
        return "City"

    if feature.startswith("weekday_"):
        return "Day of Week"

    return feature.replace(
        "_",
        " "
    ).title()


def render_explanation(explanation):

    if not explanation:
        st.caption(
            "Explanation unavailable."
        )
        return

    positive = [
        item
        for item in explanation
        if item["impact"] > 0
    ]

    negative = [
        item
        for item in explanation
        if item["impact"] < 0
    ]

    # Sort strongest impact first
    positive.sort(
        key=lambda x: x["impact"],
        reverse=True
    )

    negative.sort(
        key=lambda x: abs(x["impact"]),
        reverse=True
    )

    max_impact = max(
        [abs(item["impact"]) for item in explanation]
    )

    # --------------------------------------------------
    # Decreasing AQI contributors (shown first — good news)
    # --------------------------------------------------

    if negative:

        st.markdown(
            '<div class="shap-section-title negative">'
            '↓ Decreasing AQI'
            '</div>',
            unsafe_allow_html=True
        )

        for item in negative:

            feature = get_feature_label(
                item["feature"]
            )

            impact = float(
                item["impact"]
            )

            width = (
                abs(impact) / max_impact
            ) * 100

            st.markdown(
                f'''
                <div class="shap-row">
                    <div class="shap-feature">
                        {feature}
                    </div>
                    <div class="shap-bar-container">
                        <div class="shap-bar negative"
                             style="width: {width:.1f}%;">
                        </div>
                    </div>
                    <div class="shap-value negative">
                        {impact:.2f}
                    </div>
                </div>
                ''',
                unsafe_allow_html=True
            )

    # --------------------------------------------------
    # Increasing AQI contributors (shown second — caution)
    # --------------------------------------------------

    if positive:

        st.markdown(
            '<div class="shap-section-title positive">'
            '↑ Increasing AQI'
            '</div>',
            unsafe_allow_html=True
        )

        for item in positive:

            feature = get_feature_label(
                item["feature"]
            )

            impact = float(
                item["impact"]
            )

            width = (
                abs(impact) / max_impact
            ) * 100

            st.markdown(
                f'''
                <div class="shap-row">
                    <div class="shap-feature">
                        {feature}
                    </div>
                    <div class="shap-bar-container">
                        <div class="shap-bar positive"
                             style="width: {width:.1f}%;">
                        </div>
                    </div>
                    <div class="shap-value positive">
                        +{impact:.2f}
                    </div>
                </div>
                ''',
                unsafe_allow_html=True
            )

st.divider()


# ======================================================
# Current Conditions
# ======================================================

st.subheader("Current Conditions")

col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    st.metric(
        "Current AQI (European)",
        f"{current_aqi:.0f}"
    )

with col2:
    st.metric(
        "Temperature",
        f"{temperature:.1f}°C"
    )

with col3:
    st.metric(
        "Humidity",
        f"{humidity:.0f}%"
    )

with col4:
    st.metric(
        "Wind Speed",
        f"{wind_speed:.1f} km/h"
    )

with col5:
    st.metric(
        "PM2.5",
        f"{pm2_5:.1f} µg/m³"
    )

with col6:
    st.metric(
        "PM10",
        f"{pm10:.1f} µg/m³"
    )


# ======================================================
# 3-Day Forecast
# ======================================================

st.divider()

st.subheader("3-Day AQI Forecast with SHAP Explainability")

forecast_col1, forecast_col2, forecast_col3 = st.columns(
    3,
    gap="large"
)


with forecast_col1:

    with st.container(border=True):

        st.metric(
            "Next 24 Hours",
            f"{prediction_24h:.0f}"
        )

        st.caption(
            "Forecast"
        )

        st.markdown(
            '<div class="shap-card-title">'
            'SHAP Explanation — Why this prediction?'
            '</div>',
            unsafe_allow_html=True
        )

        render_explanation(
            explanations.get("24h", [])
        )


with forecast_col2:

    with st.container(border=True):

        st.metric(
            "Next 48 Hours",
            f"{prediction_48h:.0f}"
        )

        st.caption(
            "Forecast"
        )

        st.markdown(
            '<div class="shap-card-title">'
            'SHAP Explanation — Why this prediction?'
            '</div>',
            unsafe_allow_html=True
        )

        render_explanation(
            explanations.get("48h", [])
        )


with forecast_col3:

    with st.container(border=True):

        st.metric(
            "Next 72 Hours",
            f"{prediction_72h:.0f}"
        )

        st.caption(
            "Forecast"
        )

        st.markdown(
            '<div class="shap-card-title">'
            'SHAP Explanation — Why this prediction?'
            '</div>',
            unsafe_allow_html=True
        )

        render_explanation(
            explanations.get("72h", [])
        )

# ======================================================
# Forecast Chart
# ======================================================

st.divider()

st.subheader("AQI Forecast")

chart_data = {
    "Forecast": [
        "24 Hours",
        "48 Hours",
        "72 Hours"
    ],
    "AQI": [
        prediction_24h,
        prediction_48h,
        prediction_72h
    ]
}

import plotly.graph_objects as go

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=chart_data["Forecast"],
        y=chart_data["AQI"],
        mode="lines+markers+text",
        text=[
            f"{prediction_24h:.1f}",
            f"{prediction_48h:.1f}",
            f"{prediction_72h:.1f}"
        ],
        textposition="top center",
        line=dict(
            color="#15803d",
            width=3,
            shape="spline"
        ),
        marker=dict(
            size=9,
            color="#15803d",
            line=dict(
                color="#ffffff",
                width=2
            )
        ),
        fill="tozeroy",
        fillcolor="rgba(21, 128, 61, 0.08)",
        hovertemplate="AQI: %{y:.1f}<extra></extra>"
    )
)

fig.update_layout(
    height=360,
    margin=dict(
        l=20,
        r=20,
        t=30,
        b=20
    ),
    paper_bgcolor="#ffffff",
    plot_bgcolor="#ffffff",
    xaxis=dict(
        title=None,
        showgrid=False,
        range=[-0.25, 2.25]
    ),
    yaxis=dict(
        title="AQI",
        showgrid=True,
        gridcolor="#e5ebe7",
        zeroline=False
    ),
    font=dict(
        color="#1a2e22"
    ),
    hoverlabel=dict(
        bgcolor="#ffffff",
        font_size=13
    )
)

st.plotly_chart(
    fig,
    use_container_width=True,
    config={
        "displayModeBar": False
    }
)


# ======================================================
# AQI Status
# ======================================================

st.divider()

st.subheader("Air Quality Status")

status_col1, status_col2 = st.columns(
    [1, 2],
    gap="large"
)


with status_col1:

    st.markdown(
        '<div class="info-card">'
        '<div class="info-card-label">'
        'AQI Category'
        '</div>'
        f'<div class="info-card-value">'
        f'{aqi_category}'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )


with status_col2:

    st.markdown(
        '<div class="info-card">'
        '<div class="info-card-label">'
        'Recommendation'
        '</div>'
        f'<div class="info-card-text">'
        f'{recommendation}'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )


# ======================================================
# Footer
# ======================================================

st.divider()

st.caption(
    "AQI Predictor • AI-powered 3-day air quality forecasting"
)

