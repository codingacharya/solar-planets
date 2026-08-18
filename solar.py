"""
Real-Time Solar System Simulation — Smooth (Client-Side) Animation Edition
----------------------------------------------------------------------------
Run with: streamlit run solar_system_sim.py
Requires: streamlit>=1.30, plotly, numpy, pandas

Why no more blinking:
The previous version used st.fragment(run_every=...) to re-render the chart
every N milliseconds from the PYTHON side. Each rerun re-sent a fresh figure
to the browser, which caused a visible flash/blink as the component redrew.

This version instead builds ONE Plotly figure containing many precomputed
animation "frames" plus a native Play/Pause button and scrubber. Once it's
sent to the browser, Plotly.js animates locally in JavaScript — zero server
round-trips while playing, so there's no flicker at all.
"""

import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

st.set_page_config(page_title="Solar System Simulator", layout="wide", page_icon="🪐")

# ============================================================
# DATA — real astronomical values
# ============================================================
PLANETS = [
    dict(name="Mercury", au=0.39, period=88.0, diam=4879, incl=7.0, ecc=0.206,
         day_hr=1408, moons=0, color="#B5B5B5", size=6,
         fact="Smallest planet; swings from 430°C to -180°C with no atmosphere to trap heat."),
    dict(name="Venus", au=0.72, period=224.7, diam=12104, incl=3.4, ecc=0.007,
         day_hr=5832, moons=0, color="#E8C39E", size=9,
         fact="Hottest planet (~465°C) due to runaway greenhouse effect; spins backwards."),
    dict(name="Earth", au=1.00, period=365.25, diam=12742, incl=0.0, ecc=0.017,
         day_hr=24, moons=1, color="#4A90D9", size=10,
         fact="Only known planet with liquid water oceans and life."),
    dict(name="Mars", au=1.52, period=687.0, diam=6779, incl=1.9, ecc=0.093,
         day_hr=24.7, moons=2, color="#C1440E", size=8,
         fact="Home to Olympus Mons, the tallest volcano in the solar system."),
    dict(name="Jupiter", au=5.20, period=4331, diam=139820, incl=1.3, ecc=0.048,
         day_hr=9.9, moons=95, color="#D8A47F", size=20,
         fact="Largest planet; its Great Red Spot is a storm bigger than Earth."),
    dict(name="Saturn", au=9.58, period=10747, diam=116460, incl=2.5, ecc=0.054,
         day_hr=10.7, moons=146, color="#E3C16F", size=18,
         fact="Its rings are made of ice and rock, some pieces as small as dust."),
    dict(name="Uranus", au=19.18, period=30589, diam=50724, incl=0.8, ecc=0.047,
         day_hr=17.2, moons=27, color="#9FE3E3", size=14,
         fact="Rotates on its side, likely from an ancient massive collision."),
    dict(name="Neptune", au=30.07, period=59800, diam=49244, incl=1.8, ecc=0.010,
         day_hr=16.1, moons=14, color="#4166F5", size=14,
         fact="Windiest planet, with gusts up to 2,100 km/h."),
]

MOONS = {
    "Earth": [dict(name="Moon", rel_r=1.6, period=27.3, size=3, color="#CCCCCC")],
    "Mars": [dict(name="Phobos", rel_r=1.2, period=0.32, size=2, color="#999999"),
             dict(name="Deimos", rel_r=1.6, period=1.26, size=2, color="#777777")],
    "Jupiter": [dict(name="Io", rel_r=1.5, period=1.8, size=3, color="#E8D28A"),
                dict(name="Europa", rel_r=1.9, period=3.6, size=3, color="#C9B896"),
                dict(name="Ganymede", rel_r=2.4, period=7.2, size=4, color="#A69374"),
                dict(name="Callisto", rel_r=2.9, period=16.7, size=4, color="#8B7A66")],
    "Saturn": [dict(name="Titan", rel_r=2.0, period=15.9, size=4, color="#D9B26F")],
}

STEP_PRESETS = {
    "1 day / frame": 1,
    "1 week / frame": 7,
    "1 month / frame": 30,
    "3 months / frame": 91,
    "1 year / frame": 365,
}

START_EPOCH = datetime(2026, 1, 1)

# ============================================================
# SESSION STATE
# ============================================================
st.session_state.setdefault("selected_planet", "Earth")
st.session_state.setdefault("explore_days", 0)

# ============================================================
# SIDEBAR CONTROLS
# ============================================================
st.sidebar.header("⚙️ Simulation Controls")

step_label = st.sidebar.select_slider("Time step per animation frame",
                                       options=list(STEP_PRESETS.keys()),
                                       value="1 month / frame")
step_days = STEP_PRESETS[step_label]

n_frames = st.sidebar.slider("Number of frames (playback length)", 30, 300, 120, step=10)
frame_ms = st.sidebar.slider("Frame duration (ms) — lower = faster", 20, 300, 60, step=10)

st.sidebar.caption(
    f"This plays ≈{n_frames * step_days:,} simulated days "
    f"(~{n_frames * step_days / 365.25:.1f} years) over "
    f"~{n_frames * frame_ms / 1000:.1f} seconds, then loops if you press Play again."
)

st.sidebar.divider()
view_3d = st.sidebar.toggle("🌐 3D view (with orbital inclination)", value=False)
scale_mode = st.sidebar.radio("Distance scale", ["Compressed (visual)", "True to scale (AU)"], index=0)
show_orbits = st.sidebar.checkbox("Show orbit paths", value=True)
show_labels = st.sidebar.checkbox("Show planet labels", value=True)
show_moons = st.sidebar.checkbox("Show major moons", value=True)
show_belt = st.sidebar.checkbox("Show asteroid belt", value=True)
trail_on = st.sidebar.checkbox("Show comet-style trails", value=False)
trail_len = st.sidebar.slider("Trail length (days)", 10, 300, 60) if trail_on else 60

st.sidebar.divider()
selected_names = st.sidebar.multiselect("Planets shown", [p["name"] for p in PLANETS],
                                         default=[p["name"] for p in PLANETS])
st.session_state.selected_planet = st.sidebar.selectbox(
    "🔍 Focus / inspect planet", [p["name"] for p in PLANETS],
    index=[p["name"] for p in PLANETS].index(st.session_state.selected_planet)
    if st.session_state.selected_planet in [p["name"] for p in PLANETS] else 2
)
zoom_to_planet = st.sidebar.checkbox("Zoom camera to focused planet", value=False)

active = [p for p in PLANETS if p["name"] in selected_names]

# ============================================================
# HELPERS
# ============================================================
def scaled_radius(p):
    if scale_mode == "True to scale (AU)":
        return p["au"] * 15
    return 4 + 3.6 * np.sqrt(p["au"])  # compressed, visually balanced


def planet_position(p, t, radius):
    angle = 2 * np.pi * (t / p["period"])
    a = radius
    b = radius * np.sqrt(1 - p["ecc"] ** 2)
    x = a * np.cos(angle)
    y = b * np.sin(angle)
    if view_3d:
        incl = np.radians(p["incl"])
        z = np.sin(angle) * radius * np.sin(incl)
        y = y * np.cos(incl)
        return x, y, z
    return x, y, 0.0


def scatter_cls():
    return go.Scatter3d if view_3d else go.Scatter


def make_planet_trace(p, t):
    r = scaled_radius(p)
    x, y, z = planet_position(p, t, r)
    kwargs = dict(
        x=[x], y=[y],
        mode="markers+text" if show_labels else "markers",
        marker=dict(size=p["size"], color=p["color"], line=dict(width=1, color="white")),
        text=[p["name"]] if show_labels else None,
        textposition="top center",
        textfont=dict(color="white", size=10),
        name=p["name"],
        hovertext=f"{p['name']}<br>{p['diam']:,} km diameter<br>{p['moons']} moons",
        hoverinfo="text",
    )
    if view_3d:
        kwargs["z"] = [z]
    return scatter_cls()(**kwargs)


def make_trail_trace(p, t):
    r = scaled_radius(p)
    pts = np.linspace(t - trail_len, t, 20)
    xs, ys, zs = [], [], []
    for tt in pts:
        x, y, z = planet_position(p, tt, r)
        xs.append(x); ys.append(y); zs.append(z)
    kwargs = dict(x=xs, y=ys, mode="lines", line=dict(color=p["color"], width=2),
                  opacity=0.35, hoverinfo="skip", showlegend=False)
    if view_3d:
        kwargs["z"] = zs
    return scatter_cls()(**kwargs)


def make_moon_trace(p, m, t):
    r = scaled_radius(p)
    x, y, z = planet_position(p, t, r)
    m_angle = 2 * np.pi * (t / m["period"])
    mr = p["size"] * 0.15 + m["rel_r"] * (p["size"] / 10)
    mx, my = x + mr * np.cos(m_angle), y + mr * np.sin(m_angle)
    kwargs = dict(x=[mx], y=[my], mode="markers",
                  marker=dict(size=m["size"], color=m["color"]),
                  name=m["name"], hoverinfo="name", showlegend=False)
    if view_3d:
        kwargs["z"] = [z]
    return scatter_cls()(**kwargs)


# Fixed order of dynamic (per-frame) items so trace indices line up across all frames
dynamic_items = []
for p in active:
    dynamic_items.append(("planet", p))
    if trail_on:
        dynamic_items.append(("trail", p))
for p in active:
    if show_moons and p["name"] in MOONS:
        for m in MOONS[p["name"]]:
            dynamic_items.append(("moon", p, m))


def make_dynamic_trace(item, t):
    if item[0] == "planet":
        return make_planet_trace(item[1], t)
    if item[0] == "trail":
        return make_trail_trace(item[1], t)
    return make_moon_trace(item[1], item[2], t)


def build_static_traces():
    traces = []
    sun_kwargs = dict(x=[0], y=[0], mode="markers",
                       marker=dict(size=26 if not view_3d else 10, color="#FFD700",
                                   line=dict(width=2, color="orange")),
                       name="Sun", hoverinfo="name")
    if view_3d:
        sun_kwargs["z"] = [0]
    traces.append(scatter_cls()(**sun_kwargs))

    if show_orbits:
        for p in active:
            r = scaled_radius(p)
            theta = np.linspace(0, 2 * np.pi, 200)
            ox = r * np.cos(theta)
            oy = r * np.sqrt(1 - p["ecc"] ** 2) * np.sin(theta)
            kwargs = dict(x=ox, y=oy, mode="lines",
                          line=dict(color="rgba(255,255,255,0.15)", width=1),
                          hoverinfo="skip", showlegend=False)
            if view_3d:
                incl = np.radians(p["incl"])
                oz = np.sin(theta) * r * np.sin(incl)
                kwargs["y"] = oy * np.cos(incl)
                kwargs["z"] = oz
            traces.append(scatter_cls()(**kwargs))

    if show_belt:
        rng = np.random.default_rng(42)
        if scale_mode == "True to scale (AU)":
            belt_inner, belt_outer = 2.2 * 15, 3.2 * 15
        else:
            belt_inner, belt_outer = 4 + 3.6 * np.sqrt(2.2), 4 + 3.6 * np.sqrt(3.2)
        n = 300
        rr = rng.uniform(belt_inner, belt_outer, n)
        aa = rng.uniform(0, 2 * np.pi, n)
        bx, by = rr * np.cos(aa), rr * np.sin(aa)
        kwargs = dict(x=bx, y=by, mode="markers",
                      marker=dict(size=1.5, color="rgba(180,180,180,0.5)"),
                      hoverinfo="skip", showlegend=False, name="Asteroid Belt")
        if view_3d:
            kwargs["z"] = np.zeros(n)
        traces.append(scatter_cls()(**kwargs))
    return traces


def build_animated_figure():
    static_traces = build_static_traces()
    n_static = len(static_traces)
    dynamic_trace_indices = list(range(n_static, n_static + len(dynamic_items)))

    frame_times = [i * step_days for i in range(n_frames)]

    initial_dynamic = [make_dynamic_trace(item, frame_times[0]) for item in dynamic_items]
    fig = go.Figure(data=static_traces + initial_dynamic)

    frames = []
    for i, t in enumerate(frame_times):
        frame_data = [make_dynamic_trace(item, t) for item in dynamic_items]
        frames.append(go.Frame(data=frame_data, traces=dynamic_trace_indices, name=str(i)))
    fig.frames = frames

    max_r = max([scaled_radius(p) for p in active], default=10)
    axis_range = [-max_r - 3, max_r + 3]
    if zoom_to_planet:
        fpz = next((p for p in active if p["name"] == st.session_state.selected_planet), None)
        if fpz:
            fr = scaled_radius(fpz)
            span = max(fr * 0.5, 4)
            axis_range = [-fr - span, fr + span]

    play_button = dict(
        type="buttons", showactive=False, x=0.05, y=0.02, xanchor="left", yanchor="bottom",
        buttons=[
            dict(label="▶ Play", method="animate",
                 args=[None, {"frame": {"duration": frame_ms, "redraw": True},
                               "fromcurrent": True, "transition": {"duration": 0},
                               "mode": "immediate"}]),
            dict(label="⏸ Pause", method="animate",
                 args=[[None], {"frame": {"duration": 0, "redraw": False},
                                 "mode": "immediate", "transition": {"duration": 0}}]),
        ],
    )
    slider = dict(
        active=0, x=0.15, y=0.02, len=0.8, xanchor="left", yanchor="bottom",
        currentvalue=dict(prefix="Day: ", visible=True, font=dict(color="white", size=12)),
        pad=dict(t=30),
        steps=[dict(method="animate", label=str(int(t)),
                    args=[[str(i)], {"frame": {"duration": 0, "redraw": True},
                                      "mode": "immediate", "transition": {"duration": 0}}])
               for i, t in enumerate(frame_times)],
    )

    layout_common = dict(
        plot_bgcolor="black", paper_bgcolor="black",
        showlegend=False, margin=dict(l=0, r=0, t=0, b=0), height=700,
        updatemenus=[play_button], sliders=[slider],
    )
    if view_3d:
        fig.update_layout(
            scene=dict(
                xaxis=dict(visible=False, range=axis_range),
                yaxis=dict(visible=False, range=axis_range),
                zaxis=dict(visible=False, range=[-max_r / 2, max_r / 2]),
                bgcolor="black",
            ),
            **layout_common,
        )
    else:
        fig.update_layout(
            xaxis=dict(range=axis_range, visible=False, scaleanchor="y"),
            yaxis=dict(range=axis_range, visible=False),
            **layout_common,
        )
    return fig


# ============================================================
# HEADER + PLANET DETAIL PANEL
# ============================================================
st.title("🪐 Real-Time Solar System Simulation")
st.caption("Press ▶ Play on the chart below — the animation runs entirely in your browser, "
           "so it stays smooth with no flicker. Drag the slider to scrub to any day.")

fp = next((p for p in PLANETS if p["name"] == st.session_state.selected_planet), None)
if fp:
    with st.expander(f"ℹ️ {fp['name']} — quick facts", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Diameter", f"{fp['diam']:,} km")
        c2.metric("Orbital period", f"{fp['period']:.0f} days")
        c3.metric("Day length", f"{fp['day_hr']:.1f} hrs")
        c4.metric("Moons", fp["moons"])
        st.caption(fp["fact"])

st.plotly_chart(build_animated_figure(), use_container_width=True, key="main_chart")

# ============================================================
# Manual date explorer (separate from the animated chart — used for the
# facts / table / export snapshots below). Dragging this does trigger a
# normal Streamlit rerun, but only on release, not continuously.
# ============================================================
st.session_state.explore_days = st.slider(
    "📅 Explore a specific date (for the panels below)",
    min_value=0, max_value=365 * 20, value=st.session_state.explore_days, step=1,
)
current_date = START_EPOCH + timedelta(days=st.session_state.explore_days)
st.metric("Selected date", current_date.strftime("%d %b %Y"))

# ============================================================
# TABS: comparison table, Kepler's law, data export
# ============================================================
tab1, tab2, tab3 = st.tabs(["📊 Planet Comparison", "📐 Kepler's Third Law", "⬇️ Export Data"])

with tab1:
    df = pd.DataFrame(PLANETS)[["name", "au", "period", "diam", "moons", "day_hr", "ecc"]]
    df.columns = ["Planet", "Distance (AU)", "Orbital Period (days)", "Diameter (km)",
                  "Moons", "Day Length (hrs)", "Eccentricity"]
    st.dataframe(df, use_container_width=True, hide_index=True)

with tab2:
    st.write("Kepler's Third Law states **T² ∝ a³** — verified below using real planetary data "
             "(plotting log(T) vs log(a) gives a straight line of slope 1.5).")
    kdf = pd.DataFrame(PLANETS)
    kdf["log_a"] = np.log10(kdf["au"])
    kdf["log_T"] = np.log10(kdf["period"] / 365.25)
    fig_k = go.Figure()
    fig_k.add_trace(go.Scatter(x=kdf["log_a"], y=kdf["log_T"], mode="markers+text",
                                text=kdf["name"], textposition="top center",
                                marker=dict(size=10, color=kdf["au"], colorscale="Viridis")))
    z = np.polyfit(kdf["log_a"], kdf["log_T"], 1)
    xline = np.linspace(kdf["log_a"].min(), kdf["log_a"].max(), 50)
    fig_k.add_trace(go.Scatter(x=xline, y=np.polyval(z, xline), mode="lines",
                                name=f"Fit slope = {z[0]:.3f}", line=dict(dash="dash")))
    fig_k.update_layout(xaxis_title="log₁₀(Distance in AU)", yaxis_title="log₁₀(Period in years)",
                         height=450, plot_bgcolor="white")
    st.plotly_chart(fig_k, use_container_width=True)
    st.caption(f"Fitted slope ≈ {z[0]:.3f} (theoretical value = 1.5), confirming Kepler's Third Law.")

with tab3:
    export_rows = []
    for p in active:
        x, y, _ = planet_position(p, st.session_state.explore_days, scaled_radius(p))
        export_rows.append({"Planet": p["name"], "x": round(x, 3), "y": round(y, 3),
                             "Date": current_date.strftime("%Y-%m-%d")})
    export_df = pd.DataFrame(export_rows)
    st.dataframe(export_df, use_container_width=True, hide_index=True)
    st.download_button("Download positions as CSV", export_df.to_csv(index=False),
                        file_name="planet_positions.csv")

st.caption("Distances and periods use real astronomical data; 'Compressed' mode uses √AU scaling "
           "so inner planets remain visible alongside outer ones. No single view can keep both "
           "sizes and distances simultaneously to real scale.")