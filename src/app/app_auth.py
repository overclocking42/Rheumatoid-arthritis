import os
import json
import io
from urllib.parse import quote_plus
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import requests
from PIL import Image
import cv2
import torch
from torchvision import models, transforms
from datetime import datetime
import google.generativeai as genai
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except Exception:
    Groq = None
    GROQ_AVAILABLE = False
from database import Database
from config import GEMINI_API_KEY, GEMINI_API_KEY_ALTERNATE, GROQ_API_KEY
from model_utils import (
    ROOT,
    load_image_model_artifact,
    load_numeric_model_artifacts,
    normalize_numeric_features,
    predict_xray_image,
    preprocess_xray_image,
)

# Gemini PDF extraction feature disabled
GEMINI_AVAILABLE = False

st.set_page_config(
    page_title="Clinical RA Assessment System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize database
@st.cache_resource
def get_db():
    """Get or create database instance."""
    return Database("data/app_users.db")

db = get_db()

# Custom CSS for medical dashboard
st.markdown("""
<style>
    :root {
        --sky: #0f5c7a;
        --teal: #0d8a7c;
        --mint: #dff7f1;
        --cream: #fffaf1;
        --soft-red: #fff1f0;
        --soft-orange: #fff5eb;
        --soft-yellow: #fff9db;
        --soft-green: #edf9ef;
        --ink: #15313f;
    }
    .stApp {
        background:
            radial-gradient(circle at top right, rgba(13, 138, 124, 0.10), transparent 22%),
            radial-gradient(circle at top left, rgba(15, 92, 122, 0.10), transparent 18%),
            linear-gradient(180deg, #f8fcfd 0%, #f7fbf8 100%);
        color: var(--ink);
    }
    .topbar-card {
        background: #ffffff;
        border: 1px solid #d7e6ed;
        border-radius: 24px;
        padding: 20px 24px;
        margin: 6px 0 14px 0;
        box-shadow: 0 14px 30px rgba(15, 44, 60, 0.06);
    }
    .topbar-title {
        margin: 0;
        color: #173646;
        font-size: 2rem;
    }
    .topbar-subtitle {
        margin: 8px 0 0 0;
        color: #547283;
        font-size: 1rem;
    }
    .topbar-highlights {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
        margin-top: 14px;
    }
    .highlight-chip {
        background: linear-gradient(135deg, #f1f8fb 0%, #eefbf8 100%);
        border: 1px solid #d7e6ed;
        color: #264b5e;
        border-radius: 16px;
        padding: 10px 14px;
        font-size: 0.92rem;
        font-weight: 700;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
    }
    .healthy-card {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        padding: 20px;
        border-radius: 10px;
        color: black;
        margin: 10px 0;
    }
    .warning-card {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        padding: 20px;
        border-radius: 10px;
        color: black;
        margin: 10px 0;
    }
    .erosive-card {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
    }
    .non-erosive-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
    }
    .auth-container {
        max-width: 400px;
        margin: 50px auto;
        padding: 30px;
        border-radius: 10px;
        background: #f0f2f6;
    }
    .severity-panel {
        border-radius: 22px;
        padding: 24px;
        margin: 10px 0 18px 0;
        border: 2px solid transparent;
        box-shadow: 0 16px 32px rgba(0,0,0,0.08);
        position: relative;
        overflow: hidden;
    }
    .severity-panel::after {
        content: "";
        position: absolute;
        right: -40px;
        top: -40px;
        width: 150px;
        height: 150px;
        border-radius: 50%;
        background: rgba(255,255,255,0.28);
    }
    .severity-topline {
        display: inline-block;
        font-size: 0.86rem;
        font-weight: 700;
        padding: 6px 12px;
        border-radius: 999px;
        margin-bottom: 12px;
    }
    .severity-panel h3 {
        margin: 0 0 6px 0;
        font-size: 1.65rem;
    }
    .severity-panel h4 {
        margin: 0 0 12px 0;
        font-size: 1rem;
        opacity: 0.98;
    }
    .severity-symbol {
        font-size: 3rem;
        line-height: 1;
        margin: 4px 0 12px 0;
    }
    .severity-panel ul {
        margin: 10px 0 0 18px;
    }
    .severity-low {
        background: linear-gradient(180deg, #ecfff2 0%, #d6f5de 100%);
        border-color: #34a853;
        color: #154824;
    }
    .severity-low .severity-topline {
        background: #34a853;
        color: white;
    }
    .severity-mild {
        background: linear-gradient(180deg, #fffbe3 0%, #fff0ad 100%);
        border-color: #d7a600;
        color: #5b4300;
    }
    .severity-mild .severity-topline {
        background: #d7a600;
        color: white;
    }
    .severity-medium {
        background: linear-gradient(180deg, #fff4e8 0%, #ffd4a1 100%);
        border-color: #ef8c12;
        color: #72390b;
    }
    .severity-medium .severity-topline {
        background: #ef8c12;
        color: white;
    }
    .severity-high {
        background: linear-gradient(180deg, #fff1ef 0%, #ffc7c0 100%);
        border-color: #d93025;
        color: #6e1a13;
    }
    .severity-high .severity-topline {
        background: #d93025;
        color: white;
    }
    .color-legend {
        display: flex;
        gap: 12px;
        flex-wrap: wrap;
        margin: 12px 0 6px 0;
    }
    .legend-chip {
        display: flex;
        align-items: center;
        gap: 8px;
        background: white;
        border: 1px solid #d7e6ed;
        border-radius: 999px;
        padding: 8px 12px;
        color: #28495b;
        font-size: 0.92rem;
    }
    .legend-dot {
        width: 12px;
        height: 12px;
        border-radius: 50%;
        display: inline-block;
    }
    .plain-card {
        background: white;
        border: 1px solid #dce8ee;
        border-radius: 18px;
        padding: 18px;
        margin: 10px 0;
        box-shadow: 0 10px 24px rgba(19, 49, 63, 0.05);
    }
    .plain-card h4 {
        margin: 0 0 8px 0;
    }
    .mini-note {
        font-size: 0.92rem;
        color: #4c6573;
    }
    .doctor-card {
        background: #ffffff;
        border: 1px solid #dce8ee;
        border-radius: 18px;
        padding: 16px;
        margin: 10px 0;
        box-shadow: 0 10px 24px rgba(19, 49, 63, 0.05);
    }
    .doctor-card h4 {
        margin: 0 0 8px 0;
        color: #173646;
    }
    .doctor-meta {
        color: #5c7483;
        font-size: 0.92rem;
        margin: 4px 0;
    }
    .footer-shell {
        background: #ffffff;
        border: 1px solid #d7e6ed;
        border-radius: 24px;
        padding: 18px 22px;
        box-shadow: 0 14px 30px rgba(15, 44, 60, 0.05);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# AUTHENTICATION SYSTEM
# ============================================================================

def init_session_state():
    """Initialize session state variables."""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None
    if 'user_email' not in st.session_state:
        st.session_state.user_email = None
    if 'user_name' not in st.session_state:
        st.session_state.user_name = None
    if 'lab_result' not in st.session_state:
        st.session_state.lab_result = None
    if 'xray_result' not in st.session_state:
        st.session_state.xray_result = None
    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = []
    if 'confirm_delete' not in st.session_state:
        st.session_state.confirm_delete = False
    if 'prefill_ai_prompt' not in st.session_state:
        st.session_state.prefill_ai_prompt = None
    if 'show_doctor_lab' not in st.session_state:
        st.session_state.show_doctor_lab = False
    if 'show_doctor_xray' not in st.session_state:
        st.session_state.show_doctor_xray = False
    if 'show_ai_lab' not in st.session_state:
        st.session_state.show_ai_lab = False
    if 'show_ai_xray' not in st.session_state:
        st.session_state.show_ai_xray = False
    if 'gemini_api_key_input' not in st.session_state:
        st.session_state.gemini_api_key_input = ""
    if 'groq_api_key_input' not in st.session_state:
        st.session_state.groq_api_key_input = ""
    if 'ai_provider' not in st.session_state:
        st.session_state.ai_provider = "Auto"
    if 'ai_model_choice' not in st.session_state:
        st.session_state.ai_model_choice = "llama-3.3-70b-versatile"
    if 'inline_ai_answer_lab' not in st.session_state:
        st.session_state.inline_ai_answer_lab = None
    if 'inline_ai_answer_xray' not in st.session_state:
        st.session_state.inline_ai_answer_xray = None

init_session_state()


def get_severity_theme(level):
    themes = {
        'low': {'css': 'severity-low', 'badge': 'Green Zone', 'icon': '🟢', 'symbol': '✓'},
        'mild': {'css': 'severity-mild', 'badge': 'Yellow Zone', 'icon': '🟡', 'symbol': '!'},
        'medium': {'css': 'severity-medium', 'badge': 'Orange Zone', 'icon': '🟠', 'symbol': '!!'},
        'high': {'css': 'severity-high', 'badge': 'Red Zone', 'icon': '🔴', 'symbol': '⚠'},
    }
    return themes[level]


def render_severity_panel(level, title, summary, action_text, bullet_points, result_label=None, result_kind=None):
    theme = get_severity_theme(level)
    bullets = ''.join(f'<li>{point}</li>' for point in bullet_points)
    result_line = ""
    if result_label and result_kind:
        result_line = f'<p><strong>{result_kind}:</strong> {result_label}</p>'
    st.markdown(
        f"""
        <div class="severity-panel {theme['css']}">
            <div class="severity-topline">{theme['icon']} {theme['badge']}</div>
            <div class="severity-symbol">{theme['symbol']}</div>
            <h3>{title}</h3>
            <h4>{summary}</h4>
            {result_line}
            <p><strong>What to do now:</strong> {action_text}</p>
            <ul>{bullets}</ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_color_legend():
    st.markdown(
        """
        <div class="color-legend">
            <div class="legend-chip"><span class="legend-dot" style="background:#34a853;"></span>Green Zone: looks safer right now</div>
            <div class="legend-chip"><span class="legend-dot" style="background:#d7a600;"></span>Yellow Zone: watch symptoms carefully</div>
            <div class="legend-chip"><span class="legend-dot" style="background:#ef8c12;"></span>Orange Zone: check with a doctor soon</div>
            <div class="legend-chip"><span class="legend-dot" style="background:#d93025;"></span>Red Zone: higher urgency warning</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_saved_result(kind):
    result = st.session_state.lab_result if kind == "lab" else st.session_state.xray_result
    if not result:
        return

    if kind == "lab":
        prediction = result.get("class")
        confidence = result.get("confidence", 0)
        assessment = build_lab_assessment(prediction, confidence)
        render_severity_panel(
            assessment['severity'],
            assessment['title'],
            assessment['summary'],
            assessment['action'],
            assessment['bullets'],
            result_label=result.get('label'),
            result_kind="Detected subtype",
        )
    else:
        label = result.get("label")
        confidence = result.get("confidence", 0)
        assessment = build_xray_assessment(label, confidence)
        render_severity_panel(
            assessment['severity'],
            assessment['title'],
            assessment['summary'],
            assessment['action'],
            assessment['bullets'],
            result_label=result.get('label'),
            result_kind="Imaging finding",
        )

    render_color_legend()
    render_result_actions(kind)
    render_inline_ai_panel(kind)
    render_doctor_finder(kind)


def render_result_actions(kind):
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🤖 Ask AI about this result", key=f"ask_ai_{kind}", use_container_width=True):
            st.session_state[f"show_ai_{kind}"] = True
            if kind == "lab":
                st.session_state.prefill_ai_prompt = "Explain my blood test result in simple words and tell me what food and care I should follow."
            else:
                st.session_state.prefill_ai_prompt = "Explain my X-ray result in simple words and tell me what care steps I should follow."
            st.session_state[f"inline_ai_answer_{kind}"] = ask_case_assistant(st.session_state.prefill_ai_prompt)
    with c2:
        if st.button("🩺 Visit Doctor", key=f"visit_doctor_{kind}", use_container_width=True):
            st.session_state[f"show_doctor_{kind}"] = True


def render_inline_ai_panel(kind):
    if not st.session_state.get(f"show_ai_{kind}", False):
        return

    st.markdown(
        """
        <div class="plain-card">
            <h4>Ask AI about this result</h4>
            <div class="mini-note">Get simple guidance about food, rest, home care, and what to ask the doctor.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    suggestions = [
        "Explain this result in simple words.",
        "What food and daily care should I follow now?",
        "What should I ask the doctor next?",
    ]
    selected_prompt = None
    cols = st.columns(3)
    for idx, suggestion in enumerate(suggestions):
        with cols[idx]:
            if st.button(suggestion, key=f"inline_ai_suggestion_{kind}_{idx}", use_container_width=True):
                selected_prompt = suggestion

    custom_prompt = st.text_input(
        "Ask a custom question",
        key=f"inline_ai_prompt_{kind}",
        placeholder="Example: Can I walk daily? What foods should I avoid?",
    )

    ask_now = st.button("Send to AI", key=f"inline_ai_send_{kind}", use_container_width=True)
    if selected_prompt or (ask_now and custom_prompt.strip()):
        prompt = selected_prompt or custom_prompt.strip()
        st.session_state[f"inline_ai_answer_{kind}"] = ask_case_assistant(prompt)
        st.session_state.prefill_ai_prompt = prompt

    answer = st.session_state.get(f"inline_ai_answer_{kind}")
    if answer:
        st.markdown("### AI response")
        st.info(answer)
        st.caption("You can continue the conversation in the Ask AI tab.")


def format_osm_address(tags):
    parts = [
        tags.get("addr:housenumber"),
        tags.get("addr:street"),
        tags.get("addr:suburb"),
        tags.get("addr:city") or tags.get("addr:town") or tags.get("addr:village"),
        tags.get("addr:state"),
        tags.get("addr:postcode"),
    ]
    clean_parts = [part for part in parts if part]
    return ", ".join(clean_parts) if clean_parts else "Address not available"


@st.cache_data(show_spinner=False, ttl=1800)
def geocode_location(location_text):
    response = requests.get(
        "https://nominatim.openstreetmap.org/search",
        params={
            "q": location_text,
            "format": "jsonv2",
            "limit": 1,
            "addressdetails": 1,
        },
        headers={"User-Agent": "RA-Assessment-App/1.0"},
        timeout=12,
    )
    response.raise_for_status()
    data = response.json()
    if not data:
        return None
    item = data[0]
    return {
        "lat": float(item["lat"]),
        "lon": float(item["lon"]),
        "display_name": item.get("display_name", location_text),
    }


@st.cache_data(show_spinner=False, ttl=1800)
def search_nearby_doctors(location_text, radius_m=20000, limit=8):
    location = geocode_location(location_text)
    if not location:
        return {"location": None, "results": [], "error": "Location not found. Try city name, area, or pincode."}

    overpass_query = f"""
    [out:json][timeout:25];
    (
      nwr(around:{radius_m},{location['lat']},{location['lon']})["healthcare:speciality"~"rheumatology|arthritis|orthopaedics|orthopedics",i];
      nwr(around:{radius_m},{location['lat']},{location['lon']})["amenity"~"doctors|clinic|hospital"];
      nwr(around:{radius_m},{location['lat']},{location['lon']})["healthcare"~"doctor|clinic|hospital"];
    );
    out center tags;
    """

    response = requests.post(
        "https://overpass-api.de/api/interpreter",
        data=overpass_query.encode("utf-8"),
        headers={"User-Agent": "RA-Assessment-App/1.0"},
        timeout=25,
    )
    response.raise_for_status()
    payload = response.json()
    elements = payload.get("elements", [])

    ranked = []
    seen = set()
    for item in elements:
        tags = item.get("tags", {})
        name = tags.get("name")
        lat = item.get("lat") or (item.get("center") or {}).get("lat")
        lon = item.get("lon") or (item.get("center") or {}).get("lon")
        if not name or lat is None or lon is None:
            continue

        dedupe_key = (name.lower(), round(float(lat), 4), round(float(lon), 4))
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)

        speciality_blob = " ".join(
            [
                name,
                tags.get("healthcare:speciality", ""),
                tags.get("speciality", ""),
                tags.get("description", ""),
            ]
        ).lower()
        score = 0
        if "rheumat" in speciality_blob:
            score += 6
        if "arthritis" in speciality_blob:
            score += 5
        if "orthop" in speciality_blob or "joint" in speciality_blob:
            score += 2
        if tags.get("amenity") == "doctors" or tags.get("healthcare") == "doctor":
            score += 2
        if tags.get("amenity") == "clinic" or tags.get("healthcare") == "clinic":
            score += 1
        if tags.get("amenity") == "hospital" or tags.get("healthcare") == "hospital":
            score += 1

        ranked.append(
            {
                "name": name,
                "address": format_osm_address(tags),
                "phone": tags.get("phone") or tags.get("contact:phone") or "Phone not listed",
                "website": tags.get("website") or tags.get("contact:website"),
                "maps_url": f"https://www.google.com/maps/search/?api=1&query={lat},{lon}",
                "score": score,
                "category": tags.get("healthcare:speciality") or tags.get("healthcare") or tags.get("amenity") or "doctor",
            }
        )

    ranked.sort(key=lambda item: (-item["score"], item["name"].lower()))
    return {"location": location, "results": ranked[:limit], "error": None}


def render_doctor_finder(kind):
    if not st.session_state.get(f"show_doctor_{kind}", False):
        return
    st.markdown(
        """
        <div class="plain-card">
            <h4>Find a Rheumatology Doctor</h4>
            <div class="mini-note">Enter your city, area, or pincode. Open a live search page to view specialist address, phone number, and clinic details.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    location = st.text_input(
        "Enter city, area, or pincode",
        key=f"doctor_location_{kind}",
        placeholder="Example: Davanagere or 577004",
    )
    if not location:
        return

    search_now = st.button("Search Nearby Doctors", key=f"doctor_search_btn_{kind}", use_container_width=True)
    if search_now or st.session_state.get(f"doctor_results_{kind}"):
        try:
            if search_now:
                st.session_state[f"doctor_results_{kind}"] = search_nearby_doctors(location)
            result_bundle = st.session_state.get(f"doctor_results_{kind}")
            if not result_bundle:
                return

            if result_bundle.get("error"):
                st.warning(result_bundle["error"])
            else:
                found_for = result_bundle["location"]["display_name"]
                st.success(f"Showing nearby care options for {found_for}")
                results = result_bundle.get("results", [])
                if results:
                    for idx, doctor in enumerate(results, start=1):
                        st.markdown(
                            f"""
                            <div class="doctor-card">
                                <h4>{idx}. {doctor['name']}</h4>
                                <div class="doctor-meta"><strong>Type:</strong> {doctor['category']}</div>
                                <div class="doctor-meta"><strong>Address:</strong> {doctor['address']}</div>
                                <div class="doctor-meta"><strong>Phone:</strong> {doctor['phone']}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        c1, c2 = st.columns(2)
                        with c1:
                            st.link_button(f"Open in Maps #{idx}", doctor["maps_url"], use_container_width=True)
                        with c2:
                            if doctor.get("website"):
                                st.link_button(f"Open Website #{idx}", doctor["website"], use_container_width=True)
                else:
                    st.info("No specialist listing was found in open map data for this area. Use the fallback search links below.")
        except requests.RequestException as exc:
            st.warning(f"Live doctor search could not load right now: {exc}")

    query = quote_plus(f"rheumatologist near {location}")
    maps_url = f"https://www.google.com/maps/search/?api=1&query={query}"
    practo_url = f"https://www.practo.com/search/doctors?results_type=doctor&q=%5B%7B%22word%22%3A%22rheumatologist%22%2C%22autocompleted%22%3Atrue%7D%5D&city={quote_plus(location)}"
    justdial_url = f"https://www.justdial.com/{quote_plus(location)}/Rheumatologists/nct-10457031"
    c1, c2, c3 = st.columns(3)
    with c1:
        st.link_button("Google Maps Search", maps_url, use_container_width=True)
    with c2:
        st.link_button("Practo Search", practo_url, use_container_width=True)
    with c3:
        st.link_button("Justdial Search", justdial_url, use_container_width=True)


def build_lab_assessment(prediction, confidence):
    if prediction == 0:
        return {
            'severity': 'low',
            'title': 'Looks stable',
            'summary': 'The blood test pattern does not show strong rheumatoid arthritis warning signs.',
            'action': 'Keep normal follow-up care and watch for symptoms like long morning stiffness, swelling, or hand pain.',
            'bullets': [
                'No urgent X-ray is needed based on the blood result alone.',
                'Healthy food, movement, and routine checkups are enough for now.',
                'If pain or swelling continues, meet a doctor even if this screen looks stable.',
            ],
            'label': 'Healthy',
        }
    if prediction == 1:
        level = 'high' if confidence >= 80 else 'medium'
        return {
            'severity': level,
            'title': 'Strong RA warning from blood test',
            'summary': 'The blood markers suggest rheumatoid arthritis is likely and should be checked quickly.',
            'action': 'Move to X-ray review and show the result to a doctor or rheumatology team soon.',
            'bullets': [
                'This pattern often matches active RA warning signs.',
                'Joint swelling, pain, and stiffness should be checked early.',
                'Early treatment can reduce future joint damage.',
            ],
            'label': 'Seropositive RA',
        }
    return {
        'severity': 'medium',
        'title': 'Possible early or hidden RA warning',
        'summary': 'The blood test shows inflammation-related concern even though classic antibody signs may be lower.',
        'action': 'X-ray and doctor review are recommended to confirm whether joints are being affected.',
        'bullets': [
            'This may match milder or harder-to-detect RA patterns.',
            'Do not ignore ongoing pain, swelling, or stiffness.',
            'Further checking can catch disease earlier.',
        ],
        'label': 'Seronegative RA',
    }


def build_xray_assessment(label, confidence):
    if label == 'Erosive':
        level = 'high' if confidence >= 0.75 else 'medium'
        return {
            'severity': level,
            'title': 'Bone damage signs seen on X-ray',
            'summary': 'The X-ray suggests erosive change, which means the joints may already be getting damaged.',
            'action': 'Medical review should happen soon so treatment can start or be strengthened.',
            'bullets': [
                'This is a higher-priority result.',
                'Fast treatment may help slow future joint damage.',
                'Bring this result during your doctor visit.',
            ],
            'label': 'Erosive',
        }
    return {
        'severity': 'low',
        'title': 'No clear bone damage seen',
        'summary': 'The X-ray does not show obvious erosion right now.',
        'action': 'Continue follow-up and combine this with symptoms and blood test results.',
        'bullets': [
            'This is a reassuring imaging result.',
            'Early disease can still exist even when X-ray looks better.',
            'If pain continues, follow your doctor’s advice for repeat review.',
        ],
        'label': 'Non-Erosive',
    }


def build_case_snapshot():
    parts = []
    if st.session_state.lab_result:
        lab = st.session_state.lab_result
        parts.append(
            f"Blood test result: {lab['label']}. Severity: {lab.get('severity_label', lab.get('severity', 'not set'))}. "
            f"Simple summary: {lab.get('summary', '')}"
        )
        raw = lab.get('input', [])
        if len(raw) == 6:
            parts.append(
                f"Blood values used: Age {raw[0]}, Gender code {raw[1]}, RF {raw[2]}, Anti-CCP {raw[3]}, CRP {raw[4]}, ESR {raw[5]}."
            )
    if st.session_state.xray_result:
        xray = st.session_state.xray_result
        parts.append(
            f"X-ray result: {xray['label']}. Severity: {xray.get('severity_label', xray.get('severity', 'not set'))}. "
            f"Simple summary: {xray.get('summary', '')}"
        )
    return "\n".join(parts) if parts else "No patient results yet."


def get_ai_suggestions():
    suggestions = [
        "What food should I eat now?",
        "How should I take care of my joints at home?",
        "What should I ask the doctor next?",
    ]
    if st.session_state.lab_result and st.session_state.xray_result:
        suggestions[1] = "Explain both of my results in simple words."
    elif st.session_state.lab_result:
        suggestions[1] = "What does my blood test result mean in simple words?"
    elif st.session_state.xray_result:
        suggestions[1] = "What does my X-ray result mean in simple words?"
    return suggestions


def get_gemini_chat_model():
    api_key = (
        os.getenv('GOOGLE_API_KEY')
        or st.session_state.get('gemini_api_key_input')
        or GEMINI_API_KEY
        or GEMINI_API_KEY_ALTERNATE
    )
    if not api_key:
        return None, "Gemini API key not found. Paste it in the Ask AI tab or export GOOGLE_API_KEY before starting Streamlit."

    try:
        genai.configure(api_key=api_key)
        try:
            return genai.GenerativeModel('gemini-2.0-flash'), None
        except Exception:
            return genai.GenerativeModel('gemini-1.5-pro'), None
    except Exception as exc:
        return None, f"Gemini setup failed: {exc}"


def get_groq_client():
    if not GROQ_AVAILABLE:
        return None, "Groq SDK not installed. Run `pip install groq` or reinstall from requirements."

    api_key = (
        os.getenv('GROQ_API_KEY')
        or st.session_state.get('groq_api_key_input')
        or GROQ_API_KEY
    )
    if not api_key:
        return None, "Groq API key not found. Paste it in the Ask AI tab or export GROQ_API_KEY before starting Streamlit."

    try:
        return Groq(api_key=api_key), None
    except Exception as exc:
        return None, f"Groq setup failed: {exc}"


def get_ai_backend_status():
    provider = st.session_state.get("ai_provider", "Auto")
    if provider in ("Auto", "Groq"):
        client, error = get_groq_client()
        if not error:
            return "Groq", None
        if provider == "Groq":
            return None, error

    if provider in ("Auto", "Gemini"):
        model, error = get_gemini_chat_model()
        if not error:
            return "Gemini", None
        if provider == "Gemini":
            return None, error

    return None, "No working AI provider found. Add a valid Groq or Gemini API key."


def ask_case_assistant_with_groq(user_prompt):
    client, error = get_groq_client()
    if error:
        return error

    case_snapshot = build_case_snapshot()
    model_name = st.session_state.get("ai_model_choice", "llama-3.3-70b-versatile")
    messages = [
        {
            "role": "system",
            "content": (
                "You are a patient-friendly rheumatoid arthritis support assistant inside a demo clinical app. "
                "Use simple language. Do not mention confidence, probabilities, logits, or machine learning internals. "
                "Give practical next steps. Mention that a doctor should confirm treatment decisions."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Patient case context:\n{case_snapshot}\n\n"
                f"User question:\n{user_prompt}\n\n"
                "Answer in very simple language. Give 3 short action points when useful."
            ),
        },
    ]

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.3,
        )
        return (response.choices[0].message.content or "").strip()
    except Exception as exc:
        return f"AI helper could not answer right now: {exc}"

def render_about_footer():
    with st.expander("ℹ️ About This App", expanded=False):
        st.markdown(
            """
            <div class="footer-shell">
                <h4 style="margin-top:0;">What this page does</h4>
                <p>This app gives quick rheumatoid arthritis screening support from blood markers and hand X-rays. The patient-facing screens use simple color-based care levels, while the technical details stay here for judges, developers, and reviewers.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                """
                <div class="plain-card">
                    <h4>How the support flow works</h4>
                    <ol>
                        <li>Enter blood values or upload an X-ray.</li>
                        <li>The saved trained model predicts the likely class.</li>
                        <li>The app converts the prediction into a simple color-based care level.</li>
                        <li>The patient can ask AI follow-up questions or search for nearby doctors.</li>
                    </ol>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                """
                <div class="plain-card">
                    <h4>Model snapshot</h4>
                    <p><strong>Blood test model:</strong> ANN using age, gender, RF, Anti-CCP, CRP, and ESR.</p>
                    <p><strong>X-ray model:</strong> Swin Transformer for erosive vs non-erosive screening.</p>
                    <p><strong>App role:</strong> Fast screening support, not final diagnosis.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        perf_col1, perf_col2 = st.columns(2)
        with perf_col1:
            st.write("### Numeric model comparison")
            st.dataframe(pd.DataFrame({
                'Model': ['XGBoost', 'CatBoost', 'ANN ⭐'],
                'CV Accuracy': ['88.07%', '88.74%', '88.92%'],
                'CV Std Dev': ['±0.82%', '±1.41%', '±0.22%'],
                'F1 Seropositive': ['94.90%', '95.56%', '96.59%'],
                'Selection': ['❌', '❌', '✅']
            }), use_container_width=True, hide_index=True)
        with perf_col2:
            st.write("### Imaging model comparison")
            st.dataframe(pd.DataFrame({
                'Model': ['DenseNet121', 'Swin Transformer ⭐'],
                'Mean Accuracy': ['77.00%', '83.50%'],
                'Best Fold Accuracy': ['81.67%', '85.83%'],
                'Mean F1': ['77.34%', '90.38%'],
                'Selection': ['❌', '✅']
            }), use_container_width=True, hide_index=True)

        viz_tab1, viz_tab2 = st.tabs(["📈 Numeric Visualizations", "📷 Imaging Visualizations"])
        with viz_tab1:
            numeric_viz_path = os.path.join(ROOT, 'reports', 'numeric', 'visualizations')
            if os.path.exists(numeric_viz_path):
                for name, caption in [
                    ('01_5fold_cross_validation_comparison.png', 'Cross-validation comparison'),
                    ('08_confusion_matrices.png', 'Confusion matrices'),
                    ('11_feature_importance.png', 'Feature importance'),
                ]:
                    path = os.path.join(numeric_viz_path, name)
                    if os.path.exists(path):
                        st.image(path, use_container_width=True, caption=caption)
            else:
                st.info("Numeric visualization folder not found.")
        with viz_tab2:
            imaging_viz_path = os.path.join(ROOT, 'reports', 'imaging', 'visualizations')
            if os.path.exists(imaging_viz_path):
                for name, caption in [
                    ('01_5fold_cross_validation_imaging.png', 'Cross-validation results'),
                    ('08_confusion_matrices_imaging.png', 'Confusion matrices'),
                    ('09_radar_charts_imaging.png', 'Radar chart comparison'),
                ]:
                    path = os.path.join(imaging_viz_path, name)
                    if os.path.exists(path):
                        st.image(path, use_container_width=True, caption=caption)
            else:
                st.info("Imaging visualization folder not found.")


def ask_case_assistant(user_prompt):
    provider, error = get_ai_backend_status()
    if error:
        return error

    if provider == "Groq":
        return ask_case_assistant_with_groq(user_prompt)

    model, error = get_gemini_chat_model()
    if error:
        return error

    case_snapshot = build_case_snapshot()
    prompt = f"""
You are a patient-friendly rheumatoid arthritis support assistant inside a demo clinical app.
Use the patient case context below and answer in very simple language.
Rules:
- Keep the answer practical and easy for non-technical people.
- Do not mention confidence, probabilities, logits, or machine learning internals.
- Give 3 short action points when useful.
- If the user asks about food, care, rest, or doctor follow-up, answer with general safe guidance.
- Always mention this is support information and a doctor should confirm treatment decisions.
- If no patient result is available, say the user should complete the test first.

Patient case context:
{case_snapshot}

User question:
{user_prompt}
"""
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as exc:
        return f"AI helper could not answer right now: {exc}"

def show_auth_page():
    """Display login/signup page."""
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("# 🏥 Clinical RA Assessment System")
        st.markdown("---")
        
        auth_option = st.radio(
            "Choose action:",
            ["Login", "Sign Up"],
            horizontal=True
        )
        
        if auth_option == "Login":
            st.subheader("🔐 Login")
            
            email = st.text_input(
                "Email",
                placeholder="your@email.com",
                key="login_email"
            )
            
            password = st.text_input(
                "Password",
                type="password",
                placeholder="Enter your password",
                key="login_password"
            )
            
            if st.button("Login", use_container_width=True, type="primary"):
                if email and password:
                    success, user_id, user_name, message = db.authenticate_user(email, password)
                    
                    if success:
                        st.session_state.authenticated = True
                        st.session_state.user_id = user_id
                        st.session_state.user_email = email
                        st.session_state.user_name = user_name
                        st.success("✅ Login successful!")
                        st.rerun()
                    else:
                        st.error(f"❌ {message}")
                else:
                    st.warning("⚠️ Please enter email and password")
        
        else:  # Sign Up
            st.subheader("📝 Create Account")
            
            full_name = st.text_input(
                "Full Name",
                placeholder="John Doe",
                key="signup_name"
            )
            
            email = st.text_input(
                "Email",
                placeholder="your@email.com",
                key="signup_email"
            )
            
            password = st.text_input(
                "Password",
                type="password",
                placeholder="At least 6 characters",
                key="signup_password"
            )
            
            password_confirm = st.text_input(
                "Confirm Password",
                type="password",
                placeholder="Re-enter password",
                key="signup_password_confirm"
            )
            
            if st.button("Create Account", use_container_width=True, type="primary"):
                if not full_name:
                    st.warning("⚠️ Please enter your full name")
                elif not email or not password:
                    st.warning("⚠️ Please enter email and password")
                elif password != password_confirm:
                    st.error("❌ Passwords do not match")
                elif len(password) < 6:
                    st.error("❌ Password must be at least 6 characters")
                elif len(full_name) < 2:
                    st.error("❌ Name must be at least 2 characters")
                else:
                    success, message = db.create_user(email, password, full_name)
                    
                    if success:
                        st.success("✅ Account created successfully! Please login.")
                        st.info("Login with your credentials above.")
                    else:
                        st.error(f"❌ {message}")

def show_logout_button():
    """Show logout button in sidebar."""
    with st.sidebar:
        st.markdown("---")
        st.markdown(f"**👤 {st.session_state.user_name or st.session_state.user_email}**")
        st.caption(f"Email: {st.session_state.user_email}")
        
        if st.button("🚪 Logout", use_container_width=True, help="Click to logout"):
            st.session_state.authenticated = False
            st.session_state.user_id = None
            st.session_state.user_email = None
            st.session_state.user_name = None
            st.session_state.lab_result = None
            st.session_state.xray_result = None
            st.success("Logged out successfully")
            st.rerun()

@st.cache_resource
def load_numeric_model():
    try:
        return load_numeric_model_artifacts()
    except Exception as e:
        st.error(f"❌ Error loading ANN model: {str(e)}")
        return None, None

@st.cache_resource
def load_image_model():
    try:
        return load_image_model_artifact()
    except Exception as e:
        st.error(f"❌ Error loading Swin model: {str(e)}")
        return None

# ============================================================================
# MAIN APP - ONLY SHOW IF AUTHENTICATED
# ============================================================================

if not st.session_state.authenticated:
    show_auth_page()
else:
    # Show header with logout option
    st.markdown(
        """
        <div class="topbar-card">
            <h1 class="topbar-title">🏥 Fast RA Check Support</h1>
            <p class="topbar-subtitle">Simple blood test and X-ray support for faster rheumatoid arthritis screening.</p>
            <div class="topbar-highlights">
                <div class="highlight-chip">⚡ Quick screening in one place</div>
                <div class="highlight-chip">🧠 Simple care guidance after result</div>
                <div class="highlight-chip">🩺 Next-step support for patients</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    show_logout_button()
    st.divider()

    # Create tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "🩸 Blood Test Check",
        "🦴 X-Ray Check",
        "📊 Bulk Lab Data",
        "📸 Bulk X-Ray Analysis",
        "🤖 Ask AI",
        "📜 Reports",
        "📋 Prediction History"
    ])

    # ========================================================================
    # TAB 1: NUMERIC (LAB TEST) ASSESSMENT
    # ========================================================================
    with tab1:
        st.subheader("Blood Test Check")
        st.markdown("*Fill in the patient values below. The app will show a simple care level.*")
        
        model, scaler = load_numeric_model()
        if model is None:
            st.error("❌ Numeric model not available")
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                age = st.number_input('Age (years)', min_value=0, max_value=120, value=55, key='lab_age')
                gender = st.selectbox('Gender', ['Female', 'Male'], key='lab_gender')
                rf = st.number_input('Rheumatoid Factor (RF) [IU/mL]', value=25.0, key='lab_rf')
            
            with col2:
                anti_ccp = st.number_input('Anti-CCP [U/mL]', value=15.0, key='lab_anticcp')
                crp = st.number_input('C-Reactive Protein (CRP) [mg/dL]', value=8.0, key='lab_crp')
                esr = st.number_input('Erythrocyte Sedimentation Rate (ESR) [mm/h]', value=30.0, key='lab_esr')
            
            if st.button("🔬 Analyze Lab Results", key='lab_btn', use_container_width=True):
                device = torch.device('cpu')

                raw_features = [
                    float(age),
                    1.0 if gender == 'Female' else 0.0,
                    float(rf),
                    float(anti_ccp),
                    float(crp),
                    float(esr)
                ]

                try:
                    normalized = normalize_numeric_features(raw_features, scaler)
                except Exception as e:
                    st.error(f"❌ Scaling error: {e}")
                    st.stop()

                data = torch.tensor(normalized, dtype=torch.float32).to(device)

                try:
                    with torch.no_grad():
                        logits = model(data.unsqueeze(0))
                        proba = torch.softmax(logits, dim=1)[0].cpu().numpy()
                    prediction = int(np.argmax(proba))
                    confidence = max(proba) * 100
                except Exception as e:
                    st.error(f"❌ Prediction error: {str(e)}")
                    st.stop()
                
                st.divider()
                class_names = ['Healthy', 'Seropositive RA', 'Seronegative RA']
                result_label = class_names[prediction]
                assessment = build_lab_assessment(prediction, confidence)

                st.session_state.lab_result = {
                    'class': prediction,
                    'label': result_label,
                    'confidence': confidence,
                    'input': raw_features,
                    'severity': assessment['severity'],
                    'severity_label': get_severity_theme(assessment['severity'])['badge'],
                    'summary': assessment['summary'],
                }
                
                # Save to history
                if st.session_state.user_id:
                    success, msg = db.save_prediction(
                        user_id=st.session_state.user_id,
                        prediction_type='lab',
                        input_data={'age': age, 'gender': gender, 'rf': rf, 'anti_ccp': anti_ccp, 'crp': crp, 'esr': esr},
                        result_data={'class': int(prediction), 'label': result_label, 'probabilities': proba.tolist()},
                        confidence=float(confidence)
                    )

            if st.session_state.lab_result:
                st.divider()
                render_saved_result("lab")

    # ========================================================================
    # TAB 2: IMAGING (X-RAY) ASSESSMENT
    # ========================================================================
    with tab2:
        st.subheader("X-Ray Check")
        st.markdown("*Upload one hand or wrist X-ray. The app will show a simple damage warning level.*")
        
        img_model = load_image_model()
        if img_model is None:
            st.error("❌ Imaging model not available")
        else:
            uploaded_file = st.file_uploader("📤 Upload X-ray Image", type=['bmp','png','jpg','jpeg'], key='xray_upload')
            
            if uploaded_file is not None:
                try:
                    img = Image.open(uploaded_file)
                except:
                    img_bytes = uploaded_file.read()
                    nparr = np.frombuffer(img_bytes, np.uint8)
                    cv_img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
                    if cv_img is not None:
                        img = Image.fromarray(cv_img)
                    else:
                        st.error("Could not load image")
                        st.stop()
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.image(img, caption="Uploaded X-ray", use_container_width=True)
                
                with col2:
                    if st.button("🔍 Analyze X-Ray", key='xray_btn', use_container_width=True):
                        x_tensor = preprocess_xray_image(img)
                        label, confidence = predict_xray_image(img_model, x_tensor)
                        
                        assessment = build_xray_assessment(label, confidence)

                        st.session_state.xray_result = {
                            'label': label,
                            'confidence': confidence,
                            'severity': assessment['severity'],
                            'severity_label': get_severity_theme(assessment['severity'])['badge'],
                            'summary': assessment['summary'],
                        }

                        # Save to history
                        if st.session_state.user_id:
                            success, msg = db.save_prediction(
                                user_id=st.session_state.user_id,
                                prediction_type='xray',
                                input_data={'filename': uploaded_file.name},
                                result_data={'label': label, 'result': 'Erosive' if label == 'Erosive' else 'Non-Erosive'},
                                confidence=float(confidence * 100)
                            )

                if st.session_state.xray_result:
                    st.divider()
                    render_saved_result("xray")

    # ========================================================================
    # TAB 3: BULK LAB DATA UPLOAD
    # ========================================================================
    with tab3:
        st.subheader("📊 Bulk Lab Data Analysis")
        st.markdown("*Upload CSV with patient lab data for batch processing*")
        
        model, scaler = load_numeric_model()
        if model is None:
            st.error("❌ Numeric model not available")
        else:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.info("**CSV Format Required:**\nColumns: Age, Gender, RF, Anti-CCP, CRP, ESR (in any order)")
            
            with col2:
                # Download template
                template_df = pd.DataFrame({
                    'Age': [45, 56, 62],
                    'Gender': ['Female', 'Male', 'Female'],
                    'RF': [25.0, 45.0, 8.0],
                    'Anti-CCP': [15.0, 120.0, 5.0],
                    'CRP': [8.0, 25.0, 2.0],
                    'ESR': [30.0, 55.0, 10.0]
                })
                
                csv_template = template_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Template CSV",
                    data=csv_template,
                    file_name="lab_data_template.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            st.divider()
            
            uploaded_file = st.file_uploader("📤 Upload CSV File", type=['csv'], key='bulk_lab_upload')
            
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    
                    st.write(f"**Loaded {len(df)} records**")
                    st.dataframe(df.head(), use_container_width=True)
                    
                    if st.button("🔬 Process Batch", key='bulk_lab_btn', use_container_width=True):
                        device = torch.device('cpu')
                        
                        # Normalize column names: handle both hyphen and underscore, case insensitive
                        df_normalized = df.copy()
                        df_normalized.columns = [col.lower().replace('_', '-').replace(' ', '') for col in df_normalized.columns]
                        
                        required_cols = ['age', 'gender', 'rf', 'anti-ccp', 'crp', 'esr']
                        
                        # Check if all required columns exist
                        missing_cols = [col for col in required_cols if col not in df_normalized.columns]
                        
                        if missing_cols:
                            st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
                            st.info(f"Available columns: {', '.join(df_normalized.columns)}")
                        else:
                            predictions_list = []
                            confidence_list = []
                            
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            for idx, row in df_normalized.iterrows():
                                try:
                                    raw_features = [
                                        float(row['age']),
                                        1.0 if str(row['gender']).lower() in ['f', 'female'] else 0.0,
                                        float(row['rf']),
                                        float(row['anti-ccp']),
                                        float(row['crp']),
                                        float(row['esr'])
                                    ]
                                    
                                    if scaler is not None:
                                        try:
                                            normalized = scaler.transform([raw_features])[0]
                                        except:
                                            normalized = np.array(raw_features) / np.array([100, 1, 500, 500, 100, 100])
                                    else:
                                        normalized = np.array(raw_features) / np.array([100, 1, 500, 500, 100, 100])
                                    
                                    data = torch.tensor(normalized, dtype=torch.float32).to(device)
                                    
                                    with torch.no_grad():
                                        logits = model(data.unsqueeze(0))
                                        proba = torch.softmax(logits, dim=1)[0].cpu().numpy()
                                    
                                    prediction = int(np.argmax(proba))
                                    confidence = float(max(proba) * 100)
                                    
                                    class_names = ['Healthy', 'Seropositive RA', 'Seronegative RA']
                                    predictions_list.append(class_names[prediction])
                                    confidence_list.append(confidence)
                                    
                                except Exception as e:
                                    predictions_list.append('Error')
                                    confidence_list.append(0.0)
                                
                                progress = (idx + 1) / len(df_normalized)
                                progress_bar.progress(progress)
                                status_text.text(f"Processing {idx + 1}/{len(df_normalized)}...")
                            
                            progress_bar.empty()
                            status_text.empty()
                            
                            # Add predictions to dataframe
                            severity_list = []
                            for item, conf in zip(predictions_list, confidence_list):
                                if item == 'Healthy':
                                    severity_list.append('Green - Low concern')
                                elif item == 'Seronegative RA':
                                    severity_list.append('Orange - Needs attention')
                                elif item == 'Seropositive RA':
                                    severity_list.append('Red - High concern' if conf >= 80 else 'Orange - Needs attention')
                                else:
                                    severity_list.append('Review needed')

                            df['Prediction'] = predictions_list
                            df['Care Level'] = severity_list
                            
                            st.divider()
                            st.write("### Results")
                            st.dataframe(df, use_container_width=True, hide_index=True)
                            
                            # Statistics
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                healthy_count = predictions_list.count('Healthy')
                                st.metric("Healthy", healthy_count, f"{healthy_count/len(predictions_list)*100:.1f}%")
                            with col2:
                                seropos_count = predictions_list.count('Seropositive RA')
                                st.metric("Seropositive", seropos_count, f"{seropos_count/len(predictions_list)*100:.1f}%")
                            with col3:
                                seroneg_count = predictions_list.count('Seronegative RA')
                                st.metric("Seronegative", seroneg_count, f"{seroneg_count/len(predictions_list)*100:.1f}%")
                            
                            st.divider()
                            
                            # Download options
                            csv_output = df.to_csv(index=False)
                            st.download_button(
                                label="💾 Download Results (CSV)",
                                data=csv_output,
                                file_name=f"lab_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                            
                            # Save to history
                            if st.session_state.user_id:
                                success, msg = db.save_report(
                                    user_id=st.session_state.user_id,
                                    report_title=f"Bulk Lab Analysis - {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                                    report_content="Batch processing completed",
                                    lab_data=json.dumps({'count': len(df), 'file': uploaded_file.name}),
                                    imaging_data=None,
                                    prediction_data=json.dumps({'predictions': predictions_list, 'confidences': confidence_list}),
                                    report_type='bulk_csv'
                                )
                
                except Exception as e:
                    st.error(f"❌ Error processing file: {str(e)}")

    # ========================================================================
    # TAB 4: BULK X-RAY ANALYSIS
    # ========================================================================
    with tab4:
        st.subheader("📸 Bulk X-Ray Analysis")
        st.markdown("*Upload multiple X-ray images for batch erosion detection*")
        
        img_model = load_image_model()
        if img_model is None:
            st.error("❌ Imaging model not available")
        else:
            uploaded_files = st.file_uploader(
                "📤 Upload X-ray Images", 
                type=['bmp','png','jpg','jpeg'],
                accept_multiple_files=True,
                key='bulk_xray_upload'
            )
            
            if uploaded_files:
                st.write(f"**Loaded {len(uploaded_files)} images**")
                
                if st.button("🔍 Analyze Batch", key='bulk_xray_btn', use_container_width=True):
                    results_list = []
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for idx, uploaded_file in enumerate(uploaded_files):
                        try:
                            img = Image.open(uploaded_file)
                            x_tensor = preprocess_xray_image(img)
                            label, confidence = predict_xray_image(img_model, x_tensor)
                            care_level = 'Red - High concern' if label == 'Erosive' and confidence >= 0.75 else (
                                'Orange - Needs attention' if label == 'Erosive' else 'Green - Low concern'
                            )
                            
                            results_list.append({
                                'Filename': uploaded_file.name,
                                'Result': label,
                                'Care Level': care_level
                            })
                        except Exception as e:
                            results_list.append({
                                'Filename': uploaded_file.name,
                                'Result': 'Error',
                                'Care Level': 'Review needed'
                            })
                        
                        progress = (idx + 1) / len(uploaded_files)
                        progress_bar.progress(progress)
                        status_text.text(f"Processing {idx + 1}/{len(uploaded_files)}...")
                    
                    progress_bar.empty()
                    status_text.empty()
                    
                    # Display results
                    st.divider()
                    st.write("### Results")
                    
                    results_df = pd.DataFrame(results_list)
                    st.dataframe(results_df, use_container_width=True, hide_index=True)
                    
                    # Statistics
                    erosive_count = sum(1 for r in results_list if r['Result'] == 'Erosive')
                    non_erosive_count = sum(1 for r in results_list if r['Result'] == 'Non-Erosive')
                    error_count = sum(1 for r in results_list if r['Result'] == 'Error')
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Erosive", erosive_count, f"{erosive_count/len(results_list)*100:.1f}%")
                    with col2:
                        st.metric("Non-Erosive", non_erosive_count, f"{non_erosive_count/len(results_list)*100:.1f}%")
                    with col3:
                        st.metric("Errors", error_count)
                    
                    st.divider()
                    
                    # Download options
                    csv_output = results_df.to_csv(index=False)
                    st.download_button(
                        label="💾 Download Results (CSV)",
                        data=csv_output,
                        file_name=f"xray_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                    
                    # Save to history
                    if st.session_state.user_id:
                        success, msg = db.save_report(
                            user_id=st.session_state.user_id,
                            report_title=f"Bulk X-Ray Analysis - {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                            report_content="Batch image processing completed",
                            lab_data=None,
                            imaging_data=json.dumps({'count': len(uploaded_files), 'files': [f.name for f in uploaded_files]}),
                            prediction_data=json.dumps(results_list),
                            report_type='bulk_image'
                        )

    # ========================================================================
    # TAB 5: AI CHAT SUPPORT
    # ========================================================================
    with tab5:
        st.subheader("Ask AI")
        st.markdown("*Ask simple questions about food, care, rest, daily routine, and what to do next.*")

        with st.expander("AI setup", expanded=False):
            st.markdown(
                """
                **Groq setup**

                1. Open [Groq Console](https://console.groq.com/keys).
                2. Create or copy your API key.
                3. Paste it in the `Groq API key` field below, or start the app with:

                ```bash
                export GROQ_API_KEY="paste-your-groq-key-here"
                streamlit run src/app/app_auth.py
                ```

                **Gemini setup**

                1. Open [Google AI Studio](https://aistudio.google.com/app/apikey).
                2. Create an API key.
                3. Paste it below, or start the app with:

                ```bash
                export GOOGLE_API_KEY="paste-your-key-here"
                streamlit run src/app/app_auth.py
                ```

                Best practice: keep keys in environment variables, not inside code files.
                """
            )

        provider_col, model_col = st.columns(2)
        with provider_col:
            st.selectbox(
                "AI provider",
                ["Auto", "Groq", "Gemini"],
                key="ai_provider",
                help="Auto prefers Groq first, then Gemini.",
            )
        with model_col:
            st.selectbox(
                "Groq model",
                ["llama-3.3-70b-versatile", "openai/gpt-oss-20b", "llama-3.1-8b-instant"],
                key="ai_model_choice",
                help="Used when Groq is selected or available in Auto mode.",
            )

        st.text_input(
            "Groq API key",
            type="password",
            key="groq_api_key_input",
            placeholder="Paste Groq API key here if you did not export GROQ_API_KEY",
            help="This stays only in the current app session.",
        )

        st.text_input(
            "Gemini API key",
            type="password",
            key="gemini_api_key_input",
            placeholder="Paste Gemini API key here if you did not export GOOGLE_API_KEY",
            help="This stays only in the current app session.",
        )

        provider_name, provider_error = get_ai_backend_status()
        if provider_error:
            st.warning(provider_error)
        else:
            st.success(f"{provider_name} is connected and ready.")

        st.markdown(
            f"""
            <div class="plain-card">
                <h4>Current case context</h4>
                <div class="mini-note">{build_case_snapshot()}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        suggestion_cols = st.columns(3)
        selected_prompt = None
        for idx, suggestion in enumerate(get_ai_suggestions()):
            with suggestion_cols[idx]:
                if st.button(suggestion, key=f"ai_suggestion_{idx}", use_container_width=True):
                    selected_prompt = suggestion

        for message in st.session_state.chat_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        prompt = st.chat_input("Ask about food, exercise, care, pain, or next doctor steps...")
        prompt = prompt or selected_prompt or st.session_state.pop('prefill_ai_prompt', None)

        if prompt:
            st.session_state.chat_messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    answer = ask_case_assistant(prompt)
                st.markdown(answer)
            st.session_state.chat_messages.append({"role": "assistant", "content": answer})

    # ========================================================================
    # TAB 6: REPORTS & COMBINED ANALYSIS
    # ========================================================================
    reports_tab = tab6
    with reports_tab:
        st.subheader("📋 Clinical Reports & Combined Analysis")
        st.markdown("*Generate combined reports from lab and imaging predictions, or view saved reports*")
        
        # Create subtabs for reports and combined analysis
        report_view_tab, single_report_tab, combined_report_tab = st.tabs(["📂 Saved Reports", "🧾 Single Result Report", "📊 Generate Combined Report"])
        
        # ====================================================================
        # SAVED REPORTS TAB
        # ====================================================================
        with report_view_tab:
            # Get user reports
            reports = db.get_user_reports(st.session_state.user_id)
            
            if reports:
                st.write(f"### Total Reports: {len(reports)}")
                
                # Filter options
                report_types = ["All"] + list(set([r['report_type'] for r in reports]))
                selected_type = st.selectbox("Filter by type:", report_types, key="report_filter")
                
                filtered_reports = reports
                if selected_type != "All":
                    filtered_reports = [r for r in reports if r['report_type'] == selected_type]
                
                st.divider()
                
                # Display reports with Chrome-style history grouping
                for report in filtered_reports:
                    report_date = datetime.fromisoformat(report['created_at'].replace('Z', '+00:00')) if isinstance(report['created_at'], str) else report['created_at']
                    
                    with st.expander(
                        f"**{report['report_title']}** | {report_date.strftime('%Y-%m-%d %H:%M')} | {report['report_type'].replace('_', ' ').title()}"
                    ):
                        col1, col2 = st.columns([4, 1])
                        
                        with col1:
                            st.write(report['report_content'])
                            
                            if report['lab_data']:
                                st.write("**Lab Data:**")
                                st.json(json.loads(report['lab_data']) if isinstance(report['lab_data'], str) else report['lab_data'])
                            
                            if report['imaging_data']:
                                st.write("**Imaging Data:**")
                                st.json(json.loads(report['imaging_data']) if isinstance(report['imaging_data'], str) else report['imaging_data'])
                            
                            if report['prediction_data']:
                                st.write("**Predictions:**")
                                st.json(json.loads(report['prediction_data']) if isinstance(report['prediction_data'], str) else report['prediction_data'])
                        
                        with col2:
                            # Download as PDF (HTML)
                            html_content = f"""
                            <!DOCTYPE html>
                            <html>
                            <head>
                                <title>{report['report_title']}</title>
                                <style>
                                    body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
                                    .container {{ max-width: 900px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                                    .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 5px; text-align: center; }}
                                    .header h1 {{ margin: 0; font-size: 28px; }}
                                    .section {{ margin: 25px 0; padding: 20px; border-left: 4px solid #667eea; background: #f9f9f9; border-radius: 3px; }}
                                    .section h2 {{ color: #667eea; margin-top: 0; }}
                                    .section h3 {{ color: #333; }}
                                    .timestamp {{ color: #888; font-size: 0.9em; margin-top: 10px; }}
                                    .metrics {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 15px; margin: 15px 0; }}
                                    .metric-box {{ background: #f0f0f0; padding: 15px; border-radius: 5px; text-align: center; border: 1px solid #ddd; }}
                                    .metric-box strong {{ color: #667eea; font-size: 18px; display: block; }}
                                    .metric-box span {{ color: #666; font-size: 14px; }}
                                    table {{ width: 100%; border-collapse: collapse; margin: 15px 0; background: white; }}
                                    th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                                    th {{ background: #667eea; color: white; font-weight: bold; }}
                                    tr:nth-child(even) {{ background: #f9f9f9; }}
                                    .diagnosis {{ font-size: 18px; font-weight: bold; color: #333; padding: 10px; background: #f0f0f0; border-radius: 5px; margin: 10px 0; }}
                                    .confidence {{ color: #667eea; font-weight: bold; }}
                                    .footer {{ margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; color: #888; font-size: 12px; text-align: center; }}
                                </style>
                            </head>
                            <body>
                                <div class="container">
                                    <div class="header">
                                        <h1>🏥 Clinical RA Assessment Report</h1>
                                        <p class="timestamp">{report_date.strftime('%B %d, %Y at %H:%M')}</p>
                                    </div>
                                    <div class="section">
                                        <h2>Report Information</h2>
                                        <p><strong>Title:</strong> {report['report_title']}</p>
                                        <p><strong>Type:</strong> {report['report_type'].replace('_', ' ').title()}</p>
                                        <p>{report['report_content']}</p>
                                    </div>
            """
                            
                            if report['lab_data']:
                                lab_data = json.loads(report['lab_data']) if isinstance(report['lab_data'], str) else report['lab_data']
                                html_content += f"""
                                    <div class="section">
                                        <h2>📋 Lab Assessment Data</h2>
                                        <div class="metrics">
                                            <div class="metric-box"><strong>{lab_data.get('age', 'N/A')}</strong><span>Age (years)</span></div>
                                            <div class="metric-box"><strong>{lab_data.get('gender', 'N/A')}</strong><span>Gender</span></div>
                                            <div class="metric-box"><strong>{lab_data.get('rf', 'N/A')}</strong><span>RF (IU/mL)</span></div>
                                            <div class="metric-box"><strong>{lab_data.get('anti_ccp', 'N/A')}</strong><span>Anti-CCP (U/mL)</span></div>
                                            <div class="metric-box"><strong>{lab_data.get('crp', 'N/A')}</strong><span>CRP (mg/dL)</span></div>
                                            <div class="metric-box"><strong>{lab_data.get('esr', 'N/A')}</strong><span>ESR (mm/h)</span></div>
                                        </div>
                                    </div>
                """
                            
                            if report['imaging_data']:
                                imaging_data = json.loads(report['imaging_data']) if isinstance(report['imaging_data'], str) else report['imaging_data']
                                html_content += f"""
                                    <div class="section">
                                        <h2>📷 Imaging Assessment Data</h2>
                                        <div class="metrics">
                                            <div class="metric-box"><strong>{imaging_data.get('filename', 'N/A')}</strong><span>Image File</span></div>
                                        </div>
                                    </div>
                """
                            
                            if report['prediction_data']:
                                pred_data = json.loads(report['prediction_data']) if isinstance(report['prediction_data'], str) else report['prediction_data']
                                
                                # Handle both single prediction and combined predictions
                                if isinstance(pred_data, dict) and 'lab_result' in pred_data:
                                    # Combined prediction
                                    lab_result = pred_data.get('lab_result', {})
                                    imaging_result = pred_data.get('imaging_result', {})
                                    combined_result = pred_data.get('combined_result', {})
                                    
                                    html_content += f"""
                                        <div class="section">
                                            <h2>📊 Prediction Results</h2>
                                            <h3>Lab Assessment Result</h3>
                                            <div class="diagnosis">{lab_result.get('label', 'N/A')}</div>
                                            <p><span class="confidence">Care Level: {lab_result.get('severity_label', 'Recorded')}</span></p>
                                            
                                            <h3>Imaging Assessment Result</h3>
                                            <div class="diagnosis">{imaging_result.get('label', 'N/A')}</div>
                                            <p><span class="confidence">Care Level: {imaging_result.get('severity_label', 'Recorded')}</span></p>
                                            
                                            <h3>Combined Clinical Assessment</h3>
                                            <div class="diagnosis">{combined_result.get('final_diagnosis', 'N/A')}</div>
                                            <p>{combined_result.get('clinical_summary', 'N/A')}</p>
                                        </div>
                """
                                else:
                                    # Single prediction or list of predictions
                                    html_content += f"""
                                        <div class="section">
                                            <h2>📊 Prediction Results</h2>
                                            <table>
                                                <tr>
                                                    <th>Metric</th>
                                                    <th>Value</th>
                                                </tr>
                """
                                    if isinstance(pred_data, dict):
                                        for key, value in pred_data.items():
                                            html_content += f"<tr><td>{key}</td><td>{value}</td></tr>"
                                    elif isinstance(pred_data, list):
                                        for idx, item in enumerate(pred_data, 1):
                                            html_content += f"<tr><td>Prediction {idx}</td><td>{item}</td></tr>"
                                    html_content += "</table>"
                            
                            html_content += f"""
                                    <div class="footer">
                                        <p>This report was generated by the Clinical RA Assessment System</p>
                                        <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                                        <p>For research and clinical support use only</p>
                                    </div>
                                </div>
                            </body>
                            </html>
                            """
                            
                            st.download_button(
                                label="📥 Download PDF",
                                data=html_content,
                                file_name=f"{report['report_title'].replace(' ', '_')}.html",
                                mime="text/html",
                                use_container_width=True,
                                key=f"download_{report['id']}"
                            )
                            
                            # Delete button
                            if st.button("🗑️ Delete", key=f"delete_{report['id']}", use_container_width=True):
                                success, msg = db.delete_report(report['id'], st.session_state.user_id)
                                if success:
                                    st.success("Report deleted!")
                                    st.rerun()
                                else:
                                    st.error(f"Error: {msg}")
            else:
                st.info("📭 No reports yet. Complete analysis to generate reports!")
        
        # ====================================================================
        # SINGLE REPORT GENERATION TAB
        # ====================================================================
        with single_report_tab:
            st.write("### Generate Single Result Report")
            patient_name = st.text_input("Patient Name", key="single_report_name", placeholder="Enter patient name")
            patient_age = st.text_input("Patient Age", key="single_report_age", placeholder="Auto or manual")

            if st.session_state.lab_result or st.session_state.xray_result:
                if st.session_state.lab_result:
                    st.write("Blood Test Result Available")
                    st.metric("Blood Test", st.session_state.lab_result['label'], st.session_state.lab_result.get('severity_label', 'Care level'))
                if st.session_state.xray_result:
                    st.write("X-Ray Result Available")
                    st.metric("X-Ray", st.session_state.xray_result['label'], st.session_state.xray_result.get('severity_label', 'Care level'))

                if st.button("🧾 Create Single Report", use_container_width=True, key="single_report_btn"):
                    report_body = []
                    if st.session_state.lab_result:
                        report_body.append(
                            f"<h3>Blood Test Result</h3><p><strong>Diagnosis:</strong> {st.session_state.lab_result['label']}</p><p><strong>Care Level:</strong> {st.session_state.lab_result.get('severity_label', 'Recorded')}</p><p>{st.session_state.lab_result.get('summary', '')}</p>"
                        )
                    if st.session_state.xray_result:
                        report_body.append(
                            f"<h3>X-Ray Result</h3><p><strong>Finding:</strong> {st.session_state.xray_result['label']}</p><p><strong>Care Level:</strong> {st.session_state.xray_result.get('severity_label', 'Recorded')}</p><p>{st.session_state.xray_result.get('summary', '')}</p>"
                        )

                    html_report = f"""
                    <!DOCTYPE html>
                    <html>
                    <head>
                        <title>RA Patient Report</title>
                        <style>
                            body {{ font-family: Arial, sans-serif; background: #f6f8fb; padding: 20px; }}
                            .card {{ max-width: 860px; margin: 0 auto; background: white; padding: 32px; border-radius: 16px; border: 1px solid #d7e6ed; }}
                            .header {{ background: #0f5c7a; color: white; padding: 24px; border-radius: 12px; }}
                            .section {{ margin-top: 22px; padding: 20px; background: #f9fbfd; border-left: 4px solid #0d8a7c; border-radius: 8px; }}
                        </style>
                    </head>
                    <body>
                        <div class="card">
                            <div class="header">
                                <h1>RA Patient Report</h1>
                                <p>Date: {datetime.now().strftime('%d %B %Y %H:%M')}</p>
                            </div>
                            <div class="section">
                                <h3>Patient Details</h3>
                                <p><strong>Name:</strong> {patient_name or 'Not entered'}</p>
                                <p><strong>Age:</strong> {patient_age or 'Not entered'}</p>
                            </div>
                            <div class="section">
                                {''.join(report_body)}
                            </div>
                            <div class="section">
                                <h3>Important Note</h3>
                                <p>This report is for quick screening support. A doctor should confirm the final diagnosis and treatment plan.</p>
                            </div>
                        </div>
                    </body>
                    </html>
                    """

                    st.download_button(
                        label="📥 Download Printable Report",
                        data=html_report,
                        file_name=f"ra_single_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                        mime="text/html",
                        use_container_width=True,
                    )
            else:
                st.info("Run at least one test first to generate a report.")

        # ====================================================================
        # COMBINED REPORT GENERATION TAB
        # ====================================================================
        with combined_report_tab:
            st.write("### Generate Combined Lab + Imaging Report")
            st.markdown("*Create a comprehensive clinical assessment combining both lab test and X-ray results*")
            
            # Check if we have recent predictions
            lab_result = getattr(st.session_state, 'lab_result', None)
            xray_result = getattr(st.session_state, 'xray_result', None)
            
            if lab_result is not None and xray_result is not None:
                st.success("✅ Both lab and imaging results available")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Lab Assessment Result:**")
                    st.metric("Blood Test", lab_result['label'], lab_result.get('severity_label', 'Care level'))
                
                with col2:
                    st.write("**Imaging Assessment Result:**")
                    st.metric("X-Ray", xray_result['label'], xray_result.get('severity_label', 'Care level'))
                
                st.divider()
                
                # Generate combined assessment
                if st.button("📋 Generate Combined Report", use_container_width=True):
                    # Determine combined diagnosis
                    if lab_result['class'] == 0:  # Healthy
                        if xray_result['label'] == 'Non-Erosive':
                            final_diagnosis = "No RA Evidence - Healthy"
                            clinical_summary = "Patient shows no RA indicators on both lab tests and imaging. No intervention needed."
                        else:
                            final_diagnosis = "Lab-Imaging Discordance - Requires Further Investigation"
                            clinical_summary = "Lab tests appear healthy but imaging shows erosions. Recommend rheumatology review."
                    else:  # RA suspected
                        if xray_result['label'] == 'Erosive':
                            final_diagnosis = "RA with Bone Damage - Advanced Stage"
                            clinical_summary = "Both lab tests and imaging confirm RA with erosive changes. Aggressive treatment required."
                        else:
                            final_diagnosis = "RA Suspected - Early/Well-Controlled Stage"
                            clinical_summary = "Lab tests indicate RA but no erosions detected yet. Early intervention recommended."
                    
                    # Create comprehensive report
                    html_report = f"""
                    <!DOCTYPE html>
                    <html>
                    <head>
                        <title>Combined RA Assessment Report</title>
                        <style>
                            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }}
                            .container {{ max-width: 900px; margin: 0 auto; background: white; padding: 40px; border-radius: 10px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }}
                            .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 40px; border-radius: 8px; text-align: center; margin-bottom: 30px; }}
                            .header h1 {{ margin: 0 0 10px 0; font-size: 32px; }}
                            .header p {{ margin: 5px 0; opacity: 0.9; }}
                            .two-column {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0; }}
                            .result-box {{ background: #f9f9f9; padding: 20px; border-left: 4px solid #667eea; border-radius: 5px; }}
                            .result-box h3 {{ color: #667eea; margin-top: 0; }}
                            .diagnosis-box {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 25px; border-radius: 8px; text-align: center; margin: 20px 0; }}
                            .diagnosis-box h2 {{ margin: 0; font-size: 28px; }}
                            .section {{ margin: 30px 0; padding: 25px; border-left: 4px solid #667eea; background: #f9f9f9; border-radius: 5px; }}
                            .section h2 {{ color: #667eea; margin-top: 0; }}
                            .metrics {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 15px; margin: 15px 0; }}
                            .metric-item {{ background: white; padding: 15px; border-radius: 5px; text-align: center; border: 1px solid #ddd; }}
                            .metric-label {{ color: #666; font-size: 12px; text-transform: uppercase; }}
                            .metric-value {{ color: #667eea; font-size: 20px; font-weight: bold; margin-top: 5px; }}
                            table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
                            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                            th {{ background: #667eea; color: white; font-weight: bold; }}
                            tr:nth-child(even) {{ background: #f9f9f9; }}
                            .recommendations {{ background: #e8f4f8; padding: 20px; border-left: 4px solid #0277bd; border-radius: 5px; margin: 20px 0; }}
                            .recommendations h3 {{ color: #0277bd; margin-top: 0; }}
                            .recommendations ul {{ margin: 10px 0; padding-left: 20px; }}
                            .recommendations li {{ margin: 8px 0; }}
                            .footer {{ margin-top: 40px; padding-top: 20px; border-top: 2px solid #ddd; color: #888; font-size: 12px; text-align: center; }}
                        </style>
                    </head>
                    <body>
                        <div class="container">
                            <div class="header">
                                <h1>🏥 Clinical RA Assessment Report</h1>
                                <p>Comprehensive Combined Analysis</p>
                                <p>{datetime.now().strftime('%B %d, %Y at %H:%M')}</p>
                            </div>
                            
                            <div class="two-column">
                                <div class="result-box">
                                    <h3>📋 Lab Assessment</h3>
                                    <div style="font-size: 20px; font-weight: bold; color: #667eea; margin: 10px 0;">{lab_result['label']}</div>
                                    <p><strong>Care Level:</strong> {lab_result.get('severity_label', 'Recorded')}</p>
                                    <p><strong>Model:</strong> ANN (3-class)</p>
                                    <p><strong>Accuracy:</strong> 91.26% (±0.22%)</p>
                                </div>
                                <div class="result-box">
                                    <h3>📷 Imaging Assessment</h3>
                                    <div style="font-size: 20px; font-weight: bold; color: #667eea; margin: 10px 0;">{xray_result['label']}</div>
                                    <p><strong>Care Level:</strong> {xray_result.get('severity_label', 'Recorded')}</p>
                                    <p><strong>Model:</strong> Swin Transformer</p>
                                    <p><strong>Accuracy:</strong> 85.83% (±1.78%)</p>
                                </div>
                            </div>
                            
                            <div class="diagnosis-box">
                                <h2>{final_diagnosis}</h2>
                            </div>
                            
                            <div class="section">
                                <h2>Clinical Summary</h2>
                                <p>{clinical_summary}</p>
                            </div>
                            
                            <div class="recommendations">
                                <h3>🏥 Clinical Recommendations</h3>
                                <ul>
                                    <li>Patient should be evaluated by a rheumatologist</li>
                                    <li>Consider additional imaging if not recently done</li>
                                    <li>Monitor disease progression with follow-up assessments</li>
                                    <li>Discuss treatment options based on disease stage</li>
                                </ul>
                            </div>
                            
                            <div class="footer">
                                <p>This report is generated by the Clinical RA Assessment System for research and clinical support use only</p>
                                <p>Not a substitute for professional medical evaluation</p>
                                <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                            </div>
                        </div>
                    </body>
                    </html>
                    """
                    
                    # Save to database
                    if st.session_state.user_id:
                        success, msg = db.save_report(
                            user_id=st.session_state.user_id,
                            report_title=f"Combined RA Assessment - {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                            report_content=f"Combined analysis: {final_diagnosis}",
                            lab_data=json.dumps({
                                'label': lab_result['label'],
                                'confidence': float(lab_result['confidence']),
                                'class': int(lab_result['class'])
                            }),
                            imaging_data=json.dumps({
                                'type': 'xray',
                                'result': xray_result['label'],
                                'confidence': float(xray_result['confidence'])
                            }),
                            prediction_data=json.dumps({
                                'lab_result': {
                                    'label': lab_result['label'],
                                    'confidence': float(lab_result['confidence']),
                                    'class': int(lab_result['class']),
                                    'severity_label': lab_result.get('severity_label', 'Recorded')
                                },
                                'imaging_result': {
                                    'label': xray_result['label'],
                                    'confidence': float(xray_result['confidence']),
                                    'severity_label': xray_result.get('severity_label', 'Recorded')
                                },
                                'combined_result': {
                                    'final_diagnosis': final_diagnosis,
                                    'clinical_summary': clinical_summary
                                }
                            }),
                            report_type='combined'
                        )
                        
                        if success:
                            st.success("✅ Report saved and ready for download")
                            
                            st.download_button(
                                label="📥 Download Combined Report (PDF/HTML)",
                                data=html_report,
                                file_name=f"combined_ra_assessment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                                mime="text/html",
                                use_container_width=True,
                                key="download_combined"
                            )
                        else:
                            st.error(f"Could not save report: {msg}")
            else:
                st.warning("⚠️ Both lab and imaging results are needed to generate a combined report")
                st.info("Steps:\n1. Complete Lab Assessment in Tab 1\n2. Complete X-Ray Analysis in Tab 2\n3. Return here to generate combined report")

    # ========================================================================
    # TAB 7: PREDICTION HISTORY
    # ========================================================================
    history_tab = tab7
    with history_tab:
        st.subheader("📜 Your Prediction History")
        
        # Get user stats
        stats = db.get_user_stats(st.session_state.user_id)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Predictions", stats['total_predictions'])
        with col2:
            st.metric("Lab Tests", stats['by_type'].get('lab', 0))
        with col3:
            st.metric("X-Ray Analysis", stats['by_type'].get('xray', 0))
        
        st.divider()
        
        # Filter options
        filter_type = st.selectbox(
            "Filter by type:",
            ["All", "Lab Assessment", "X-Ray Analysis"],
            key="history_filter"
        )
        
        type_map = {"All": None, "Lab Assessment": "lab", "X-Ray Analysis": "xray"}
        selected_type = type_map[filter_type]
        
        # Get history
        history = db.get_user_history(st.session_state.user_id, selected_type)
        
        if history:
            st.write(f"### Showing {len(history)} predictions")
            
            # Display as expandable items
            for pred in history:
                with st.expander(
                    f"**{pred['prediction_type'].upper()}** - {pred['created_at']} | Care level recorded"
                ):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Input Data:**")
                        st.json(pred['input_data'])
                    
                    with col2:
                        st.write("**Results:**")
                        st.json(pred['result_data'])
            
            st.divider()
            
            # Export options
            st.write("### 📥 Export Data")
            
            if st.button("📊 Export as CSV", use_container_width=True):
                # Convert history to DataFrame
                export_data = []
                for pred in history:
                    export_data.append({
                        'Date': pred['created_at'],
                        'Type': pred['prediction_type'],
                        'Input': json.dumps(pred['input_data']),
                        'Result': json.dumps(pred['result_data']),
                        'Confidence': pred['confidence']
                    })
                
                df = pd.DataFrame(export_data)
                csv = df.to_csv(index=False)
                
                st.download_button(
                    label="💾 Download CSV",
                    data=csv,
                    file_name=f"prediction_history_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            if st.button("📄 Export as JSON", use_container_width=True):
                json_str = json.dumps(history, indent=2, default=str)
                
                st.download_button(
                    label="💾 Download JSON",
                    data=json_str,
                    file_name=f"prediction_history_{datetime.now().strftime('%Y%m%d')}.json",
                    mime="application/json",
                    use_container_width=True
                )
            
            st.divider()
            
            # Clear history option
            col1, col2, col3 = st.columns([1, 1, 2])
            with col1:
                if st.button("🗑️ Clear All History", use_container_width=True):
                    if st.session_state.get('confirm_delete'):
                        success, msg = db.clear_user_history(st.session_state.user_id)
                        if success:
                            st.success("✅ History cleared!")
                            st.rerun()
                        else:
                            st.error(f"❌ {msg}")
                    else:
                        st.session_state.confirm_delete = True
                        st.warning("⚠️ Click again to confirm deletion")
        
        else:
            st.info("📭 No predictions yet. Start by analyzing lab tests or X-rays!")

    st.divider()
    render_about_footer()
    st.caption("Made for simple rheumatoid arthritis screening support.")
