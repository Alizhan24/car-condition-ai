import io
import json
import time
import base64
import requests
import numpy as np
from typing import Dict, Any, List, Tuple
from PIL import Image, ImageDraw, ImageFont
import streamlit as st

WORKFLOW_URL = ("https://serverless.roboflow.com/infer/workflows/alizhan-lmqza/detect-and-classify") 
API_KEY = ("T1JLVY4nTf5JflZpUoAK")

APP_TITLE = "Car Condition AI — Cleanliness + Damage"
APP_TAGLINE = "Классификация чистоты и детекция вмятин/царапин на фото авто"

# ====================================

def _build_workflow_url() -> str:
    # можно оставить ключ в URL, если так удобнее
    if "api_key=" in WORKFLOW_URL:
        return WORKFLOW_URL
    return WORKFLOW_URL  # ключ пойдёт в заголовке

def _auth_headers() -> Dict[str, str]:
    h = {"Content-Type": "application/json"}
    if API_KEY:
        h["Authorization"] = f"Bearer {API_KEY}"
    return h

def post_image_to_workflow(image_bytes: bytes, conf_det: float, iou_det: float):
    url = WORKFLOW_URL
    if not url:
        raise RuntimeError("WORKFLOW_URL пуст")

    payload = {
        "api_key": API_KEY,   # 👈 Ключ теперь в теле JSON
        "inputs": {
            "image": {
                "type": "base64",
                "value": base64.b64encode(image_bytes).decode("utf-8"),
            },
            "input_parameters": {
                "confidence": conf_det,
                "iou": iou_det,
            },
        },
    }

    r = requests.post(url, headers={"Content-Type": "application/json"}, json=payload, timeout=60)
    try:
        r.raise_for_status()
    except Exception:
        st.error("Запрос отклонён. Сырой ответ сервера:")
        st.code(r.text[:800], language="json")
        raise
    return r.json()



def pil_from_bytes(b: bytes) -> Image.Image:
    return Image.open(io.BytesIO(b)).convert("RGB")

def draw_detections(
    img: Image.Image,
    detections: List[Dict[str, Any]],
    color_map: Dict[str, Tuple[int, int, int]] = None,
) -> Image.Image:
    """
    Рисуем боксы из damage-детектора.
    Ожидается список элементов с ключами: x, y, width, height, class, confidence.
    Координаты — как в твоём JSON: центр-бокс (Roboflow формат).
    """
    if color_map is None:
        color_map = {"dent": (255, 77, 77), "scratch": (255, 185, 55), "rust": (90, 197, 250)}

    im = img.copy()
    draw = ImageDraw.Draw(im)

    # Шрифт (опционально): без внешних файлов используем базовый
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()

    W, H = im.size
    for det in detections:
        cls = det.get("class", "damage")
        conf = float(det.get("confidence", 0.0))
        w = float(det.get("width", 0.0))
        h = float(det.get("height", 0.0))
        cx = float(det.get("x", 0.0))
        cy = float(det.get("y", 0.0))

        # Roboflow bbox (cx,cy,w,h) -> (x1,y1,x2,y2)
        x1 = int(max(0, cx - w / 2))
        y1 = int(max(0, cy - h / 2))
        x2 = int(min(W - 1, cx + w / 2))
        y2 = int(min(H - 1, cy + h / 2))

        color = color_map.get(cls, (120, 220, 120))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label = f"{cls} {conf:.2f}"
        tw, th = draw.textlength(label, font=font), 16
        # фон под текст
        draw.rectangle([x1, y1 - th - 4, x1 + tw + 8, y1], fill=color)
        draw.text((x1 + 4, y1 - th - 2), label, fill=(0, 0, 0), font=font)

    return im

def _is_det_item(p: dict) -> bool:
    return all(k in p for k in ("x", "y", "width", "height"))

def _extract_detections(obj) -> list:
    # ищем predictions, которые похожи на детекции (есть x,y,w,h)
    if isinstance(obj, dict):
        preds = obj.get("predictions")
        if isinstance(preds, list) and preds and isinstance(preds[0], dict) and _is_det_item(preds[0]):
            return preds
        for v in obj.values():
            r = _extract_detections(v)
            if r:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = _extract_detections(v)
            if r:
                return r
    return []

def _extract_classification(obj) -> list:
    # 1) пробуем явный ключ из воркфлоу
    if isinstance(obj, dict) and "classification_prediction" in obj:
        return obj["classification_prediction"].get("predictions", []) or []

    # 2) общий поиск списка {class, confidence} БЕЗ координат
    if isinstance(obj, dict):
        preds = obj.get("predictions")
        if isinstance(preds, list) and preds and isinstance(preds[0], dict):
            p0 = preds[0]
            if "class" in p0 and "confidence" in p0 and not _is_det_item(p0):
                return preds
        for v in obj.values():
            r = _extract_classification(v)
            if r:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = _extract_classification(v)
            if r:
                return r
    return []

def parse_workflow(json_resp: Dict[str, Any]) -> Dict[str, Any]:
    # на всякий
    if isinstance(json_resp, list) and json_resp:
        json_resp = json_resp[0]

    dets = _extract_detections(json_resp) or []
    cls  = _extract_classification(json_resp) or []

    cleanliness_top = cleanliness_conf = None
    if cls:
        best = max(cls, key=lambda x: x.get("confidence", 0.0))
        cleanliness_top = best.get("class")
        cleanliness_conf = float(best.get("confidence", 0.0))

    return {
        "damage_predictions": dets,
        "cleanliness_top": cleanliness_top,
        "cleanliness_conf": cleanliness_conf,
    }

def filter_detections(
    dets: list,
    img_size: Tuple[int, int],
    min_conf: float = 0.45,
    min_rel_area: float = 0.006,  # 0.6% кадра — отсекаем мелкие «царапки-шумы»
) -> list:
    W, H = img_size
    area_total = max(1, W * H)
    out = []
    for d in dets:
        if float(d.get("confidence", 0.0)) < min_conf:
            continue
        area = float(d.get("width", 0.0)) * float(d.get("height", 0.0))
        if area / area_total < min_rel_area:
            continue
        out.append(d)
    return out



def severity_score(detections: List[Dict[str, Any]], img_size: Tuple[int, int]) -> float:
    """
    Простая эвристика "насколько всё плохо" = суммарная площадь боксов / площадь кадра.
    """
    if not detections:
        return 0.0
    W, H = img_size
    area_total = W * H
    s = 0.0
    for d in detections:
        s += float(d.get("width", 0.0)) * float(d.get("height", 0.0))
    return min(1.0, s / max(1.0, area_total))

def badge(text: str, color: str = "accent"):
    # color: accent | good | warn | bad
    st.markdown(f'<span class="neo-chip {color}">{text}</span>', unsafe_allow_html=True)

def neo_section(title: str = None):
    class _Ctx:
        def __enter__(self):
            st.markdown('<div class="neo-card">', unsafe_allow_html=True)
            if title:
                st.markdown(f"<h3>{title}</h3>", unsafe_allow_html=True)
            return self
        def __exit__(self, *exc):
            st.markdown('</div>', unsafe_allow_html=True)
    return _Ctx()

def neo_progress(pct: float, label: str = ""):
    pct = max(0.0, min(1.0, float(pct)))
    st.markdown(f'''
        <div class="neo-progress"><span style="width:{pct*100:.1f}%"></span></div>
        <div class="small-muted">{label}</div>
    ''', unsafe_allow_html=True)


# ============== UI ==============
st.set_page_config(page_title=APP_TITLE, page_icon="🚗", layout="wide")
# === Global Neumorphism CSS ===
st.markdown("""
<style>
:root{
  --bg: #0b1020;
  --surface: #111831;
  --surface-2: #0f162d;
  --text: #e5e7eb;
  --muted: #9aa4b2;
  --accent: #7c3aed;
  --good: #22c55e;
  --warn: #f59e0b;
  --bad: #ef4444;
  --shadow-dark: rgba(2,4,12,0.9);
  --shadow-light: rgba(35,45,80,0.45);
  --radius: 18px;
}

html, body, [data-testid="stAppViewContainer"], [data-testid="stSidebar"]{
  background: var(--bg) !important;
  color: var(--text) !important;
}

section.main > div { padding-top: 0.25rem; }

/* Page title */
h1, .stMarkdown h1{
  letter-spacing: .2px;
  text-shadow: 0 6px 24px rgba(124,58,237,.25);
}

/* Sidebar */
[data-testid="stSidebar"]{
  border-right: none !important;
  background: linear-gradient(180deg, #0b1020 0%, #0c1224 100%) !important;
}
[data-testid="stSidebar"] > div{
  padding: 10px 8px 18px 8px;
}
[data-testid="stSidebarNav"] { display: none; }

/* Inputs → neumorphic shells */
.neo-shell{
  border-radius: var(--radius);
  background: var(--surface);
  box-shadow:
    10px 10px 24px var(--shadow-dark),
    -10px -10px 24px var(--shadow-light),
    inset 0 0 0 rgba(0,0,0,0);
  padding: 14px 16px;
  transition: .25s ease;
}
.neo-shell:hover{
  box-shadow:
    14px 14px 28px var(--shadow-dark),
    -14px -14px 28px var(--shadow-light);
}

/* Neumorphic CARD */
.neo-card{
  border-radius: 24px;
  background: linear-gradient(145deg, var(--surface), var(--surface-2));
  box-shadow:
    14px 14px 28px var(--shadow-dark),
    -14px -14px 28px var(--shadow-light);
  padding: 18px 20px;
  border: 1px solid rgba(255,255,255,0.04);
}

/* Header inside card */
.neo-card h3, .neo-card h4{
  margin: 0 0 10px 0;
  font-weight: 700;
}

/* Chips / badges */
.neo-chip{
  display:inline-flex; align-items:center; gap:.5rem;
  padding:6px 10px; border-radius:999px;
  background: var(--surface-2);
  box-shadow:
    6px 6px 14px var(--shadow-dark),
    -6px -6px 14px var(--shadow-light);
  font-weight: 600; font-size: .9rem; color: var(--text);
}
.neo-chip.good{ background: rgba(34,197,94,.12); border:1px solid rgba(34,197,94,.25); }
.neo-chip.warn{ background: rgba(245,158,11,.12); border:1px solid rgba(245,158,11,.25); }
.neo-chip.bad{  background: rgba(239,68,68,.12);  border:1px solid rgba(239,68,68,.25);  }
.neo-chip.accent{ background: rgba(124,58,237,.14); border:1px solid rgba(124,58,237,.35); }

/* Neo buttons */
.stButton > button{
  width: 100%;
  border-radius: 18px;
  border: 0;
  padding: 12px 16px;
  font-weight: 700;
  background: linear-gradient(145deg, #6d28d9, #7c3aed);
  color: white;
  box-shadow:
    10px 10px 24px var(--shadow-dark),
    -10px -10px 24px rgba(124,58,237,.25);
  transition: .25s ease;
}
.stButton > button:hover{
  transform: translateY(-1px);
  box-shadow:
    14px 14px 28px var(--shadow-dark),
    -14px -14px 28px rgba(124,58,237,.28);
}

/* Sliders */
[data-testid="stSlider"] > div:nth-child(1){
  background: transparent !important;
}
[data-testid="stTickBar"]{ display:none; }
[data-testid="stSlider"] .st-emotion-cache-16j0g1l{
  background: var(--surface-2) !important;
  border-radius: 999px !important;
  box-shadow:
    inset 8px 8px 18px var(--shadow-dark),
    inset -8px -8px 18px var(--shadow-light);
}
[data-testid="stSlider"] .st-emotion-cache-1gv3huu{
  background: linear-gradient(90deg, #6d28d9, #7c3aed) !important;
}

/* Dataframe glass effect */
[data-testid="stDataFrame"]{
  border-radius: 16px;
  background: radial-gradient(160% 120% at 0% 0%, rgba(124,58,237,.06), transparent 60%),
              rgba(255,255,255,0.02);
  box-shadow:
    inset 6px 6px 14px rgba(0,0,0,.35),
    inset -6px -6px 14px rgba(255,255,255,.03),
    8px 8px 24px var(--shadow-dark);
}

/* Images */
[data-testid="stImage"] img{
  border-radius: 16px;
  box-shadow:
    14px 14px 28px var(--shadow-dark),
    -14px -14px 28px var(--shadow-light);
}

/* Download buttons */
a[download]{
  display:inline-flex; align-items:center; justify-content:center;
  gap:.5rem; text-decoration:none !important;
  border-radius: 16px; padding: 10px 14px; font-weight:700;
  background: linear-gradient(145deg, #0f162d, #111a33);
  color: var(--text); border:1px solid rgba(255,255,255,0.06);
  box-shadow: 10px 10px 24px var(--shadow-dark), -10px -10px 24px var(--shadow-light);
}
a[download]:hover{ filter: brightness(1.08); }

/* Custom progress (if you use it) */
.neo-progress{
  width:100%; height:12px; border-radius:999px;
  background: rgba(255,255,255,.06);
  box-shadow:
    inset 8px 8px 18px rgba(0,0,0,.45),
    inset -8px -8px 18px rgba(255,255,255,.04);
  overflow:hidden;
}
.neo-progress > span{
  display:block; height:100%;
  background: linear-gradient(90deg, #22c55e, #7c3aed);
  border-radius:999px; width:0%;
  box-shadow: 0 0 24px rgba(124,58,237,.35);
  transition: width .6s ease;
}
.small-muted{ color: var(--muted); font-size:.85rem; }
</style>
""", unsafe_allow_html=True)

st.title("🚗 Car Condition AI")
st.caption(APP_TAGLINE)

with st.sidebar:
    st.subheader("Настройки")
    st.markdown("**Подключение к Roboflow Workflow**")
    st.text_input("WORKFLOW_URL", value=WORKFLOW_URL, key="wf_url_help", help="Ссылка из 'Run from Anywhere' (можно с api_key=)")
    conf_det = st.slider("Порог детекции (confidence)", 0.05, 0.95, 0.30, 0.05)
    iou_det = st.slider("IoU NMS", 0.1, 0.9, 0.3, 0.05)
    st.markdown("---")
    st.info("Загрузи фото автомобиля (без номеров).")

uploaded = st.file_uploader("Загрузить изображение", type=["jpg", "jpeg", "png", "webp"])
demo_cols = st.columns(3)
with demo_cols[0]:
    demo_url = st.text_input("…или вставь URL картинки", value="")
go = st.button("▶️ Запустить анализ", use_container_width=True)

if go:
    if not uploaded and not demo_url:
        st.warning("Добавь файл или URL.")
        st.stop()

    # Загружаем bytes
    if uploaded:
        image_bytes = uploaded.read()
        input_img = pil_from_bytes(image_bytes)
    else:
        try:
            r = requests.get(demo_url, timeout=20)
            r.raise_for_status()
            image_bytes = r.content
            input_img = pil_from_bytes(image_bytes)
        except Exception as e:
            st.error(f"Не удалось скачать URL: {e}")
            st.stop()

    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.subheader("Входное изображение")
        st.image(input_img, use_column_width=True)

    # Инференс
    t0 = time.time()
    try:
        json_resp = post_image_to_workflow(image_bytes, conf_det=conf_det, iou_det=iou_det)
    except Exception as e:
        st.error(f"Ошибка при запросе в Workflow: {e}")
        st.stop()
    dt = (time.time() - t0) * 1000

    parsed = parse_workflow(json_resp)

    # Визуализация damage
    dets = parsed["damage_predictions"]
    vis_img = draw_detections(input_img, dets)

    # Правый столбец: карточки статусов
    with c2:
        st.subheader("Результаты")
        st.write(f"⏱️ Время инференса: **{dt:.0f} ms**")

        # Cleanliness badge
        clean = parsed["cleanliness_top"]
        clean_conf = parsed["cleanliness_conf"]
        if clean:
            color_map = {
                "super clean": "#10b981",
                "clean": "#22c55e",
                "slightly dirty": "#f59e0b",
                "dirty": "#ef4444",
                "super dirty": "#b91c1c",
            }
            color = color_map.get(clean.lower(), "#7c3aed")
            badge(f"🧼 {clean} ({clean_conf:.2f})", color=color)
        else:
            badge("🧼 нет предсказания чистоты", "#6b7280")

        st.markdown("—")

        # Damage summary
        if dets:
            n_dents = sum(1 for d in dets if d.get("class") == "dent")
            n_scr = sum(1 for d in dets if d.get("class") == "scratch")
            sev = severity_score(dets, input_img.size)
            badge(f"🚨 Повреждений: {len(dets)}  |  dent: {n_dents}, scratch: {n_scr}", "#ef4444" if len(dets) else "#22c55e")
            st.progress(min(1.0, sev), text="Оценка серьёзности (по площади боксов)")
            # Таблица
            rows = [
                {
                    "class": d.get("class"),
                    "confidence": round(float(d.get("confidence", 0.0)), 3),
                    "x": int(d.get("x", 0)),
                    "y": int(d.get("y", 0)),
                    "w": int(d.get("width", 0)),
                    "h": int(d.get("height", 0)),
                }
                for d in dets
            ]
            st.dataframe(rows, use_container_width=True)
        else:
            badge("✅ Видимых повреждений не обнаружено", "#22c55e")

    st.subheader("Аннотированное изображение")
    st.image(vis_img, use_column_width=True)

    # Кнопки скачивания
    col_dl1, col_dl2 = st.columns(2)

    with io.BytesIO() as buf:
        vis_img.save(buf, format="JPEG", quality=92)
        img_b64 = buf.getvalue()

    with col_dl1:
        st.download_button(
            "⬇️ Скачать аннотацию (JPEG)",
            data=img_b64,
            file_name="annotated.jpg",
            mime="image/jpeg",
            use_container_width=True,
        )

    with col_dl2:
        st.download_button(
            "⬇️ Скачать JSON ответ",
            data=json.dumps(json_resp, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name="inference.json",
            mime="application/json",
            use_container_width=True,
        )

else:
    st.info("Загрузи фото или вставь URL, потом жми **Запустить анализ**.")

st.markdown("---")
st.caption("⚙️ Powered by Roboflow Workflows • Streamlit UI")
