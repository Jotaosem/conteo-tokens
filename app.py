"""Herramienta de Predicción de Consumo y Optimización de Prompts."""

import re

import streamlit as st
import tiktoken

# ---------------------------------------------------------------------------
# Configuración de modelos
# Precios orientativos en USD por 1M de tokens. Verifícalos en la web de
# cada proveedor antes de usarlos para presupuestar.
# ---------------------------------------------------------------------------
MODELOS = {
    # OpenAI
    "GPT-5.4": {"encoding": "o200k_base", "contexto": 272_000, "precio_in": 2.50, "precio_out": 15.00},
    "GPT-5.4 mini": {"encoding": "o200k_base", "contexto": 400_000, "precio_in": 0.75, "precio_out": 4.50},
    "GPT-5.4 nano": {"encoding": "o200k_base", "contexto": 400_000, "precio_in": 0.20, "precio_out": 1.25},
    "GPT-4o": {"encoding": "o200k_base", "contexto": 128_000, "precio_in": 2.50, "precio_out": 10.00},
    "GPT-4o mini": {"encoding": "o200k_base", "contexto": 128_000, "precio_in": 0.15, "precio_out": 0.60},
    # Anthropic
    "Claude Opus 5.5": {"encoding": "cl100k_base", "contexto": 1_000_000, "precio_in": 4.00, "precio_out": 20.00},
    "Claude Sonnet 5.5": {"encoding": "cl100k_base", "contexto": 1_000_000, "precio_in": 2.00, "precio_out": 10.00},
    "Claude Haiku 5.5": {"encoding": "cl100k_base", "contexto": 1_000_000, "precio_in": 0.10, "precio_out": 0.50},
    "Claude Opus 4.6": {"encoding": "cl100k_base", "contexto": 1_000_000, "precio_in": 5.00, "precio_out": 25.00},
    "Claude Sonnet 4.6": {"encoding": "cl100k_base", "contexto": 1_000_000, "precio_in": 3.00, "precio_out": 15.00},
    "Claude Haiku 4.5": {"encoding": "cl100k_base", "contexto": 200_000, "precio_in": 1.00, "precio_out": 5.00},
    "Claude 3.5 Sonnet": {"encoding": "cl100k_base", "contexto": 200_000, "precio_in": 3.00, "precio_out": 15.00},
    "Claude 3.5 Haiku": {"encoding": "cl100k_base", "contexto": 200_000, "precio_in": 0.80, "precio_out": 4.00},
    # Google
    "Gemini 3.5 Pro": {"encoding": "o200k_base", "contexto": 2_000_000, "precio_in": 1.25, "precio_out": 10.00},
    "Gemini 3.5 Flash": {"encoding": "o200k_base", "contexto": 1_000_000, "precio_in": 1.50, "precio_out": 9.00},
    "Gemini 3.1 Pro": {"encoding": "o200k_base", "contexto": 1_000_000, "precio_in": 2.00, "precio_out": 12.00},
    "Gemini 3 Flash": {"encoding": "o200k_base", "contexto": 1_000_000, "precio_in": 0.50, "precio_out": 3.00},
    "Gemini 3.1 Flash-Lite": {"encoding": "o200k_base", "contexto": 1_000_000, "precio_in": 0.25, "precio_out": 1.50},
    "Gemini 2.5 Pro": {"encoding": "o200k_base", "contexto": 1_048_576, "precio_in": 1.25, "precio_out": 10.00},
    "Gemini 2.5 Flash": {"encoding": "o200k_base", "contexto": 1_000_000, "precio_in": 0.30, "precio_out": 2.50},
    "Gemini 2.5 Flash-Lite": {"encoding": "o200k_base", "contexto": 1_000_000, "precio_in": 0.10, "precio_out": 0.40},
    "Gemini 1.5 Pro": {"encoding": "o200k_base", "contexto": 2_000_000, "precio_in": 1.25, "precio_out": 5.00},
    "Gemini 1.5 Flash": {"encoding": "o200k_base", "contexto": 1_000_000, "precio_in": 0.075, "precio_out": 0.30},
}

# Tokens de salida estimados según el tipo de respuesta esperado.
TIPOS_RESPUESTA = {
    "Corta / Directa (Código breve, confirmaciones, resúmenes breves)": 300,
    "Media / Detallada (Análisis estándar, explicaciones paso a paso)": 1_200,
    "Larga / Compleja (Programación completa, reportes extensos, redacción creativa)": 3_500,
}

# ---------------------------------------------------------------------------
# Patrones de ineficiencia
# ---------------------------------------------------------------------------
PATRON_CORTESIA = re.compile(
    r"\b(por favor|porfavor|serías tan amable|serias tan amable|muchas gracias|gracias|"
    r"te agradecería|le agradecería|si no es molestia)\b",
    re.IGNORECASE,
)
PATRON_ESTRUCTURA = re.compile(r"^\s*([-*#]|\d+[.)])\s", re.MULTILINE)
PATRON_INTENSIFICADOR = re.compile(
    r"\b(muy|bastante|extremadamente|s[uú]per|realmente|sumamente|totalmente|absolutamente)\b",
    re.IGNORECASE,
)


@st.cache_resource(show_spinner=False)
def obtener_encoding(nombre):
    """Carga el encoding de tiktoken una sola vez por sesión del servidor."""
    return tiktoken.get_encoding(nombre)


def contar_tokens(texto, nombre_encoding):
    """Cuenta tokens. Si tiktoken no puede cargar el encoding, estima ~4 caracteres por token."""
    if not texto:
        return 0
    try:
        return len(obtener_encoding(nombre_encoding).encode(texto))
    except Exception:
        st.warning("No se pudo cargar tiktoken; se usa una estimación de ~4 caracteres por token.")
        return max(1, len(texto) // 4)


def analizar_prompt(texto):
    """Devuelve una lista de (tipo, mensaje) con las ineficiencias detectadas."""
    hallazgos = []

    cortesias = PATRON_CORTESIA.findall(texto)
    if cortesias:
        ahorro_min = 3 * len(cortesias)
        ahorro_max = 10 * len(cortesias)
        hallazgos.append((
            "warning",
            f"**Cortesía detectada** ({len(cortesias)} vez/veces: {', '.join(sorted(set(c.lower() for c in cortesias)))}). "
            f"La IA no tiene sentimientos. Quitar frases de cortesía ahorra entre "
            f"{ahorro_min} y {ahorro_max} tokens.",
        ))

    if len(texto) > 1000 and not PATRON_ESTRUCTURA.search(texto):
        hallazgos.append((
            "warning",
            "**Prompt largo sin estructura** (más de 1.000 caracteres). Estructura con viñetas (-) "
            "o títulos (#) para que la IA entienda el contexto con menos repeticiones.",
        ))

    intensificadores = PATRON_INTENSIFICADOR.findall(texto)
    if len(intensificadores) >= 2:
        hallazgos.append((
            "warning",
            f"**Adjetivos repetitivos o relleno** ({len(intensificadores)} intensificadores como "
            f"'{intensificadores[0]}'). Define un rol claro al inicio en vez de usar adjetivos repetitivos. "
            f"Ej: 'Actúa como Senior QA'.",
        ))

    return hallazgos


# ---------------------------------------------------------------------------
# Interfaz
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Predicción de Consumo de Tokens", page_icon="🧮", layout="wide")

ESTILOS = """
<style>
/* Botón de cálculo: compacto, con degradado y bordes redondeados */
div[data-testid="stForm"] button {
    width: 100%;
    min-height: 2.25rem;
    padding: 0.3rem 1rem;
    border: none !important;
    border-radius: 999px !important;
    font-size: 0.85rem;
    font-weight: 600;
    letter-spacing: 0.02em;
    color: #ffffff !important;
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    box-shadow: 0 4px 12px rgba(99, 102, 241, 0.35);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
div[data-testid="stForm"] button:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 16px rgba(99, 102, 241, 0.45);
}

/* Responsive: ajustes para tablets y móviles */
@media (max-width: 768px) {
    .block-container { padding-left: 1rem; padding-right: 1rem; }
    h1 { font-size: 1.6rem !important; }
    div[data-testid="stMetricValue"] { font-size: 1.25rem; }
}
</style>
"""
st.markdown(ESTILOS, unsafe_allow_html=True)
st.title("🧮 Herramienta de Predicción de Consumo y Optimización de Prompts")
st.caption("Estima el costo total de una tarea (entrada + salida) y detecta ineficiencias en tu prompt.")

with st.sidebar:
    st.header("⚙️ Configuración")
    nombre_modelo = st.selectbox("Modelo de IA", list(MODELOS.keys()))
    modelo = MODELOS[nombre_modelo]
    st.metric("Límite de contexto", f"{modelo['contexto']:,} tokens")
    st.divider()
    st.subheader("📤 Respuesta esperada")
    tipo_respuesta = st.radio(
        "¿Qué tipo de respuesta esperas de la IA?",
        list(TIPOS_RESPUESTA.keys()),
        index=1,
    )
    tokens_salida = TIPOS_RESPUESTA[tipo_respuesta]

# Dentro de un formulario, Ctrl+Intro y el botón envían el texto.
with st.form("formulario_prompt"):
    prompt = st.text_area(
        "✍️ Escribe tu prompt (Ctrl+Intro o botón para calcular)",
        height=260,
        placeholder="Pega aquí el prompt que enviarás a la IA...",
    )
    _, col_boton = st.columns([4, 1])
    with col_boton:
        st.form_submit_button("🧮 Calcular", type="primary")

# --- Cálculo predictivo ---------------------------------------------------
tokens_entrada = contar_tokens(prompt, modelo["encoding"])
total_predicho = tokens_entrada + tokens_salida
costo_usd = (tokens_entrada * modelo["precio_in"] + tokens_salida * modelo["precio_out"]) / 1_000_000

st.subheader("📊 Predicción de la tarea")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Tokens de entrada", f"{tokens_entrada:,}", help="Tokens de tu prompt (aprox. para modelos no OpenAI)")
c2.metric("Tokens de salida estimados", f"{tokens_salida:,}", help="Según el tipo de respuesta elegido")
c3.metric("Consumo Total Predicho de la Tarea", f"{total_predicho:,}")
c4.metric("Costo estimado", f"${costo_usd:,.4f} USD", help="Precios orientativos, ver MODELOS en app.py")

# --- Barra de progreso del contexto --------------------------------------
st.subheader("🧠 Contexto total acumulado")
ratio = total_predicho / modelo["contexto"]
st.progress(min(ratio, 1.0), text=f"{total_predicho:,} / {modelo['contexto']:,} tokens ({ratio:.2%})")

if tokens_entrada > modelo["contexto"]:
    st.error(f"❌ El prompt supera el límite de contexto de {nombre_modelo}. Reduce el texto o cambia de modelo.")
elif total_predicho > modelo["contexto"]:
    st.error("❌ La entrada más la salida estimada superan el contexto. La respuesta podría cortarse.")
elif ratio > 0.8:
    st.warning("⚠️ Estás usando más del 80% del contexto disponible.")
else:
    st.success("✅ El consumo previsto cabe cómodamente en el contexto del modelo.")

# --- Recomendaciones de optimización -------------------------------------
st.subheader("💡 Recomendaciones de Optimización")

if prompt.strip():
    hallazgos = analizar_prompt(prompt)
    if hallazgos:
        for tipo, mensaje in hallazgos:
            if tipo == "warning":
                st.warning(mensaje)
    else:
        st.success("No se detectaron ineficiencias comunes en tu prompt.")
else:
    st.info("Escribe un prompt para analizarlo.")

st.markdown("#### 🧰 Tácticas de ahorro de tokens")
st.info("🌐 Traduce contextos masivos a inglés si es posible (el inglés gasta hasta un 40% menos tokens que el español).")
st.info("🎯 Sustituye ejemplos redundantes por solo 1 o 2 casos de ejemplo bien estructurados (Few-Shot comprimido).")
