# Herramienta de Predicción de Consumo y Optimización de Prompts

Aplicación en Streamlit que estima el consumo total de una tarea con IA (tokens de entrada + tokens de salida previstos), calcula un costo orientativo y analiza el prompt para sugerir optimizaciones.

## Requisitos

- Python 3.9 o superior

## Instalación

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

pip install -r requirements.txt
```

## Ejecución

```bash
streamlit run app.py
```

Se abrirá el navegador en `http://localhost:8501`.

## Notas

- La primera ejecución descarga los encodings de `tiktoken` (requiere conexión a internet). Después quedan en caché.
- Los conteos para Claude y Gemini son **aproximaciones** basadas en los encodings de OpenAI; cada proveedor tokeniza de forma distinta.
- Los precios de la configuración `MODELOS` en `app.py` son orientativos: actualízalos según la tarifa vigente de cada proveedor.
