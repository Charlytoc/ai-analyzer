# Guía para instalar vLLM en un servidor Linux

## Requisitos previos

- Tener instalado `Python` 3.10 <= 3.12 (desde la 3.10 hasta la 3.12)
- Tener instalado `pip`

## Instalación

1. Crear un entorno virtual y activarlo

```bash
python3 -m venv vllm_env
source vllm_env/bin/activate
```

2. Instalar vLLM

```bash
pip install vllm
```

3. Iniciar el servidor vLLM

```bash
vllm serve google/gemma-3-12b-it --host 0.0.0.0 --port 8009 --dtype bfloat16 --max-model-len 15000 --max-num-seqs 3
```

4. Una vez el servidor está corriendo (habrá un log que diga "Application startup complete"), puedes usar el siguiente comando en otra terminal para hacer una prueba

```bash
curl http://localhost:8009/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "google/gemma-3-12b-it",
    "messages": [
      {
        "role": "user",
        "content": "¿Cuál es la capital de Francia?"
      }
    ]
  }'

```

5. Agregar la URL del servidor vLLM a la variable de entorno `PROVIDER_BASE_URL` en el archivo `.env`

Nota: Si el servidor vLLM está corriendo en un servidor remoto, puedes usar la URL del servidor remoto, solo asegúrate de colocar: **HOST:PORT/v1** (debe terminar en /v1 sin slash al final

Por ejemplo, si el servidor vLLM está corriendo en `http://192.168.1.100:8009`, debes agregar: `PROVIDER_BASE_URL=http://192.168.1.100:8009/v1`

6. Reiniciar el servidor

Detener el proceso del servidor del intérprete y ejecutar:

```bash
./start.sh
```
