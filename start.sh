#!/bin/bash

VENV_DIR="venv"
MODE=""
CHROMA="true"  # Por defecto true

# Parsear argumentos con validación de valor
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        -m|--mode)
            if [[ -n "$2" && "$2" != -* ]]; then
                MODE="$2"
                shift
            else
                echo "❌ Se esperaba un valor para $1 (dev o prod)"
                exit 1
            fi
            ;;
        -c|--chroma)
            if [[ -n "$2" && "$2" != -* ]]; then
                CHROMA="$2"
                # Normalizar a minúsculas para evitar problemas
                CHROMA=$(echo "$CHROMA" | tr '[:upper:]' '[:lower:]')
                if [[ "$CHROMA" != "true" && "$CHROMA" != "false" ]]; then
                    echo "❌ Valor inválido para $1: debe ser 'true' o 'false'"
                    exit 1
                fi
                shift
            else
                echo "❌ Se esperaba un valor para $1 (true o false)"
                exit 1
            fi
            ;;
        *)
            echo "❌ Argumento desconocido: $1"
            exit 1
            ;;
    esac
    shift
done

# Preguntar al usuario si no se especificó modo
if [ -z "$MODE" ]; then
    echo "🛠️ ¿Desea iniciar en modo desarrollo o producción? (dev/prod)"
    read MODE
fi

# Validar modo
if [[ "$MODE" != "dev" && "$MODE" != "prod" ]]; then
    echo "❌ Modo inválido. Usa 'dev' o 'prod'."
    exit 1
fi

# Detectar sistema operativo
if [[ "$OSTYPE" == "msys" ]]; then
    ACTIVATE_CMD="$VENV_DIR/Scripts/activate"
else
    ACTIVATE_CMD="$VENV_DIR/bin/activate"
fi

echo "🔍 Verificando entorno virtual..."
if [ ! -d "$VENV_DIR" ]; then
    echo "📁 No se encontró un entorno virtual. Creando uno nuevo..."
    python3 -m venv $VENV_DIR
    echo "✅ Entorno virtual creado."
else
    echo "✅ Se encontró el entorno virtual existente."
fi

echo "⚙️ Activando el entorno virtual..."
source $ACTIVATE_CMD
echo "✅ Entorno virtual activado."

echo "📦 Instalando dependencias desde requirements.txt..."
pip install -r requirements.txt -q
echo "✅ Dependencias instaladas."

# Solo en modo desarrollo corre React
if [ "$MODE" == "dev" ]; then
    echo "📦 Instalando dependencias del cliente en modo desarrollo..."
    pushd client
    npm install
    echo "🚀 Iniciando cliente en modo desarrollo..."

    # Igual tiene que hacer un build
    npm run build
    echo "✅ Cliente en modo desarrollo instalado y listo para usar."
    npm run dev &
    popd
else
    echo "🏗️ Modo producción: NO se iniciará cliente React."
fi

echo "🚀 Verificando estado de Redis en Docker..."
if [ "$(docker ps -aq -f name=document_redis)" ]; then
    if [ "$(docker ps -q -f name=document_redis)" ]; then
        echo "✅ Redis ya está corriendo."
    else
        echo "🔄 Redis existe pero está detenido. Iniciando..."
        docker start document_redis
    fi
else
    echo "📦 Redis no existe. Creando contenedor..."
    docker run -d --name document_redis -p 6380:6379 redis
fi

if [ "$CHROMA" == "true" ]; then
    echo "🚀 Iniciando servidor de Chroma..."
    chroma run --path media/vector_storage/ --port 8004 &
    echo "🚀 Esperando a que el servidor de Chroma esté listo..."
    sleep 5
else
    echo "⚠️ Servidor de Chroma NO será iniciado (CHROMA=false)."
fi

echo "🚀 Iniciando worker de Celery..."
celery -A server.tasks worker --loglevel=info --concurrency=5 &

export ENVIRONMENT=$MODE

echo "🚀 Iniciando la aplicación FastAPI..."
python main.py
