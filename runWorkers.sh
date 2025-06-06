#!/bin/bash

# Verificar si existe el entorno virtual
if [ ! -d "venv" ]; then
    echo "❌ No se encontró el directorio 'venv'."
    echo "💡 Debes crear un entorno virtual primero con:"
    echo ""
    echo "    python3 -m venv venv"
    echo ""
    exit 1
fi

# Activar el entorno virtual
# Activar el entorno virtual según el sistema operativo
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" || "$OS" == "Windows_NT" ]]; then
    ACTIVATE_SCRIPT="venv\\Scripts\\activate"
else
    ACTIVATE_SCRIPT="venv/bin/activate"
fi

if [ ! -f "$ACTIVATE_SCRIPT" ]; then
    echo "❌ No se encontró el script de activación del entorno virtual: $ACTIVATE_SCRIPT"
    exit 1
fi

echo "🐍 Activando entorno virtual..."
# shellcheck disable=SC1090
source "$ACTIVATE_SCRIPT"

# Verificar que existan los requerimientos
REQUIREMENTS_FILE="requirements.txt"
if [ ! -f "$REQUIREMENTS_FILE" ]; then
    echo "❌ No se encontró el archivo $REQUIREMENTS_FILE"
    echo "💡 Crea uno con las dependencias mínimas para Celery."
    exit 1
fi

# Instalar requerimientos
echo "📦 Instalando dependencias desde $REQUIREMENTS_FILE..."
pip install -r "$REQUIREMENTS_FILE"

# Cargar variables del .env si existe
ENV_FILE=".env"
if [ -f "$ENV_FILE" ]; then
    echo "📄 Cargando variables desde .env..."
    set -o allexport
    source "$ENV_FILE"
    set +o allexport
fi

# Preguntar si no existen las variables necesarias
ask_if_missing() {
    VAR_NAME=$1
    DEFAULT=$2
    CURRENT_VALUE=$(printenv "$VAR_NAME")

    if [ -z "$CURRENT_VALUE" ]; then
        echo "🔧 Ingrese el valor para $VAR_NAME (default: $DEFAULT):"
        read INPUT
        if [ -z "$INPUT" ]; then
            INPUT=$DEFAULT
        fi
        export "$VAR_NAME"="$INPUT"

        # Agregar al .env si no existe
        if ! grep -q "^$VAR_NAME=" "$ENV_FILE" 2>/dev/null; then
            echo "$VAR_NAME=$INPUT" >> "$ENV_FILE"
            echo "✅ Agregado $VAR_NAME=$INPUT al .env"
        fi
    else
        export "$VAR_NAME"="$CURRENT_VALUE"
    fi
}

ask_if_missing "REDIS_HOST" "localhost"
ask_if_missing "REDIS_PORT" "6379"
ask_if_missing "REDIS_DB" "0"

# Obtener concurrencia desde argumento o preguntar
CONCURRENCY=""
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        -c|--concurrency)
            if [[ -n "$2" && "$2" != -* ]]; then
                CONCURRENCY="$2"
                shift
            else
                echo "❌ Se esperaba un valor para $1 (ej: 4)"
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

if [ -z "$CONCURRENCY" ]; then
    echo "🔢 ¿Cuántos procesos de concurrencia desea para Celery? (default: 4)"
    read CONCURRENCY
    if [ -z "$CONCURRENCY" ]; then
        CONCURRENCY=4
    fi
fi

BROKER_URL="redis://${REDIS_HOST}:${REDIS_PORT}/${REDIS_DB}"
echo "🚀 Iniciando Celery worker con broker: $BROKER_URL"


celery -A server.celery_app worker --loglevel=info --concurrency=$CONCURRENCY -E -n "worker@%h"

