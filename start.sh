#!/bin/bash

VENV_DIR="venv"
MODE=""

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
        *)
            echo "❌ Argumento desconocido: $1"
            exit 1
            ;;
    esac
    shift
done

# Preguntar al usuario si no se especificó
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
pip install -r requirements.txt
echo "✅ Dependencias instaladas."

# Solo en modo desarrollo corre React
if [ "$MODE" == "dev" ]; then
    echo "📦 Instalando dependencias del cliente en modo desarrollo..."
    pushd client
    npm install
    echo "🚀 Iniciando cliente en modo desarrollo..."
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

echo "🚀 Iniciando servidor de Chroma..."
chroma run --path media/vector_storage/ --port 8004 &

echo "🚀 Esperando a que el servidor de Chroma esté listo..."
sleep 5

echo "🚀 Iniciando la aplicación FastAPI..."
python main.py
