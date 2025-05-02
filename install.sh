#!/usr/bin/env bash
set -euo pipefail

PYTHON_VERSION="3.12.7"
ENV_FILE=".env"
ENV_EXAMPLE=".env.example"

echo "💀 Iniciando instalación furiosa…"

# Detectar gestor de paquetes
if command -v apt-get &>/dev/null; then
  PM="apt-get"; SUDO="sudo"
elif command -v yum &>/dev/null; then
  PM="yum";    SUDO="sudo"
else
  echo "❌ Gestor de paquetes no soportado. Solo apt-get o yum." >&2
  exit 1
fi

install_pyenv_and_python() {
  if command -v python3 &>/dev/null && [[ "$(python3 -V 2>&1)" == "Python $PYTHON_VERSION" ]]; then
    echo "✅ Python $PYTHON_VERSION ya está instalado."
    return
  fi
  echo "🔧 Instalando dependencias de pyenv..."
  if [ "$PM" = "apt-get" ]; then
    $SUDO apt-get update
    $SUDO apt-get install -y make build-essential libssl-dev zlib1g-dev \
      libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
      libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev
  else
    $SUDO yum install -y gcc zlib-devel bzip2 bzip2-devel readline-devel \
      sqlite sqlite-devel openssl-devel tk-devel libffi-devel xz-devel
  fi
  echo "📂 Clonando pyenv..."
  git clone https://github.com/pyenv/pyenv.git ~/.pyenv 2>/dev/null || true
  export PYENV_ROOT="$HOME/.pyenv"
  export PATH="$PYENV_ROOT/bin:$PATH"
  eval "$(pyenv init --path)"
  echo "🚀 Instalando Python $PYTHON_VERSION con pyenv..."
  pyenv install -s "$PYTHON_VERSION"
  pyenv global "$PYTHON_VERSION"
  echo "✅ Python $PYTHON_VERSION instalado y activo."
}

install_docker() {
  if command -v docker &>/dev/null; then
    echo "✅ Docker ya instalado."; return
  fi
  echo "🐳 Instalando Docker…"

  # 1) Amazon Linux
  if command -v amazon-linux-extras &>/dev/null; then
    sudo amazon-linux-extras install -y docker

  # 2) Debian/Ubuntu
  elif command -v apt-get &>/dev/null; then
    … # tu bloque apt-get

  # 3) Fedora (dnf)
  elif command -v dnf &>/dev/null; then
    sudo dnf install -y docker

  # 4) RHEL/CentOS (yum)
  elif command -v yum &>/dev/null; then
    … # tu bloque yum

  else
    echo "⚠️ No se detectó gestor compatible (apt, dnf, yum)." >&2
    exit 1
  fi

  sudo systemctl enable --now docker
  echo "✅ Docker listo."
}



install_ollama() {
  if command -v ollama &>/dev/null; then
    echo "✅ Ollama ya está instalado."
    return
  fi
  echo "🔮 Instalando Ollama..."
  curl -fsSL https://ollama.com/install.sh | sh
  echo "✅ Ollama instalado."
}

install_tesseract() {
  if command -v tesseract &>/dev/null; then
    echo "✅ Tesseract ya está instalado."
    return
  fi

  echo "🔍 Instalando Tesseract OCR…"
  # Detectar distro
  . /etc/os-release

  if [[ "$ID" == "amzn" ]]; then
    # Amazon Linux 2 / 2023
    echo "👉 Habilitando repositorio EPEL en Amazon Linux…"
    if command -v amazon-linux-extras &>/dev/null; then
      sudo amazon-linux-extras install -y epel
    else
      sudo dnf install -y epel-release
    fi
    echo "👉 Instalando Tesseract desde EPEL…"
    sudo yum install -y tesseract tesseract-langpack-spa

  elif command -v apt-get &>/dev/null; then
    # Debian/Ubuntu
    sudo apt-get update
    sudo apt-get install -y tesseract-ocr tesseract-ocr-spa

  elif command -v dnf &>/dev/null; then
    # Fedora / RHEL8+ / CentOS8+
    sudo dnf install -y epel-release
    sudo dnf install -y tesseract tesseract-langpack-spa

  else
    # Otros yum-based (CentOS7/RHEL7)
    sudo yum install -y epel-release
    sudo yum install -y tesseract tesseract-langpack-spa
  fi

  echo "✅ Tesseract instalado."
}

main() {
  install_pyenv_and_python
  install_docker
  install_ollama
  install_tesseract

  # Configurar .env
  if [ ! -f "$ENV_FILE" ] && [ -f "$ENV_EXAMPLE" ]; then
    cp "$ENV_EXAMPLE" "$ENV_FILE"
    echo "📋 Copiado $ENV_EXAMPLE → $ENV_FILE"
  fi

  # Ajustar TESSERACT_CMD
  T_CMD=$(command -v tesseract)
  if grep -q "^TESSERACT_CMD=" "$ENV_FILE"; then
    sed -i "s|^TESSERACT_CMD=.*|TESSERACT_CMD=$T_CMD|" "$ENV_FILE"
  else
    echo "TESSERACT_CMD=$T_CMD" >> "$ENV_FILE"
  fi

  echo -e "\n🔥 ¡Listo! Edita $ENV_FILE para configurar PROVIDER y MODEL.\nLuego ejecuta ./start.sh -m <dev|prod>."
}

main
