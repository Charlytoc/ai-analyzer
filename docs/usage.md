# Cómo usar el proyecto

Luego de que el proyecto esté instalado y ejecutándose, puedes usar el proyecto de la siguiente manera.

## Activar entorno virtual

```bash
python -m venv venv
source venv/bin/activate
```

o en Windows

```bash
python -m venv venv
.\venv\Scripts\activate
```

## Instalar dependencias

```bash
pip install -r requirements.txt
```

## Ejecutar el proyecto

```bash
python main.py
```

## En otra terminal ejecuta el contenedor de redis

```bash
docker run -d --name document_redis -p 6380:6379 redis
```

o si ya tienes el contenedor creado, solo lo inicias

```bash
docker start document_redis
```

---

### Alternativamente puedes usar el siguiente comando para iniciar el proyecto y el contenedor de redis

```bash
./start.sh -m prod
```

