import os
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from server.utils.ai_interface import check_ollama_installation
from server.utils.printer import Printer
from server.routes import router

printer = Printer("MAIN")
ENVIRONMENT = os.getenv("ENVIRONMENT", "prod")

# Crear carpetas necesarias
os.makedirs("uploads/images", exist_ok=True)
os.makedirs("uploads/documents", exist_ok=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    printer.yellow("Checking ollama installation")
    result = check_ollama_installation()
    if result["installed"]:
        printer.green("Ollama is installed")
        printer.green("Ollama version: ", result["version"])
        printer.green("Ollama server running: ", result["server_running"])
    else:
        printer.red("Ollama is not installed, please install it first")
    yield

# Inicializar FastAPI
app = FastAPI(lifespan=lifespan)

# Leer ALLOWED_ORIGINS
raw_origins = os.getenv("ALLOWED_ORIGINS", "*")
if raw_origins != "*":
    ALLOWED_ORIGINS = []
    for origin in raw_origins.split(","):
        origin = origin.strip()
        if not origin.startswith("http"):
            origin = f"http://{origin}"
        ALLOWED_ORIGINS.append(origin)
else:
    printer.red(
        "PELIGRO: ALLOWED_ORIGINS es *, cualquier origen puede acceder a la API. Se recomienda configurar valores explícitos como 'http://localhost:3000,http://localhost:8000'"
    )
    if ENVIRONMENT == "prod":
        raise Exception("ALLOWED_ORIGINS is * en producción")
    
printer.green("ALLOWED_ORIGINS: ", ALLOWED_ORIGINS)

# Middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middleware personalizado para bloquear Origins prohibidos
app.middleware("http") 
async def block_disallowed(request: Request, call_next): 
    origin = request.headers.get("origin") 
    if origin: 
        if ALLOWED_ORIGINS != "*" and origin not in ALLOWED_ORIGINS: 
            return JSONResponse(status_code=403, content={"detail": f"Origin '{origin}' no permitido."}) 
    else: 

# Si no hay Origin, validamos la IP 
        client_ip = request.client.host 
        if client_ip not in ALLOWED_ORIGINS:
            return JSONResponse(status_code=403, content={"detail": f"IP '{client_ip}' no permitida."}) 
    return await call_next(request)

# Registrar rutas
app.include_router(router)

# Servir archivos estáticos si no estamos en prod
if ENVIRONMENT != "prod":
    app.mount("/", StaticFiles(directory="client/dist", html=True), name="client")

# Definir puerto
PORT = int(os.getenv("PORT", 8005))

# Levantar el servidor
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=True)
