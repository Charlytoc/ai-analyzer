import os
import subprocess

MEDIA_ROOT = os.environ.get("MEDIA_ROOT", "media")

VECTOR_STORAGE_PATH = os.environ.get(
    "VECTOR_STORAGE_PATH", os.path.join(MEDIA_ROOT, "vector_storage/")
)

CHROMA_PORT = os.environ.get("CHROMA_PORT", 8004)


def start_chroma_server():

    process = subprocess.Popen(
        ["chroma", "run", "--path", VECTOR_STORAGE_PATH, "--port", str(CHROMA_PORT)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    process.wait()
