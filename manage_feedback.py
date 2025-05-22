from server.utils.printer import Printer
from server.ai.vector_store import get_chroma_client

printer = Printer("FEEDBACK_MANAGER")
chroma_client = get_chroma_client()

collection_name = "sentence_feedbacks"
collection = chroma_client.get_or_create_collection(collection_name)

count = collection.count()
printer.green(count, "chunks in collection")

if count == 0:
    printer.yellow("No hay feedbacks en la colección. Nada que borrar.")
    exit()

# Obtener todos los feedbacks con metadata
chunks = collection.get()
ids = chunks["ids"]
metas = chunks["metadatas"]

# Mostrar los feedbacks desde metadata
for idx, (meta, id_) in enumerate(zip(metas, ids)):
    feedback = meta.get("feedback", "[Sin feedback]")
    printer.blue(f"[{idx}] {feedback}")

# Preguntar al usuario
print("\nOpciones:")
print("  - Escribe 'all' para borrar todo")
print("  - Escribe índices separados por coma para borrar selectivamente (ej: 0,2,5)")
user_input = input("¿Qué feedbacks quieres borrar? ").strip()

if user_input.lower() == "all":
    collection.delete(ids=ids)
    printer.red("Todos los feedbacks han sido eliminados.")
else:
    try:
        indices = [int(i) for i in user_input.split(",") if i.strip().isdigit()]
        ids_to_delete = [ids[i] for i in indices]
        collection.delete(ids=ids_to_delete)
        printer.red(f"{len(ids_to_delete)} feedbacks eliminados.")
    except Exception as e:
        printer.red(f"Error al eliminar: {e}")
