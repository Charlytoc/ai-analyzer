import os
from server.utils.printer import Printer
from server.utils.redis_cache import RedisCache

printer = Printer("FEEDBACK_MANAGER")
redis_cache = RedisCache()

key = "all_feedbacks"

def show_feedbacks():
    feedbacks = redis_cache.lrange(key, 0, -1)
    if not feedbacks:
        printer.yellow("No hay feedbacks en Redis. Nada que mostrar.")
        return []
    for idx, fb in enumerate(feedbacks):
        printer.blue(f"[{idx}] {fb}")
    return feedbacks

def add_feedback():
    feedback = input("Escribe el feedback a agregar: ").strip()
    if feedback:
        redis_cache.rpush(key, feedback)
        printer.green("Feedback agregado.")
    else:
        printer.yellow("No se agregó feedback vacío.")

def delete_feedbacks():
    feedbacks = show_feedbacks()
    if not feedbacks:
        return
    print("\nOpciones:")
    print("  - Escribe 'all' para borrar todo")
    print("  - Escribe índices separados por coma para borrar selectivamente (ej: 0,2,5)")
    user_input = input("¿Qué feedbacks quieres borrar? ").strip()
    if user_input.lower() == "all":
        redis_cache.delete(key)
        printer.red("Todos los feedbacks han sido eliminados.")
    else:
        try:
            indices = [int(i) for i in user_input.split(",") if i.strip().isdigit()]
            # Redis no permite borrar por índice, así que marcamos y luego removemos
            for i in sorted(indices, reverse=True):
                if 0 <= i < len(feedbacks):
                    redis_cache.lset(key, i, "__DELETED__")
            redis_cache.lrem(key, 0, "__DELETED__")
            printer.red(f"{len(indices)} feedbacks eliminados.")
        except Exception as e:
            printer.red(f"Error al eliminar: {e}")

def main():
    while True:
        print("\nOpciones:")
        print("  1. Mostrar feedbacks")
        print("  2. Agregar feedback")
        print("  3. Eliminar feedbacks")
        print("  4. Salir")
        choice = input("Selecciona una opción: ").strip()
        if choice == "1":
            show_feedbacks()
        elif choice == "2":
            add_feedback()
        elif choice == "3":
            delete_feedbacks()
        elif choice == "4":
            print("Saliendo.")
            break
        else:
            printer.yellow("Opción no válida.")

if __name__ == "__main__":
    main()
