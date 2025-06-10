import sys
import pandas as pd
from tabulate import tabulate

# ANSI escapes
GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"
BOLD = "\033[1m"
CYAN = "\033[96m"
YELLOW = "\033[93m"


def barra_porcentaje(pct_exito, largo=30):
    """Barra visual con colores: verde para éxito, rojo para error."""
    ok = int(largo * pct_exito / 100)
    err = largo - ok
    barra = f"{GREEN}{'█' * ok}{RED}{'█' * err}{RESET}"
    return barra


def resumen_general(df):
    print(f"\n{BOLD}=== Resumen General ==={RESET}")
    total = len(df)
    exit_status_counts = df["exit_status"].value_counts()
    exito = exit_status_counts.get(0, 0) + exit_status_counts.get("0", 0)
    error = exit_status_counts.get(1, 0) + exit_status_counts.get("1", 0)
    pct_exito = (exito / total) * 100 if total else 0
    pct_error = (error / total) * 100 if total else 0

    print(f"{CYAN}Total de registros:{RESET} {total}")
    print(f"{CYAN}Endpoints únicos:{RESET} {df['endpoint'].nunique()}")
    print(f"{CYAN}Hashes únicos:{RESET} {df['hash'].nunique()}")
    print(f"\n{BOLD}Status HTTP:{RESET}")
    print(df["http_status"].value_counts())
    print(f"\n{BOLD}Exit Status:{RESET}")
    print(df["exit_status"].value_counts())
    print(f"\n{GREEN}✔ % Éxito:{RESET} {pct_exito:.2f}%")
    print(f"{RED}✖ % Error:{RESET} {pct_error:.2f}%")
    print(f"\n{BOLD}Barra visual (éxito vs error):{RESET}")
    print(
        f"[{barra_porcentaje(pct_exito)}] {GREEN}{pct_exito:.2f}% éxito{RESET} / {RED}{pct_error:.2f}% error{RESET}"
    )


def errores_por_hash(df):
    errores = df[df["http_status"] != 200]
    if errores.empty:
        print(f"\n{GREEN}No hay errores registrados.{RESET}")
        return
    print(f"\n{RED}{BOLD}=== Errores agrupados por hash ==={RESET}")
    agrupado = errores.groupby("hash").size().reset_index(name="veces")
    print(tabulate(agrupado, headers="keys", tablefmt="psql", showindex=False))


def resumen_temporal(df):
    print(f"\n{BOLD}=== Resumen Temporal (por día) ==={RESET}")
    df["fecha"] = pd.to_datetime(df["timestamp"]).dt.date
    tabla = df.groupby(["fecha", "http_status"]).size().unstack(fill_value=0)
    print(tabulate(tabla, headers="keys", tablefmt="psql"))


def ultimos_registros(df, n=10):
    print(f"\n{BOLD}=== Últimos {n} registros ==={RESET}")
    df_copy = df.tail(n).copy()
    if "message" in df_copy.columns:
        df_copy["message"] = df_copy["message"].astype(str).str[:80]
    print(tabulate(df_copy, headers="keys", tablefmt="psql", showindex=False))


def mensajes_error_unicos(df):
    errores = df[df["exit_status"].astype(str) != "0"]
    if errores.empty:
        print(f"\n{GREEN}No hay mensajes de error registrados.{RESET}")
        return
    print(
        f"\n{RED}{BOLD}=== Mensajes de error únicos (últimos 40 caracteres) ==={RESET}"
    )
    # Agrupar por hash y mensaje para evitar duplicados
    mensajes = errores[["hash", "message", "http_status"]].drop_duplicates()
    for i, (h, msg, http_status) in enumerate(mensajes.values, 1):
        msg_str = str(msg)
        resumen = msg_str[-200:] if len(msg_str) > 40 else msg_str
        print(
            f"\n{RED}--- Error #{i} HTTP {http_status}---{RESET} {CYAN}Hash:{RESET} {h}"
        )
        print(f"{YELLOW}{resumen}{RESET}")
    print(
        f"\n{BOLD}¿Quieres ver el mensaje completo? Usa la opción 5 e introduce el hash correspondiente.{RESET}"
    )


def barra_porcentaje_hash(pct_exito, largo=30):
    ok = int(largo * pct_exito / 100)
    err = largo - ok
    barra = f"{GREEN}{'█' * ok}{RED}{'█' * err}{RESET}"
    return barra


def mostrar_resumen_hash(filtrado):
    total = len(filtrado)
    exit_status_counts = filtrado["exit_status"].value_counts()
    exito = exit_status_counts.get(0, 0) + exit_status_counts.get("0", 0)
    error = exit_status_counts.get(1, 0) + exit_status_counts.get("1", 0)
    pct_exito = (exito / total) * 100 if total else 0
    pct_error = (error / total) * 100 if total else 0

    print(f"\n{BOLD}Resumen para este hash:{RESET}")
    print(
        f"{GREEN}✔ Éxito:{RESET} {exito}  {RED}✖ Error:{RESET} {error}  {CYAN}Total:{RESET} {total}"
    )
    print(
        f"[{barra_porcentaje_hash(pct_exito)}] {GREEN}{pct_exito:.2f}% éxito{RESET} / {RED}{pct_error:.2f}% error{RESET}"
    )

    # Mostrar mensajes de error completos, uno por bloque
    errores = filtrado[filtrado["exit_status"].astype(str) != "0"]
    mensajes = errores["message"].dropna().unique()
    if len(mensajes) > 0:
        print(f"\n{BOLD}Mensajes de error completos para este hash:{RESET}")
        for i, msg in enumerate(mensajes, 1):
            print(f"\n{RED}--- Error #{i} ---{RESET}\n{YELLOW}{msg}{RESET}")
    else:
        print(f"\n{GREEN}No hay mensajes de error para este hash.{RESET}")


def main():
    if len(sys.argv) < 2:
        print(f"{RED}Uso: python analize_logs.py <archivo.csv>{RESET}")
        sys.exit(1)
    archivo = sys.argv[1]
    df = pd.read_csv(archivo)
    while True:
        print(f"\n{BOLD}Opciones:{RESET}")
        print(f"{CYAN}1.{RESET} Resumen general")
        print(f"{CYAN}2.{RESET} Ver errores agrupados por hash")
        print(f"{CYAN}3.{RESET} Resumen temporal por día")
        print(f"{CYAN}4.{RESET} Ver últimos registros")
        print(f"{CYAN}5.{RESET} Filtrar por hash")
        print(f"{CYAN}6.{RESET} Ver mensajes de error únicos")
        print(f"{CYAN}7.{RESET} Salir")
        op = input("Elige opción: ")
        if op == "1":
            resumen_general(df)
        elif op == "2":
            errores_por_hash(df)
        elif op == "3":
            resumen_temporal(df)
        elif op == "4":
            ultimos_registros(df)
        elif op == "5":
            h = input("Introduce el hash: ").strip()
            filtrado = df[df["hash"] == h]
            if filtrado.empty:
                print(f"{CYAN}No hay registros con ese hash.{RESET}")
            else:
                mostrar_resumen_hash(filtrado)
                print(
                    tabulate(filtrado, headers="keys", tablefmt="psql", showindex=False)
                )
        elif op == "6":
            mensajes_error_unicos(df)
        elif op == "7":
            break
        else:
            print(f"{RED}Opción no válida.{RESET}")


if __name__ == "__main__":
    main()
