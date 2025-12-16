import cv2
import numpy as np
import pyrealsense2 as rs
import socket

# =========================
# 1. CLASIFICACION HSV  (IGUAL QUE TU VERSION BUENA)
# =========================

def clasificar_color(h, s, v):
    """
    Devuelve (nombre, id) usando HSV.
    IDs:
      1: verde_claro
      2: rosa
      3: rojo_suave
      4: morado
      5: verde_oscuro
      6: naranja
      7: azul
      8: rojo_fuerte
    """
    nombre = "desconocido"
    id_ = 0

    # Filtro suave
    if v < 30 or s < 25:
        return nombre, id_

    # --- Rosa: rojo pastel ---
    if ((h <= 10 or h >= 170) and 60 <= s <= 160 and v >= 170):
        nombre = "rosa"
        id_ = 2

    # --- Naranja ---
    elif (10 < h <= 28 and s >= 110 and v >= 150):
        nombre = "naranja"
        id_ = 6

    # --- Azules / morado ---
    elif 90 <= h < 125:
        nombre = "azul"
        id_ = 7
    elif 125 <= h <= 160:
        nombre = "morado"
        id_ = 4

    # --- Verdes (refino luego por tamaño) ---
    elif 35 <= h < 85:
        nombre = "verde"
        id_ = 0

    # --- Rojos genericos (refino luego) ---
    elif (h <= 10 or h >= 170) and s > 150 and v > 90:
        nombre = "rojo"
        id_ = 0

    return nombre, id_


# =========================
# 2. REFINO VERDES/ROJOS  (IGUAL QUE TENÍAS)
# =========================

def refinar_por_tamano(circulos_info):
    # Verdes
    idx_verdes = [i for i, c in enumerate(circulos_info)
                  if c["nombre"] in ("verde", "verde_claro", "verde_oscuro")]
    if len(idx_verdes) >= 2:
        verdes_orden = sorted(idx_verdes, key=lambda i: circulos_info[i]["r"])
        idx_small = verdes_orden[0]
        circulos_info[idx_small]["nombre"] = "verde_claro"
        circulos_info[idx_small]["id"] = 1
        for i in verdes_orden[1:]:
            circulos_info[i]["nombre"] = "verde_oscuro"
            circulos_info[i]["id"] = 5

    # Cálidos
    idx_calidos = [i for i, c in enumerate(circulos_info)
                   if (c["h"] <= 25 or c["h"] >= 170)]

    if len(idx_calidos) >= 4:
        calidos_orden = sorted(idx_calidos, key=lambda i: circulos_info[i]["r"])
        mapping = [
            ("rosa",        2),
            ("rojo_suave",  3),
            ("naranja",     6),
            ("rojo_fuerte", 8)
        ]
        for idx, (nombre, ident) in zip(calidos_orden[0:4], mapping):
            circulos_info[idx]["nombre"] = nombre
            circulos_info[idx]["id"] = ident
    elif len(idx_calidos) >= 3:
        calidos_orden = sorted(idx_calidos, key=lambda i: circulos_info[i]["r"])
        small = calidos_orden[0]
        big   = calidos_orden[-1]
        circulos_info[small]["nombre"] = "rosa"
        circulos_info[small]["id"] = 2
        circulos_info[big]["nombre"] = "rojo_fuerte"
        circulos_info[big]["id"] = 8

    # Rosa vs rojo_suave por saturacion
    idx_sub = [i for i, c in enumerate(circulos_info)
               if c["nombre"] in ("rosa", "rojo_suave")]
    if len(idx_sub) >= 2:
        idx_rosa = min(idx_sub, key=lambda i: circulos_info[i]["s"])
        for i in idx_sub:
            if i == idx_rosa:
                circulos_info[i]["nombre"] = "rosa"
                circulos_info[i]["id"] = 2
            else:
                circulos_info[i]["nombre"] = "rojo_suave"
                circulos_info[i]["id"] = 3

    return circulos_info


# =========================
# 3. COLOR MEDIO EN CIRCULO  (IGUAL)
# =========================

def color_en_circulo(hsv, x, y, r):
    h_img, w_img = hsv.shape[:2]
    r_in = int(r * 0.7)

    x1 = max(x - r_in, 0)
    y1 = max(y - r_in, 0)
    x2 = min(x + r_in, w_img - 1)
    y2 = min(y + r_in, h_img - 1)

    patch = hsv[y1:y2, x1:x2]
    if patch.size == 0:
        return np.nan, np.nan, np.nan

    mask = np.zeros(patch.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (x - x1, y - y1), r_in, 255, -1)

    h_chan, s_chan, v_chan = cv2.split(patch)
    h_vals = h_chan[mask == 255]
    s_vals = s_chan[mask == 255]
    v_vals = v_chan[mask == 255]

    if len(h_vals) == 0:
        return np.nan, np.nan, np.nan

    return float(np.median(h_vals)), float(np.median(s_vals)), float(np.median(v_vals))


# =========================
# 3b. ASIGNAR CELDAS 11–14 / 21–24
# =========================

def asignar_celdas(circulos_info):
    """
    Usa la posición (x,y) para asignar celdas:
      fila de arriba: 11,12,13,14 (izq->der)
      fila de abajo: 21,22,23,24 (izq->der)
    """
    if len(circulos_info) < 8:
        print("⚠️ Ojo: se detectaron menos de 8 círculos, no se asignan todas las celdas.")
        return circulos_info

    idx_por_y = sorted(range(len(circulos_info)), key=lambda i: circulos_info[i]["y"])

    fila_arriba = idx_por_y[:4]
    fila_abajo  = idx_por_y[4:8]

    fila_arriba = sorted(fila_arriba, key=lambda i: circulos_info[i]["x"])
    fila_abajo  = sorted(fila_abajo,  key=lambda i: circulos_info[i]["x"])

    celdas_arriba = [11, 12, 13, 14]
    celdas_abajo  = [21, 22, 23, 24]

    for idx, celda in zip(fila_arriba, celdas_arriba):
        circulos_info[idx]["celda"] = celda

    for idx, celda in zip(fila_abajo, celdas_abajo):
        circulos_info[idx]["celda"] = celda

    print("Resumen por círculo (id, radio, celda, centro):")
    for c in circulos_info:
        print(f"  id={c['id']}  r={c['r']:.1f}  celda={c.get('celda')}  center=({c['x']},{c['y']})")

    return circulos_info


# =========================
# 4. PROCESAR FRAME
# =========================

def procesar_frame(frame_bgr):
    src = frame_bgr.copy()
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    rows = gray.shape[0]
    minR = int(rows * 0.05)
    maxR = int(rows * 0.18)

    mejor_lista = []

    for param2 in [60, 55, 50, 45, 40, 35]:
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=rows / 8,
            param1=120,
            param2=param2,
            minRadius=minR,
            maxRadius=maxR
        )

        if circles is None:
            continue

        circles = np.uint16(np.around(circles[0]))
        candidatos = []

        for (x, y, r) in circles:
            es_nuevo = True
            for (xx, yy, rr) in candidatos:
                if np.hypot(x - xx, y - yy) < 0.4 * r and abs(r - rr) < 0.3 * r:
                    es_nuevo = False
                    break
            if es_nuevo:
                candidatos.append((int(x), int(y), int(r)))

        if len(candidatos) >= 8:
            mejor_lista = candidatos
            break
        if len(candidatos) > len(mejor_lista):
            mejor_lista = candidatos

    print(f"[INFO] Círculos finales en frame: {len(mejor_lista)}")

    hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    circulos_info = []

    for (x, y, r) in mejor_lista:
        h_med, s_med, v_med = color_en_circulo(hsv, x, y, r)
        if np.isnan(h_med):
            continue
        nombre, id_color = clasificar_color(h_med, s_med, v_med)
        circulos_info.append({
            "x": x, "y": y, "r": float(r),
            "h": h_med, "s": s_med, "v": v_med,
            "nombre": nombre, "id": id_color
        })

    circulos_info = refinar_por_tamano(circulos_info)
    circulos_info = asignar_celdas(circulos_info)

    for c in circulos_info:
        x, y, r = int(c["x"]), int(c["y"]), int(c["r"])
        nombre, id_ = c["nombre"], c["id"]

        cv2.circle(src, (x, y), r, (255, 0, 255), 3)
        cv2.circle(src, (x, y), 3, (0, 255, 255), -1)

        etiqueta = f"{nombre} #{id_}" if id_ != 0 else nombre
        cv2.putText(
            src, etiqueta,
            (x - r + 5, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
            (255, 255, 255), 2, cv2.LINE_AA
        )

        print(f" -> círculo: center=({x},{y}), r={r}, "
              f"H={c['h']:.1f}, S={c['s']:.1f}, V={c['v']:.1f} => {etiqueta}")

    return src, circulos_info


# =========================
# 5. CALCULAR SECUENCIA PARA EL ROBOT
# =========================

# Mapa celda -> comando (Rapid)
MAP_CELDA_TO_CMD = {
    11: "1",  # Pick_11
    12: "2",  # Pick_12
    13: "3",  # Pick_13
    14: "4",  # Pick_14
    21: "5",  # Pick_21
    22: "6",  # Pick_22
    23: "7",  # Pick_23
    24: "8",  # Pick_24
}

def calcular_secuencia_robot(circulos_info):
    # Solo con color válido y celda asignada
    validos = [
        c for c in circulos_info
        if 1 <= c["id"] <= 8 and "celda" in c
    ]

    # *** ORDEN ÚNICO POR ID: 8,7,6,5,4,3,2,1 ***
    ordenados = sorted(validos, key=lambda c: -c["id"])

    ids = [c["id"] for c in ordenados]
    radios = [c["r"] for c in ordenados]
    celdas = [c["celda"] for c in ordenados]

    print("IDs (8->1):", ids)
    print("Radios correspondientes:", radios)
    print("Celdas correspondientes:", celdas)

    comandos = [MAP_CELDA_TO_CMD[c] for c in celdas]
    secuencia = "".join(comandos)

    print("Secuencia de comandos (1..8) para el robot:", secuencia)
    return secuencia


# =========================
# 6. ENVIAR SECUENCIA AL ROBOT
# =========================

def enviar_secuencia_robot(secuencia, host="192.168.125.1", port=5000):
    mensaje = secuencia + "9"   # '9' para que el RAPID cierre el socket

    print(f"Enviando al robot: '{mensaje}'")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, port))
        s.sendall(mensaje.encode("ascii"))


# =========================
# 7. BUCLE CON REALSENSE
# =========================

def main_realsense():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    pipeline.start(config)
    print("RealSense iniciado. Ventanas:")
    print("  - 'Preview': imagen en vivo")
    print("Teclas:")
    print("  - 'c': capturar frame y clasificar aros")
    print("  - 'r': enviar secuencia al robot (con la última clasificación)")
    print("  - 'q' o ESC: salir")

    circulos_ultima = None

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())

            cv2.imshow('Preview', color_image)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('c'):
                print("\n===== Captura para clasificación =====")
                resultado, circulos_info = procesar_frame(color_image)
                circulos_ultima = circulos_info
                cv2.imshow('Aros detectados y clasificados', resultado)
                cv2.waitKey(1)

            elif key == ord('r'):
                if circulos_ultima is None:
                    print("⚠️ Primero pulsa 'c' para clasificar.")
                else:
                    sec = calcular_secuencia_robot(circulos_ultima)
                    enviar_secuencia_robot(sec)

            elif key == ord('q') or key == 27:
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("Pipeline detenido. Fin.")


if __name__ == "__main__":
    main_realsense()
