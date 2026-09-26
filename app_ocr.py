# ============================================================
#  OCR Studio  ·  Comparador de Thresholding
#  --------------------------------------------------------
#  Una aplicación para comparar visualmente el thresholding
#  global (Otsu) y local (adaptativo) aplicado a OCR.
#
#  Diseñada para ser clara, precisa y agradable de usar.
# ============================================================

import cv2
import numpy as np
import pytesseract
import os
import sys
import threading
import time
from datetime import datetime

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk

try:
    from jiwer import cer, wer
    JIWER_OK = True
except ImportError:
    JIWER_OK = False

# ------------------------------------------------------------
#  CONFIGURACIÓN DE TESSERACT
#  Si tu instalación está en otra ruta, cámbiala aquí.
# ------------------------------------------------------------
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# ============================================================
#  PALETA DE COLORES (estilo Material Design)
# ============================================================
COLOR_FONDO      = "#F5F7FA"
COLOR_PANEL      = "#FFFFFF"
COLOR_PRIMARIO   = "#1976D2"   # azul
COLOR_EXITO      = "#2E7D32"   # verde
COLOR_ADVERT     = "#F57C00"   # naranja
COLOR_PELIGRO    = "#C62828"   # rojo
COLOR_TEXTO      = "#263238"   # gris oscuro
COLOR_SECUNDARIO = "#546E7A"   # gris azulado
COLOR_BORDE      = "#CFD8DC"   # gris claro


# ============================================================
#  FUNCIONES DE PROCESAMIENTO
# ============================================================

def suavizar(img):
    """Aplica un pequeño desenfoque para reducir ruido."""
    return cv2.GaussianBlur(img, (3, 3), 0)


def threshold_otsu(img):
    """Binariza con el método global de Otsu. Retorna (imagen, umbral)."""
    umbral, binaria = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binaria, umbral


def threshold_adaptativo(img, block_size=31, C=10):
    """Binariza con umbral local adaptativo (media gaussiana)."""
    return cv2.adaptiveThreshold(
        img, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=block_size,
        C=C
    )


def asegurar_texto_oscuro(img_bin):
    """Tesseract lee mejor texto oscuro sobre fondo claro."""
    if np.sum(img_bin == 0) > np.sum(img_bin == 255):
        return cv2.bitwise_not(img_bin)
    return img_bin


def extraer_texto(img_bin, idioma="spa"):
    img_ok = asegurar_texto_oscuro(img_bin)
    try:
        return pytesseract.image_to_string(img_ok, lang=idioma, config="--psm 6").strip()
    except pytesseract.TesseractError:
        return pytesseract.image_to_string(img_ok, lang="eng", config="--psm 6").strip()


def limpiar_texto(texto):
    return " ".join(texto.split())


def metricas_imagen(img):
    """Métricas útiles para describir la imagen."""
    hist = cv2.calcHist([img], [0], None, [256], [0, 256]).ravel()
    p = hist / hist.sum()
    p = p[p > 0]
    return {
        "brillo": float(np.mean(img)),
        "contraste": float(np.std(img)),
        "entropia": float(-np.sum(p * np.log2(p))),
        "nitidez": float(cv2.Laplacian(img, cv2.CV_64F).var()),
    }


# ============================================================
#  TOOLTIP (ayuda emergente)
# ============================================================
class Tooltip:
    def __init__(self, widget, texto):
        self.widget = widget
        self.texto = texto
        self.tip = None
        widget.bind("<Enter>", self._mostrar)
        widget.bind("<Leave>", self._ocultar)

    def _mostrar(self, _=None):
        if self.tip:
            return
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 5
        self.tip = tk.Toplevel(self.widget)
        self.tip.wm_overrideredirect(True)
        self.tip.wm_geometry(f"+{x}+{y}")
        tk.Label(self.tip, text=self.texto, bg="#FFF9C4", fg=COLOR_TEXTO,
                 font=("Segoe UI", 9), padx=8, pady=4,
                 relief="solid", borderwidth=1).pack()

    def _ocultar(self, _=None):
        if self.tip:
            self.tip.destroy()
            self.tip = None


# ============================================================
#  APLICACIÓN PRINCIPAL
# ============================================================
class OCRStudio:
    def __init__(self, root):
        self.root = root
        self.root.title("OCR Studio  ·  Comparador de Thresholding")
        self.root.geometry("1400x860")
        self.root.minsize(1100, 700)
        self.root.configure(bg=COLOR_FONDO)

        # Estado
        self.ruta = None
        self.img_original = None
        self.img_otsu = None
        self.img_adapt = None
        self.texto_otsu = ""
        self.texto_adapt = ""
        self.umbral_otsu = 0
        self.t_otsu = 0.0
        self.t_adapt = 0.0
        self.metricas = {}

        self._aplicar_estilos()
        self._construir()

    # ------------------------------------------------------------
    def _aplicar_estilos(self):
        s = ttk.Style()
        s.theme_use("clam")
        s.configure("TNotebook", background=COLOR_FONDO, borderwidth=0)
        s.configure("TNotebook.Tab", font=("Segoe UI", 10, "bold"),
                    padding=[14, 8], background=COLOR_PANEL, foreground=COLOR_SECUNDARIO)
        s.map("TNotebook.Tab",
              background=[("selected", COLOR_PRIMARIO)],
              foreground=[("selected", "white")])
        s.configure("TProgressbar", troughcolor=COLOR_BORDE,
                    background=COLOR_PRIMARIO, thickness=6)

    # ------------------------------------------------------------
    def _construir(self):
        # --- Encabezado ---
        header = tk.Frame(self.root, bg=COLOR_PRIMARIO, height=70)
        header.pack(fill=tk.X)
        header.pack_propagate(False)

        tk.Label(header, text="🔬  OCR Studio",
                 font=("Segoe UI", 18, "bold"),
                 bg=COLOR_PRIMARIO, fg="white").pack(side=tk.LEFT, padx=20)
        tk.Label(header, text="Compara thresholding global (Otsu) y local (adaptativo)",
                 font=("Segoe UI", 10),
                 bg=COLOR_PRIMARIO, fg="#E3F2FD").pack(side=tk.LEFT, padx=10)

        # --- Barra de botones ---
        self._crear_barra_acciones()

        # --- Panel de parámetros ---
        self._crear_panel_parametros()

        # --- Pestañas ---
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=15, pady=(5, 10))

        self.tab_visual = tk.Frame(self.notebook, bg=COLOR_FONDO)
        self.tab_metricas = tk.Frame(self.notebook, bg=COLOR_FONDO)
        self.tab_texto = tk.Frame(self.notebook, bg=COLOR_FONDO)
        self.tab_ayuda = tk.Frame(self.notebook, bg=COLOR_FONDO)

        self.notebook.add(self.tab_visual, text="🖼️  Comparación")
        self.notebook.add(self.tab_metricas, text="📊  Métricas")
        self.notebook.add(self.tab_texto, text="📝  Texto OCR")
        self.notebook.add(self.tab_ayuda, text="❓  Ayuda")

        self._construir_tab_visual()
        self._construir_tab_metricas()
        self._construir_tab_texto()
        self._construir_tab_ayuda()

        # --- Barra de estado ---
        self.status = tk.Label(self.root, text="👋  ¡Bienvenido! Comienza cargando una imagen.",
                               bg=COLOR_SECUNDARIO, fg="white", anchor="w",
                               padx=15, pady=6, font=("Segoe UI", 9))
        self.status.pack(fill=tk.X, side=tk.BOTTOM)

    # ------------------------------------------------------------
    def _crear_barra_acciones(self):
        barra = tk.Frame(self.root, bg=COLOR_FONDO, pady=10)
        barra.pack(fill=tk.X, padx=15)

        def boton(texto, color, comando, estado="normal", tip=None):
            b = tk.Button(barra, text=texto, font=("Segoe UI", 10, "bold"),
                          bg=color, fg="white", padx=16, pady=8,
                          activebackground=color, activeforeground="white",
                          relief="flat", cursor="hand2",
                          command=comando, state=estado)
            b.pack(side=tk.LEFT, padx=4)
            if tip:
                Tooltip(b, tip)
            return b

        boton("📂  Cargar imagen", COLOR_PRIMARIO, self.cargar_imagen,
              tip="Selecciona una imagen (PNG, JPG, BMP, TIFF) para analizar.")
        self.btn_procesar = boton("⚡  Procesar", COLOR_EXITO, self.procesar,
                                  estado=tk.DISABLED,
                                  tip="Aplica ambos thresholdings y ejecuta OCR.")
        boton("💾  Guardar resultados", COLOR_ADVERT, self.guardar_resultados,
              tip="Guarda las imágenes y los textos OCR en una carpeta.")
        boton("📄  Informe HTML", "#6A1B9A", self.exportar_informe,
              tip="Genera un informe listo para imprimir o adjuntar.")

        # Botón salir a la derecha
        b_salir = tk.Button(barra, text="✖  Salir", font=("Segoe UI", 10),
                            bg=COLOR_PELIGRO, fg="white", padx=16, pady=8,
                            activebackground=COLOR_PELIGRO, activeforeground="white",
                            relief="flat", cursor="hand2", command=self.root.quit)
        b_salir.pack(side=tk.RIGHT, padx=4)

    # ------------------------------------------------------------
    def _crear_panel_parametros(self):
        panel = tk.Frame(self.root, bg=COLOR_PANEL,
                         highlightbackground=COLOR_BORDE, highlightthickness=1)
        panel.pack(fill=tk.X, padx=15, pady=(0, 8))

        tk.Label(panel, text="⚙️  Ajustes del método adaptativo",
                 font=("Segoe UI", 10, "bold"),
                 bg=COLOR_PANEL, fg=COLOR_TEXTO).pack(side=tk.LEFT, padx=(15, 20), pady=8)

        # Block Size
        tk.Label(panel, text="Tamaño de vecindad:", bg=COLOR_PANEL,
                 font=("Segoe UI", 9), fg=COLOR_SECUNDARIO).pack(side=tk.LEFT)
        self.var_block = tk.IntVar(value=31)
        sp_block = tk.Spinbox(panel, from_=3, to=99, increment=2,
                              textvariable=self.var_block, width=5,
                              font=("Segoe UI", 9), relief="flat",
                              highlightbackground=COLOR_BORDE, highlightthickness=1)
        sp_block.pack(side=tk.LEFT, padx=5)
        Tooltip(sp_block, "Tamaño de la ventana local. Valores mayores = más suave.")

        # C
        tk.Label(panel, text="Ajuste fino (C):", bg=COLOR_PANEL,
                 font=("Segoe UI", 9), fg=COLOR_SECUNDARIO).pack(side=tk.LEFT, padx=(15, 0))
        self.var_C = tk.IntVar(value=10)
        sp_C = tk.Spinbox(panel, from_=1, to=50, textvariable=self.var_C, width=5,
                          font=("Segoe UI", 9), relief="flat",
                          highlightbackground=COLOR_BORDE, highlightthickness=1)
        sp_C.pack(side=tk.LEFT, padx=5)
        Tooltip(sp_C, "Constante restada a la media local. Más alto = menos sensible al ruido.")

        # Idioma
        tk.Label(panel, text="Idioma OCR:", bg=COLOR_PANEL,
                 font=("Segoe UI", 9), fg=COLOR_SECUNDARIO).pack(side=tk.LEFT, padx=(15, 0))
        self.var_idioma = tk.StringVar(value="spa")
        cb_idioma = ttk.Combobox(panel, textvariable=self.var_idioma,
                                 values=["spa", "eng", "spa+eng"],
                                 width=8, state="readonly")
        cb_idioma.pack(side=tk.LEFT, padx=5)
        Tooltip(cb_idioma, "Idioma del texto en la imagen.")

    # ------------------------------------------------------------
    def _construir_tab_visual(self):
        marco = tk.Frame(self.tab_visual, bg=COLOR_FONDO)
        marco.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        for i in range(3):
            marco.columnconfigure(i, weight=1)
        marco.rowconfigure(0, weight=1)

        self.cv_orig = self._tarjeta_imagen(marco, "🖼️  Imagen original", 0)
        self.cv_otsu = self._tarjeta_imagen(marco, "🌐  Otsu (global)", 1)
        self.cv_adapt = self._tarjeta_imagen(marco, "🔍  Adaptativo (local)", 2)

        # Etiqueta inferior con recomendación
        self.lbl_recomendacion = tk.Label(
            self.tab_visual, text="",
            font=("Segoe UI", 10, "italic"), bg=COLOR_FONDO, fg=COLOR_SECUNDARIO,
            pady=8
        )
        self.lbl_recomendacion.pack(fill=tk.X)

    def _tarjeta_imagen(self, parent, titulo, col):
        card = tk.Frame(parent, bg=COLOR_PANEL,
                        highlightbackground=COLOR_BORDE, highlightthickness=1)
        card.grid(row=0, column=col, sticky="nsew", padx=6, pady=6)

        tk.Label(card, text=titulo, font=("Segoe UI", 10, "bold"),
                 bg=COLOR_PANEL, fg=COLOR_TEXTO, pady=8).pack(fill=tk.X)

        canvas = tk.Canvas(card, bg="#ECEFF1", highlightthickness=0)
        canvas.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))
        return canvas

    # ------------------------------------------------------------
    def _construir_tab_metricas(self):
        self.lbl_metricas = tk.Label(
            self.tab_metricas,
            text="Aún no hay métricas.\n\nProcesa una imagen para verlas aquí.",
            font=("Consolas", 10), bg=COLOR_FONDO, fg=COLOR_TEXTO,
            justify=tk.LEFT, anchor="nw", padx=20, pady=20
        )
        self.lbl_metricas.pack(fill=tk.BOTH, expand=True)

    # ------------------------------------------------------------
    def _construir_tab_texto(self):
        marco = tk.Frame(self.tab_texto, bg=COLOR_FONDO)
        marco.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # --- Ground truth ---
        gt_card = tk.Frame(marco, bg=COLOR_PANEL,
                           highlightbackground=COLOR_BORDE, highlightthickness=1)
        gt_card.pack(fill=tk.X, pady=(0, 10))

        tk.Label(gt_card, text="🎯  Texto real esperado (opcional)",
                 font=("Segoe UI", 10, "bold"),
                 bg=COLOR_PANEL, fg=COLOR_TEXTO).pack(anchor="w", padx=12, pady=(8, 2))
        tk.Label(gt_card, text="Escríbelo aquí para calcular la precisión del OCR (CER/WER).",
                 font=("Segoe UI", 9, "italic"),
                 bg=COLOR_PANEL, fg=COLOR_SECUNDARIO).pack(anchor="w", padx=12)

        self.txt_gt = scrolledtext.ScrolledText(gt_card, height=3, wrap=tk.WORD,
                                                font=("Consolas", 9), relief="flat",
                                                highlightbackground=COLOR_BORDE,
                                                highlightthickness=1)
        self.txt_gt.pack(fill=tk.X, padx=12, pady=6)

        btn_calc = tk.Button(gt_card, text="📊  Calcular CER / WER",
                             font=("Segoe UI", 9, "bold"),
                             bg=COLOR_EXITO, fg="white", padx=12, pady=5,
                             relief="flat", cursor="hand2",
                             command=self.calcular_cer_wer)
        btn_calc.pack(padx=12, pady=(0, 10), anchor="w")
        Tooltip(btn_calc, "Compara el texto OCR con el ground truth.")

        # --- Resultados OCR lado a lado ---
        ocr = tk.Frame(marco, bg=COLOR_FONDO)
        ocr.pack(fill=tk.BOTH, expand=True)
        ocr.columnconfigure(0, weight=1)
        ocr.columnconfigure(1, weight=1)

        self.txt_otsu = self._tarjeta_texto(ocr, "🌐  Texto detectado con OTSU", 0, COLOR_PRIMARIO)
        self.txt_adapt = self._tarjeta_texto(ocr, "🔍  Texto detectado con ADAPTATIVO", 1, COLOR_ADVERT)

    def _tarjeta_texto(self, parent, titulo, col, color_titulo):
        card = tk.Frame(parent, bg=COLOR_PANEL,
                        highlightbackground=COLOR_BORDE, highlightthickness=1)
        card.grid(row=0, column=col, sticky="nsew", padx=6)

        tk.Label(card, text=titulo, font=("Segoe UI", 10, "bold"),
                 bg=COLOR_PANEL, fg=color_titulo).pack(anchor="w", padx=12, pady=(10, 4))

        txt = scrolledtext.ScrolledText(card, wrap=tk.WORD, font=("Consolas", 9),
                                        relief="flat",
                                        highlightbackground=COLOR_BORDE,
                                        highlightthickness=1)
        txt.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 12))
        return txt

    # ------------------------------------------------------------
    def _construir_tab_ayuda(self):
        marco = tk.Frame(self.tab_ayuda, bg=COLOR_FONDO)
        marco.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        ayuda = (
            "¿Qué hace esta aplicación?\n\n"
            "  Esta herramienta compara dos formas de convertir una imagen en blanco y negro\n"
            "  (binarización) antes de aplicar reconocimiento óptico de caracteres (OCR).\n\n"
            "Métodos comparados:\n\n"
            "  🌐  Otsu (global)\n"
            "      • Calcula UN solo umbral para toda la imagen.\n"
            "      • Funciona muy bien si la iluminación es uniforme.\n"
            "      • Es muy rápido.\n"
            "      • Falla cuando hay sombras o iluminación desigual.\n\n"
            "  🔍  Adaptativo (local)\n"
            "      • Calcula un umbral DISTINTO para cada zona de la imagen.\n"
            "      • Excelente para fotos con sombras o iluminación desigual.\n"
            "      • Más lento que Otsu.\n"
            "      • Puede generar algo de ruido en zonas planas.\n\n"
            "Consejo:\n\n"
            "  Si tu imagen es un escaneo limpio → Otsu suele bastar.\n"
            "  Si es una foto con sombras → el Adaptativo casi siempre gana.\n\n"
            "Sobre las métricas:\n\n"
            "  • CER: porcentaje de caracteres mal reconocidos.\n"
            "  • WER: porcentaje de palabras mal reconocidas.\n"
            "  Mientras más bajos, mejor.\n"
        )

        tk.Label(marco, text=ayuda, font=("Segoe UI", 10),
                 bg=COLOR_FONDO, fg=COLOR_TEXTO, justify=tk.LEFT,
                 anchor="nw").pack(fill=tk.BOTH, expand=True)

    # ============================================================
    #  ACCIONES
    # ============================================================
    def cargar_imagen(self):
        ruta = filedialog.askopenfilename(
            title="Selecciona una imagen para analizar",
            filetypes=[("Imágenes", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                       ("Todos los archivos", "*.*")]
        )
        if not ruta:
            return

        img = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)
        if img is None:
            messagebox.showerror("No se pudo abrir",
                                 "El archivo seleccionado no es una imagen válida.")
            return

        self.ruta = ruta
        self.img_original = img
        self.metricas = metricas_imagen(img)

        self._mostrar(self.cv_orig, img)
        self.cv_otsu.delete("all")
        self.cv_adapt.delete("all")
        self.txt_otsu.delete("1.0", tk.END)
        self.txt_adapt.delete("1.0", tk.END)
        self.lbl_metricas.config(text="Procesa la imagen para ver las métricas.")
        self.lbl_recomendacion.config(text="")

        nombre = os.path.basename(ruta)
        alto, ancho = img.shape
        self.status.config(text=f"✅  Imagen cargada: {nombre}  ·  {ancho}×{alto} px")
        self.btn_procesar.config(state=tk.NORMAL)

    # ------------------------------------------------------------
    def procesar(self):
        if self.img_original is None:
            return
        self.btn_procesar.config(state=tk.DISABLED)
        self.status.config(text="⏳  Procesando… esto puede tardar unos segundos.")
        self.root.update_idletasks()
        threading.Thread(target=self._procesar_en_hilo, daemon=True).start()

    def _procesar_en_hilo(self):
        try:
            base = suavizar(self.img_original)

            t0 = time.time()
            self.img_otsu, self.umbral_otsu = threshold_otsu(base)
            self.t_otsu = time.time() - t0

            t0 = time.time()
            self.img_adapt = threshold_adaptativo(
                base, block_size=self.var_block.get(), C=self.var_C.get()
            )
            self.t_adapt = time.time() - t0

            idioma = self.var_idioma.get()
            self.texto_otsu = extraer_texto(self.img_otsu, idioma)
            self.texto_adapt = extraer_texto(self.img_adapt, idioma)

            self.root.after(0, self._refrescar_ui)
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Algo salió mal", str(e)))
            self.root.after(0, lambda: self.btn_procesar.config(state=tk.NORMAL))

    def _refrescar_ui(self):
        self._mostrar(self.cv_otsu, self.img_otsu)
        self._mostrar(self.cv_adapt, self.img_adapt)

        self.txt_otsu.delete("1.0", tk.END)
        self.txt_otsu.insert(tk.END, self.texto_otsu or "[No se detectó texto]")
        self.txt_adapt.delete("1.0", tk.END)
        self.txt_adapt.insert(tk.END, self.texto_adapt or "[No se detectó texto]")

        self._actualizar_metricas()
        self._actualizar_recomendacion()

        self.btn_procesar.config(state=tk.NORMAL)
        self.status.config(
            text=f"✅  Procesamiento completado  ·  Otsu: {self.t_otsu*1000:.1f} ms  ·  "
                 f"Adaptativo: {self.t_adapt*1000:.1f} ms"
        )

    # ------------------------------------------------------------
    def _mostrar(self, canvas, img):
        canvas.update_idletasks()
        w = max(canvas.winfo_width(), 200)
        h = max(canvas.winfo_height(), 200)
        alto, ancho = img.shape[:2]
        escala = min(w / ancho, h / alto)
        nw, nh = max(1, int(ancho * escala)), max(1, int(alto * escala))
        redim = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        if redim.ndim == 2:
            redim = cv2.cvtColor(redim, cv2.COLOR_GRAY2RGB)
        else:
            redim = cv2.cvtColor(redim, cv2.COLOR_BGR2RGB)
        foto = ImageTk.PhotoImage(Image.fromarray(redim))
        canvas.delete("all")
        canvas.create_image(w // 2, h // 2, image=foto, anchor=tk.CENTER)
        canvas.image = foto

    # ------------------------------------------------------------
    def _actualizar_metricas(self):
        m = self.metricas
        txt = (
            f"{'─'*70}\n"
            f"  📷  IMAGEN ORIGINAL\n"
            f"{'─'*70}\n"
            f"     Brillo medio        : {m['brillo']:.1f}\n"
            f"     Contraste (σ)       : {m['contraste']:.1f}\n"
            f"     Entropía (bits)     : {m['entropia']:.3f}\n"
            f"     Nitidez (Laplaciano): {m['nitidez']:.1f}\n\n"
            f"{'─'*70}\n"
            f"  ⏱️  TIEMPOS DE PROCESAMIENTO\n"
            f"{'─'*70}\n"
            f"     Otsu (global)       : {self.t_otsu*1000:.1f} ms\n"
            f"     Adaptativo (local)  : {self.t_adapt*1000:.1f} ms\n"
            f"     Umbral Otsu         : {int(self.umbral_otsu)}\n\n"
            f"{'─'*70}\n"
            f"  📝  OCR\n"
            f"{'─'*70}\n"
            f"     Caracteres (Otsu)       : {len(self.texto_otsu)}\n"
            f"     Caracteres (Adaptativo) : {len(self.texto_adapt)}\n"
        )
        self.lbl_metricas.config(text=txt)

    # ------------------------------------------------------------
    def _actualizar_recomendacion(self):
        contraste = self.metricas["contraste"]
        if contraste < 35:
            msg = "⚠️  La imagen tiene bajo contraste. El thresholding ADAPTATIVO suele dar mejores resultados."
            color = COLOR_ADVERT
        elif contraste > 70:
            msg = "✅  La imagen tiene buen contraste. OTSU probablemente sea suficiente."
            color = COLOR_EXITO
        else:
            msg = "ℹ️  Contraste moderado. Compara visualmente ambos resultados."
            color = COLOR_SECUNDARIO
        self.lbl_recomendacion.config(text=msg, fg=color)

    # ------------------------------------------------------------
    def calcular_cer_wer(self):
        if not JIWER_OK:
            messagebox.showwarning("Falta jiwer",
                                   "Para calcular CER/WER instala 'jiwer':\n\npip install jiwer")
            return
        gt = limpiar_texto(self.txt_gt.get("1.0", tk.END))
        if not gt:
            messagebox.showinfo("Falta información",
                                "Escribe el texto real esperado (ground truth) para comparar.")
            return
        if not (self.texto_otsu or self.texto_adapt):
            messagebox.showinfo("Sin resultados", "Primero procesa una imagen.")
            return

        t_o = limpiar_texto(self.texto_otsu)
        t_a = limpiar_texto(self.texto_adapt)

        cer_o, wer_o = cer(gt, t_o), wer(gt, t_o)
        cer_a, wer_a = cer(gt, t_a), wer(gt, t_a)

        mejor = "OTSU" if cer_o < cer_a else "ADAPTATIVO"

        resultado = (
            f"{'─'*50}\n"
            f"   COMPARACIÓN CON EL TEXTO REAL\n"
            f"{'─'*50}\n\n"
            f"  🌐  OTSU\n"
            f"        CER : {cer_o:.4f}\n"
            f"        WER : {wer_o:.4f}\n\n"
            f"  🔍  ADAPTATIVO\n"
            f"        CER : {cer_a:.4f}\n"
            f"        WER : {wer_a:.4f}\n\n"
            f"{'─'*50}\n"
            f"  🏆  MEJOR MÉTODO: {mejor}\n"
            f"{'─'*50}\n\n"
            f"  (CER y WER más bajos significan mejor precisión)"
        )
        messagebox.showinfo("Resultados de precisión", resultado)

    # ------------------------------------------------------------
    def guardar_resultados(self):
        if self.img_otsu is None:
            messagebox.showinfo("Sin resultados", "Primero procesa una imagen.")
            return
        carpeta = filedialog.askdirectory(title="Elige dónde guardar los resultados")
        if not carpeta:
            return

        base = os.path.splitext(os.path.basename(self.ruta))[0]
        cv2.imwrite(os.path.join(carpeta, f"{base}_original.png"), self.img_original)
        cv2.imwrite(os.path.join(carpeta, f"{base}_otsu.png"), self.img_otsu)
        cv2.imwrite(os.path.join(carpeta, f"{base}_adaptativo.png"), self.img_adapt)

        with open(os.path.join(carpeta, f"{base}_texto_otsu.txt"), "w", encoding="utf-8") as f:
            f.write(self.texto_otsu)
        with open(os.path.join(carpeta, f"{base}_texto_adaptativo.txt"), "w", encoding="utf-8") as f:
            f.write(self.texto_adapt)

        messagebox.showinfo("Guardado", f"✅  Los resultados se guardaron en:\n{carpeta}")

    # ------------------------------------------------------------
    def exportar_informe(self):
        if self.img_otsu is None:
            messagebox.showinfo("Sin resultados", "Primero procesa una imagen.")
            return
        carpeta = filedialog.askdirectory(title="Elige dónde guardar el informe")
        if not carpeta:
            return

        base = os.path.splitext(os.path.basename(self.ruta))[0]
        cv2.imwrite(os.path.join(carpeta, "original.png"), self.img_original)
        cv2.imwrite(os.path.join(carpeta, "otsu.png"), self.img_otsu)
        cv2.imwrite(os.path.join(carpeta, "adaptativo.png"), self.img_adapt)

        html = f"""<!DOCTYPE html>
<html lang="es"><head><meta charset="UTF-8">
<title>Informe OCR - {base}</title>
<style>
    body {{ font-family: 'Segoe UI', Arial, sans-serif; background:#F5F7FA; padding:30px; color:#263238; }}
    h1 {{ color:#1976D2; border-bottom:3px solid #1976D2; padding-bottom:8px; }}
    h2 {{ color:#263238; margin-top:30px; }}
    .grid {{ display:grid; grid-template-columns:1fr 1fr 1fr; gap:15px; }}
    .card {{ background:white; border-radius:8px; padding:12px; box-shadow:0 2px 8px rgba(0,0,0,0.08); }}
    .card img {{ width:100%; border-radius:5px; }}
    .card h3 {{ text-align:center; color:#263238; margin:6px 0 10px; }}
    pre {{ background:#263238; color:#ECEFF1; padding:15px; border-radius:6px; overflow-x:auto; font-family:Consolas,monospace; font-size:13px; }}
    .info {{ font-size:14px; line-height:1.7; }}
</style></head><body>
<h1>🔬 Informe de Comparación OCR</h1>
<p class="info"><strong>Archivo:</strong> {base}<br>
<strong>Fecha:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

<h2>🖼️ Comparación visual</h2>
<div class="grid">
  <div class="card"><h3>Original</h3><img src="original.png"></div>
  <div class="card"><h3>Otsu (global)</h3><img src="otsu.png"></div>
  <div class="card"><h3>Adaptativo (local)</h3><img src="adaptativo.png"></div>
</div>

<h2>📊 Métricas</h2>
<div class="card"><pre>
Brillo medio        : {self.metricas.get('brillo',0):.1f}
Contraste (σ)       : {self.metricas.get('contraste',0):.1f}
Entropía (bits)     : {self.metricas.get('entropia',0):.3f}
Nitidez (Laplaciano): {self.metricas.get('nitidez',0):.1f}
Umbral Otsu         : {int(self.umbral_otsu)}
Tiempo Otsu         : {self.t_otsu*1000:.1f} ms
Tiempo Adaptativo   : {self.t_adapt*1000:.1f} ms
</pre></div>

<h2>📝 Texto detectado con OTSU</h2>
<div class="card"><pre>{self.texto_otsu or '[Sin texto]'}</pre></div>

<h2>📝 Texto detectado con ADAPTATIVO</h2>
<div class="card"><pre>{self.texto_adapt or '[Sin texto]'}</pre></div>

<p style="text-align:center;color:#90A4AE;margin-top:40px;">
  Generado por OCR Studio · {datetime.now().year}
</p>
</body></html>"""

        ruta_html = os.path.join(carpeta, f"informe_{base}.html")
        with open(ruta_html, "w", encoding="utf-8") as f:
            f.write(html)

        messagebox.showinfo("Informe generado", f"✅  Informe creado en:\n{ruta_html}")
        try:
            os.startfile(ruta_html)  # Windows
        except AttributeError:
            pass


# ============================================================
#  EJECUCIÓN
# ============================================================
if __name__ == "__main__":
    root = tk.Tk()
    app = OCRStudio(root)
    root.mainloop()