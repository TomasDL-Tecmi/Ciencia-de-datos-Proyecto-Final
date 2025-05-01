# ============================== [ Librerias ] ==============================

import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import customtkinter as ctk
from tkinter import filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
from fpdf import FPDF
import io
from PIL import Image

# Configuración del tema de CustomTkinter
ctk.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
ctk.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

# ============================== [ Funciones principales ] ==============================

# --- Funcion para calcular k ---

def calcular_k(lista):
    N = len(lista)
    k = 1 + 3.3 * math.log10(N) if N > 0 else None
    return k

# --- Funcion para cargar archivo CSV ---

def cargar_archivo():
    global datos
    ruta_archivo = filedialog.askopenfilename(
        title="Seleccionar archivo de datos",
        filetypes=[
            ("Archivos CSV", "*.csv"),
            ("Archivos Excel", "*.xlsx"),
            ("Archivos JSON", "*.json"),
            ("Todos los archivos", "*.*")
        ]
    )
    
    if ruta_archivo:
        try:
            # Determinar el tipo de archivo por su extensión
            extension = os.path.splitext(ruta_archivo)[1].lower()
            
            if extension == '.csv':
                df = pd.read_csv(ruta_archivo)
            elif extension == '.xlsx':
                df = pd.read_excel(ruta_archivo)
            elif extension == '.json':
                df = pd.read_json(ruta_archivo)
            else:
                # Intentar con CSV por defecto
                df = pd.read_csv(ruta_archivo)
                
            mostrar_selector_columna(df)
            mensaje_var.set(f"Archivo cargado: {os.path.basename(ruta_archivo)}")
        except Exception as e:
            mensaje_var.set(f"Error al cargar el archivo: {str(e)}")

# --- Funcion para mostrar selector de columna ---

def mostrar_selector_columna(df):
    selector = ctk.CTkToplevel(ventana)
    selector.title("Seleccionar columna de datos")
    selector.geometry("300x200")
    
    # Asegurar que la ventana aparezca por encima
    selector.attributes('-topmost', True)
    selector.focus_force()
    selector.grab_set()
    
    # Centrar la ventana emergente respecto a la ventana principal
    selector.withdraw()  # Ocultar temporalmente
    selector.update_idletasks()  # Actualizar para obtener dimensiones correctas
    
    # Calcular posición para centrar
    ancho_selector = selector.winfo_width()
    alto_selector = selector.winfo_height()
    ancho_ventana = ventana.winfo_width()
    alto_ventana = ventana.winfo_height()
    x_ventana = ventana.winfo_x()
    y_ventana = ventana.winfo_y()
    
    # Calcular coordenadas para centrar
    x = x_ventana + (ancho_ventana - ancho_selector) // 2
    y = y_ventana + (alto_ventana - alto_selector) // 2
    
    # Posicionar y mostrar
    selector.geometry(f"+{x}+{y}")
    selector.deiconify()
    
    ctk.CTkLabel(selector, text="Seleccione la columna con los datos numericos:").pack(pady=10)
    
    columna_var = ctk.StringVar()
    combo = ctk.CTkComboBox(selector, variable=columna_var, values=list(df.columns))
    combo.pack(pady=10)
    
    ctk.CTkButton(selector, text="Seleccionar", 
              command=lambda: procesar_columna(df, columna_var.get(), selector)).pack(pady=10)

# --- Funcion para procesar columna seleccionada ---

def procesar_columna(df, columna, ventana_selector):
    global datos
    if columna:
        try:
            datos = df[columna].dropna().astype(float).tolist()
            mostrar_datos()
            mensaje_var.set(f"Datos cargados de la columna: {columna}")
            ventana_selector.destroy()
            
            frame_inicio.pack_forget()
            frame_calculos.pack(fill="both", expand=True)
            
            # Procesar datos automaticamente despues de cargar
            procesar_datos()

        except Exception as e:
            mensaje_var.set(f"Cabezon, no funciona, paso esto: {str(e)}")
    else:
        mensaje_var.set("Selecciona una columna")

# --- Funcion para mostrar los datos cargados ---

def mostrar_datos():
    if datos is not None:
        # Limpiar el widget Text
        text_datos.delete("1.0", "end")
        # Insertar datos en el widget Text
        for i, valor in enumerate(datos):
            text_datos.insert("end", f"Dato {i+1}: {valor}\n")
    else:
        text_datos.insert("end", "No hay datos cargados.\n")

# --- Funcion principal para procesar los datos ---

def procesar_datos():
    global df_resultados, figuras
    if not datos:
        mensaje_var.set("No hay datos para procesar, proporcione un archivo CSV primero.")
        return
    
    try:
        # Calculos basicos
        n = datos
        promedio = np.mean(n)
        ct = len(n)
        rango = max(n) - min(n)
        k = round(calcular_k(n))
        c = rango / k
        
        # Crear clases
        min_valor = min(n)
        clases = []
        for i in range(k):
            limite_inf = min_valor + i * c
            limite_sup = min_valor + (i + 1) * c
            clases.append((limite_inf, limite_sup))
        
        # Crear tabla de frecuencias
        tabla = {
            "Numero de clase": list(range(1, k+1)),
            "Clase": [f"{round(clase[0], 2)} - {round(clase[1], 2)}" for clase in clases],
        }
        
        # Calcular marca de clase
        tabla["Marca de clase"] = [(clase[0] + clase[1]) / 2 for clase in clases]
        
        # Calcular frecuencias
        frecuencia_absoluta = [sum(1 for dato in n if clase[0] <= dato < clase[1]) for clase in clases]
        tabla["Frecuencia absoluta"] = frecuencia_absoluta
        tabla["Frecuencia acumulada"] = [sum(tabla["Frecuencia absoluta"][:i+1]) for i in range(len(tabla["Frecuencia absoluta"]))]
        
        total_frecuencia = sum(tabla["Frecuencia absoluta"])
        tabla["Frecuencia relativa"] = [f / total_frecuencia for f in tabla["Frecuencia absoluta"]]
        tabla["Frecuencia porcentual"] = [fr * 100 for fr in tabla["Frecuencia relativa"]]
        
        tabla["(x-X)"] = [x - promedio for x in tabla["Marca de clase"]]
        tabla["(x-X)^2"] = [(x - promedio) ** 2 for x in tabla["Marca de clase"]]
        
        # Crear dataframe de resultados
        df_resultados = pd.DataFrame(tabla)
        
        # Calcular estadisticas adicionales
        sumatoria = df_resultados["(x-X)^2"].sum()
        varianza = (sumatoria / (ct - 1))
        desviacion_estandar = varianza ** 0.5
        
        # Mostrar resultados
        mostrar_resultados(promedio, varianza, desviacion_estandar, rango, k, c)
        
        # Generar graficos
        generar_graficos()
        
        # Calcular regresión lineal
        calcular_regresion_lineal()
        
        mensaje_var.set("Datos procesados correctamente")

    except Exception as e:
        mensaje_var.set(f"No se procesaron los datos, paso esto: {str(e)}")

# --- Funcion para mostrar resultados ---

def mostrar_resultados(promedio, varianza, desviacion, rango, k, c):
    # Mostrar resultados en el text widget
    text_resultados.delete("1.0", "end")
    
    resultados = [
        f"Promedio: {promedio:.4f}",
        f"Varianza: {varianza:.4f}",
        f"Desviacion Estandar: {desviacion:.4f}",
        f"Rango: {rango:.4f}",
        f"Numero de clases (k): {k}",
        f"Amplitud de clase (c): {c:.4f}",
        f"Cantidad de datos: {len(datos)}"
    ]
    
    for resultado in resultados:
        text_resultados.insert("end", resultado + "\n\n")
    
    # Mostrar tabla de frecuencias con mejor formato
    text_tabla.delete("1.0", "end")
    text_tabla.configure(font=("Courier New", 12))  # Usar fuente monoespaciada
    
    # Formatear la tabla con pandas para mejor alineación
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.colheader_justify', 'center')
    pd.set_option('display.precision', 4)
    
    # Insertar la tabla formateada
    text_tabla.insert("end", df_resultados.to_string(index=False) + "\n")

# --- Funcion para calcular regresión lineal ---

def calcular_regresion_lineal():
    if not datos:
        text_regresion.insert("end", "No hay datos para calcular la regresión lineal.\n")
        return
    
    try:
        # Crear datos X e Y para la regresión
        # Usaremos el índice como X y los datos como Y para un ejemplo simple
        x = np.array(range(len(datos)))
        y = np.array(datos)
        
        # Calcular la regresión lineal
        pendiente, intercepto = np.polyfit(x, y, 1)
        r_cuadrado = np.corrcoef(x, y)[0, 1]**2
        
        # Crear la línea de regresión
        linea_regresion = pendiente * x + intercepto
        
        # Mostrar resultados en el widget de texto
        text_regresion.delete("1.0", "end")
        text_regresion.insert("end", "Resultados de la Regresión Lineal:\n\n")
        text_regresion.insert("end", f"Ecuación de la recta: Y = {pendiente:.4f}X + {intercepto:.4f}\n\n")
        text_regresion.insert("end", f"Coeficiente de determinación (R²): {r_cuadrado:.4f}\n\n")
        text_regresion.insert("end", f"Interpretación:\n")
        text_regresion.insert("end", f"- La pendiente ({pendiente:.4f}) indica el cambio promedio en Y por cada unidad de cambio en X.\n")
        text_regresion.insert("end", f"- El intercepto ({intercepto:.4f}) es el valor de Y cuando X es 0.\n")
        text_regresion.insert("end", f"- R² ({r_cuadrado:.4f}) indica qué proporción de la variabilidad en Y es explicada por X.\n")
        
        # Crear gráfico de regresión
        fig_regresion = plt.Figure(figsize=(8, 5))
        ax = fig_regresion.add_subplot(111)
        
        # Graficar puntos de datos
        ax.scatter(x, y, color='blue', alpha=0.5, label='Datos')
        
        # Graficar línea de regresión
        ax.plot(x, linea_regresion, color='red', label=f'Y = {pendiente:.4f}X + {intercepto:.4f}')
        
        # Añadir etiquetas y leyenda
        ax.set_title("Regresión Lineal")
        ax.set_xlabel("Índice")
        ax.set_ylabel("Valor")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Añadir texto con R²
        ax.text(0.05, 0.95, f'R² = {r_cuadrado:.4f}', transform=ax.transAxes, 
                fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        fig_regresion.tight_layout()
        
        # Limpiar frame de gráfico de regresión
        for widget in frame_regresion.winfo_children():
            widget.destroy()
        
        # Mostrar gráfico en el frame
        canvas_regresion = FigureCanvasTkAgg(fig_regresion, frame_regresion)
        canvas_regresion.draw()
        canvas_regresion.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)
        
    except Exception as e:
        text_regresion.delete("1.0", "end")
        text_regresion.insert("end", f"Error al calcular la regresión lineal: {str(e)}\n")

# --- Funcion para generar las graficas ---

def generar_graficos():
    global figuras
    figuras = []
    
    # Limpiar frame de graficos
    for widget in frame_graficos.winfo_children():
        widget.destroy()
    
    # Crear un frame para cada fila de graficos
    frame_fila1 = ctk.CTkFrame(frame_graficos)
    frame_fila1.pack(fill="both", expand=True)
    
    frame_fila2 = ctk.CTkFrame(frame_graficos)
    frame_fila2.pack(fill="both", expand=True)
    
    # Grafico de barras
    fig1 = plt.Figure(figsize=(5, 3))
    ax1 = fig1.add_subplot(111)
    ax1.bar(df_resultados["Clase"], df_resultados["Frecuencia absoluta"], color='skyblue')
    ax1.set_title("Frecuencia Absoluta por Clase")
    ax1.set_xticklabels(df_resultados["Clase"], rotation=45)
    fig1.tight_layout()
    
    canvas1 = FigureCanvasTkAgg(fig1, frame_fila1)
    canvas1.draw()
    canvas1.get_tk_widget().pack(side="left", fill="both", expand=True, padx=5, pady=5)
    
    figuras.append(fig1)
    
    # Grafico de pastel
    fig2 = plt.Figure(figsize=(5, 3))
    ax2 = fig2.add_subplot(111)
    ax2.pie(df_resultados["Frecuencia absoluta"], labels=df_resultados["Clase"], autopct='%1.1f%%', startangle=90)
    ax2.set_title("Distribucion de Frecuencias")
    fig2.tight_layout()
    
    canvas2 = FigureCanvasTkAgg(fig2, frame_fila1)
    canvas2.draw()
    canvas2.get_tk_widget().pack(side="right", fill="both", expand=True, padx=5, pady=5)
    
    figuras.append(fig2)
    
    # Comparacion de series por posicion
    fig3 = plt.Figure(figsize=(5, 3))
    ax3 = fig3.add_subplot(111)
    
    # Crear barras para comparar multiples series
    x = np.arange(len(df_resultados["Clase"]))
    width = 0.15  # ancho de las barras
    
    # Generar 5 series de datos para comparar
    series_base = df_resultados["Marca de clase"]
    series = []
    for i in range(5):
        # Crear series con valores incrementales para la visualización
        series.append([val + (i*5) for val in series_base])
    
    # Colores para las barras
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # Dibujar las barras para cada serie
    for i in range(5):
        ax3.bar(x + (i * width), series[i], width, color=colors[i], 
                label=f'Serie {i+1}')
    
    ax3.set_title("Comparación de series por posición")
    ax3.set_xlabel("Posición X")
    ax3.set_ylabel("Valor")
    ax3.set_xticks(x + width * 2)  # Centrar las etiquetas
    ax3.set_xticklabels(range(len(df_resultados["Clase"])))
    ax3.legend()
    fig3.tight_layout()
    
    canvas3 = FigureCanvasTkAgg(fig3, frame_fila2)
    canvas3.draw()
    canvas3.get_tk_widget().pack(side="left", fill="both", expand=True, padx=5, pady=5)
    
    figuras.append(fig3)
    
    # Grafico pastel para la primera serie
    fig4 = plt.Figure(figsize=(5, 3))
    ax4 = fig4.add_subplot(111)
    
    # Usar la marca de clase como serie 1
    marcas = df_resultados["Marca de clase"]
    ax4.pie(marcas, labels=df_resultados["Clase"], autopct='%1.1f%%', startangle=90)
    ax4.set_title("Distribución de Marcas de Clase")
    fig4.tight_layout()
    
    canvas4 = FigureCanvasTkAgg(fig4, frame_fila2)
    canvas4.draw()
    canvas4.get_tk_widget().pack(side="right", fill="both", expand=True, padx=5, pady=5)
    
    figuras.append(fig4)

# --- Funcion para exportar resultados a pdf ---

def exportar_pdf():
    # Verificar si hay datos cargados
    if 'df_resultados' not in globals():
        mensaje_var.set("No hay resultados para exportar. Procese los datos primero.")
        return
    
    try:
        archivo = filedialog.asksaveasfilename(
            title="Guardar PDF",
            defaultextension=".pdf",
            filetypes=[("Archivos PDF", "*.pdf")]
        )
        
        if not archivo:
            return
        
        # Crear pdf
        pdf = FPDF()
        pdf.add_page()
        
        # Titulo del pdf
        pdf.set_font("Arial", "B", 16) #fuente y tamaño
        pdf.cell(0, 10, "Informe de Analisis Estadistico", 0, 1, "C")
        pdf.ln(10)
        
        # Datos estadisticos
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "Resultados Estadisticos:", 0, 1)
        pdf.set_font("Arial", "", 10)
        
        # Usar los valores que ya calculamos y que estan en el widget de resultados
        text_content = text_resultados.get("1.0", "end")
        resultados = [line.strip() for line in text_content.split('\n') if line.strip()]
        
        for resultado in resultados:
            if resultado:  # Evitar lineas vacias
                pdf.cell(0, 8, resultado, 0, 1)
        
        pdf.ln(10)
        
        # Tabla de frecuencias
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "Tabla de Frecuencias:", 0, 1)
        pdf.set_font("Arial", "", 8)
        
        # Guardar figuras en la memoria usando bytesio
        
        img_buffers = []
        for i, fig in enumerate(figuras):
            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)
            img_buffers.append(buf)
        
        # Agregar tabla simplificada
        columnas = ["Clase", "Marca de clase", "Frec. Abs.", "Frec. Rel.", "Frec. %"]
        anchos = [40, 30, 25, 25, 25]
        
        # Encabezados
        for i, col in enumerate(columnas):
            pdf.cell(anchos[i], 10, col, 1, 0, "C")
        pdf.ln()
        
        # Datos
        for i, row in df_resultados.iterrows():
            pdf.cell(anchos[0], 8, str(row["Clase"]), 1, 0, "C")
            pdf.cell(anchos[1], 8, f"{row['Marca de clase']:.2f}", 1, 0, "C")
            pdf.cell(anchos[2], 8, str(row["Frecuencia absoluta"]), 1, 0, "C")
            pdf.cell(anchos[3], 8, f"{row['Frecuencia relativa']:.4f}", 1, 0, "C")
            pdf.cell(anchos[4], 8, f"{row['Frecuencia porcentual']:.2f}%", 1, 0, "C")
            pdf.ln()
        
        # Agregar graficos desde los buffers de memoria
        for i, buf in enumerate(img_buffers):
            pdf.add_page()
            pdf.set_font("Arial", "B", 12)
            
            if i == 0:
                pdf.cell(0, 10, "Grafico de Barras - Frecuencia Absoluta", 0, 1, "C")
            elif i == 1:
                pdf.cell(0, 10, "Grafico de Pastel - Distribucion de Frecuencias", 0, 1, "C")
            elif i == 2:
                pdf.cell(0, 10, "Comparación de series por posición", 0, 1, "C")
            elif i == 3:
                pdf.cell(0, 10, "Distribución de Marcas de Clase", 0, 1, "C")
            
            # Guardar buffer a un archivo temporal en memoria y añadirlo al PDF
            img_temp = f"temp_img_{i}.png"
            with open(img_temp, 'wb') as img_file:
                img_file.write(buf.getbuffer())
            
            pdf.image(img_temp, x=10, y=40, w=190)
            
            # Eliminar el archivo temporal despues de usarlo
            if os.path.exists(img_temp):
                os.remove(img_temp)
        
        # Guardar pdf
        pdf.output(archivo)
        
        mensaje_var.set(f"Awebo si jalo, se guardo en: {archivo}")

    except Exception as e:
        mensaje_var.set(f"No funciono we, el error es: {str(e)}")

# --- El boton Atras ---

def atras():
    frame_calculos.pack_forget()
    frame_inicio.pack(fill="both", expand=True)

# ============================== [ Configuracion de la ventana ] ==============================

ventana = ctk.CTk()
ventana.title("Actividad 3 - Ciencia de datos")
ventana.geometry("800x600")
ventana.resizable(True, True)

# ============================== [ Variables globales ] ==============================

datos = None
df_resultados = None
figuras = []
mensaje_var = ctk.StringVar()

# ============================== [ Frame para inicio ] ==============================

frame_inicio = ctk.CTkFrame(ventana)
frame_inicio.pack(fill="both", expand=True)

titulo_inicio = ctk.CTkLabel(frame_inicio, text="Analisis Estadistico", font=("Arial", 16, "bold"))
titulo_inicio.pack(pady=20)

boton_cargar = ctk.CTkButton(frame_inicio, text="Cargar archivo CSV", command=cargar_archivo, width=200, height=40)
boton_cargar.pack(pady=20)

# ============================== [ Frame para calculos ] ==============================

frame_calculos = ctk.CTkFrame(ventana)

# Frame de los botones de la segunda pantalla superiores
frame_botones = ctk.CTkFrame(frame_calculos)
frame_botones.pack(fill="x", pady=10)

# Boton para exportar pdf 
boton_exportar = ctk.CTkButton(frame_botones, text="Exportar a PDF", command=exportar_pdf)
boton_exportar.pack(side="left", padx=5)

# Boton atras
boton_regresar = ctk.CTkButton(frame_botones, text="Atras", command=atras)
boton_regresar.pack(side="right", padx=5)

# Notebook para resultados (CustomTkinter no tiene un widget Notebook nativo, usamos un TabView)
notebook = ctk.CTkTabview(frame_calculos)
notebook.pack(fill="both", expand=True, padx=10, pady=10)

# Crear pestañas
tab_datos = notebook.add("Datos")
tab_resultados = notebook.add("Resultados")
tab_tabla = notebook.add("Tabla de Frecuencias")
tab_graficos = notebook.add("Graficos")
tab_regresion = notebook.add("Regresion Lineal")

# Contenido de la pestaña Datos
text_datos = ctk.CTkTextbox(tab_datos, wrap="word")
text_datos.pack(fill="both", expand=True)

# Contenido de la pestaña Resultados
text_resultados = ctk.CTkTextbox(tab_resultados, wrap="word")
text_resultados.pack(fill="both", expand=True)

# Contenido de la pestaña Tabla de Frecuencias
text_tabla = ctk.CTkTextbox(tab_tabla, wrap="none")
text_tabla.pack(fill="both", expand=True)

# Contenido de la pestaña Graficos
frame_graficos = ctk.CTkFrame(tab_graficos)
frame_graficos.pack(fill="both", expand=True)

# Contenido de la pestaña Regresión Lineal
frame_regresion_main = ctk.CTkFrame(tab_regresion)
frame_regresion_main.pack(fill="both", expand=True)

# Dividir en dos secciones: texto y gráfico
text_regresion = ctk.CTkTextbox(frame_regresion_main, wrap="word")
text_regresion.pack(fill="both", expand=True, padx=10, pady=10)

frame_regresion = ctk.CTkFrame(frame_regresion_main)
frame_regresion.pack(fill="both", expand=True, padx=10, pady=10)

# Etiqueta para mensajes
mensaje_label = ctk.CTkLabel(frame_calculos, textvariable=mensaje_var)
mensaje_label.pack(pady=5)

# ============================== [ Loop principal ] ==============================
ventana.mainloop()