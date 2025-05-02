# ============================== [ Librerias ] ==============================

import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
plt.ioff()
import math
import pandas as pd
import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog
from tkinter.ttk import Combobox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.metrics import r2_score
import os
import statsmodels.api as sm
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
    global datos  # Declarar 'datos' como global
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
                df = pd.read_csv(ruta_archivo)  # fallback en caso de que no sea ninguno de los anteriores

            datos = df  # Asignar globalmente el DataFrame
            mostrar_selector_columna(df)  # Llamar la función para mostrar el selector de columnas
            mensaje_var.set(f"Archivo cargado: {os.path.basename(ruta_archivo)}")

            # === Poblar comboboxes de regresión lineal ===
            columnas = list(datos.columns)  # Obtener las columnas del dataframe
            combo_x['values'] = columnas  # Asignar las columnas al combobox de X
            combo_y['values'] = columnas  # Asignar las columnas al combobox de Y
            combo_x.set('')  # Limpiar la selección previa en el combobox X
            combo_y.set('')  # Limpiar la selección previa en el combobox Y

            # === Poblar selector de regresión múltiple ===
            actualizar_combo_variables()  # <<--- Esta línea es clave

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
    global datos, datos_procesados
    if columna:
        try:
            datos = df  # Guardamos todo el DataFrame original
            datos_procesados = df[columna].dropna().astype(float)  # Solo los datos de la columna

            mostrar_datos()  # Seguramente debe mostrar `datos_procesados`
            mensaje_var.set(f"Datos cargados de la columna: {columna}")
            ventana_selector.destroy()
            
            frame_inicio.pack_forget()
            frame_calculos.pack(fill="both", expand=True)
            
            # Procesar datos automáticamente después de cargar
            procesar_datos()

        except Exception as e:
            mensaje_var.set(f"Cabezon, no funciona, paso esto: {str(e)}")
    else:
        mensaje_var.set("Selecciona una columna")


# --- Funcion para mostrar los datos cargados ---

def mostrar_datos():
    if datos_procesados is not None:
        text_datos.delete("1.0", "end")
        for i, valor in enumerate(datos_procesados):
            text_datos.insert("end", f"Dato {i+1}: {valor}\n")
    else:
        text_datos.insert("end", "No hay datos cargados.\n")


# --- Funcion principal para procesar los datos ---

def procesar_datos():
    global df_resultados, figuras

    if datos_procesados is None or datos_procesados.empty:
        mensaje_var.set("No hay datos válidos para procesar.")
        return

    try:
        # Usar los datos ya procesados desde procesar_columna
        n = datos_procesados

        # Cálculos básicos
        promedio = np.mean(n)
        ct = len(n)
        rango = max(n) - min(n)
        k = round(calcular_k(n))
        c = rango / k if k != 0 else 1  # evitar división entre cero

        # Crear clases
        min_valor = min(n)
        clases = []
        for i in range(k):
            limite_inf = min_valor + i * c
            limite_sup = min_valor + (i + 1) * c
            clases.append((limite_inf, limite_sup))

        # Crear tabla de frecuencias
        tabla = {
            "Numero de clase": list(range(1, k + 1)),
            "Clase": [f"{round(clase[0], 2)} - {round(clase[1], 2)}" for clase in clases],
        }

        # Calcular marca de clase
        tabla["Marca de clase"] = [(clase[0] + clase[1]) / 2 for clase in clases]

        # Calcular frecuencias
        frecuencia_absoluta = [sum(1 for dato in n if clase[0] <= dato < clase[1]) for clase in clases]
        tabla["Frecuencia absoluta"] = frecuencia_absoluta
        tabla["Frecuencia acumulada"] = [sum(frecuencia_absoluta[:i + 1]) for i in range(len(frecuencia_absoluta))]

        total_frecuencia = sum(frecuencia_absoluta)
        tabla["Frecuencia relativa"] = [f / total_frecuencia for f in frecuencia_absoluta]
        tabla["Frecuencia porcentual"] = [fr * 100 for fr in tabla["Frecuencia relativa"]]

        # Cálculo de (x - X̄) y su cuadrado
        tabla["(x-X)"] = [x - promedio for x in tabla["Marca de clase"]]
        tabla["(x-X)^2"] = [(x - promedio) ** 2 for x in tabla["Marca de clase"]]

        # Crear DataFrame de resultados
        df_resultados = pd.DataFrame(tabla)

        # Estadísticas adicionales
        sumatoria = df_resultados["(x-X)^2"].sum()
        varianza = (sumatoria / (ct - 1)) if ct > 1 else 0
        desviacion_estandar = varianza ** 0.5

        # Mostrar resultados
        mostrar_resultados(promedio, varianza, desviacion_estandar, rango, k, c)

        # Generar gráficos
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

canvas_regresion = None
figura_regresion = None

def calcular_regresion_lineal():
    global datos
    try:
        # Obtener las columnas seleccionadas desde los combobox
        columna_x = combo_x.get()
        columna_y = combo_y.get()

        # Verificar si las columnas seleccionadas existen en el DataFrame
        if columna_x not in datos.columns or columna_y not in datos.columns:
            mensaje_var.set("Por favor, selecciona columnas válidas.")
            return

        # Extraer las columnas como Series, quitando valores nulos y convirtiendo a float
        x = datos[columna_x].dropna().astype(float)
        y = datos[columna_y].dropna().astype(float)

        # Asegurar que ambas series tengan la misma longitud
        min_len = min(len(x), len(y))
        x = x.iloc[:min_len]
        y = y.iloc[:min_len]

        if len(x) == 0 or len(y) == 0:
            mensaje_var.set("Las columnas seleccionadas no tienen suficientes datos.")
            return

        # Agregar una constante (intercepto) a la variable x (requerido por statsmodels)
        x = sm.add_constant(x)

        # Calcular la regresión lineal con statsmodels
        modelo = sm.OLS(y, x)  # OLS = Ordinary Least Squares (mínimos cuadrados ordinarios)
        resultados = modelo.fit()

        # Obtener el resumen completo de los resultados
        resumen = resultados.summary()

        # Mostrar el resumen de la regresión en el TextBox
        text_regresion.delete("1.0", "end")
        text_regresion.insert("end", str(resumen))

        # Limpiar gráfico anterior
        for widget in frame_regresion.winfo_children():
            widget.destroy()

        # Crear y mostrar el gráfico
        fig, ax = plt.subplots()
        ax.scatter(x.iloc[:, 1], y, label="Datos", color="blue")  # x.iloc[:, 1] para obtener la columna sin la constante
        ax.plot(x.iloc[:, 1], resultados.fittedvalues, color='red', label="Regresión lineal")
        ax.set_xlabel(columna_x)
        ax.set_ylabel(columna_y)
        ax.legend()
        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=frame_regresion)
        canvas.draw()
        canvas.get_tk_widget().pack()

    except Exception as e:
        mensaje_var.set(f"Error en la regresión: {str(e)}")

# Función para calcular regresión múltiple
def calcular_regresion_multiple():
    global datos, variables_x, variable_y, text_resultado, frame_grafica, figura_canvas

    if datos is None or datos.empty:
        text_resultado.configure(state="normal")
        text_resultado.delete("1.0", "end")
        text_resultado.insert("1.0", "No se han cargado los datos correctamente.")
        text_resultado.configure(state="disabled")
        return

    if not variables_x or not variable_y:
        text_resultado.configure(state="normal")
        text_resultado.delete("1.0", "end")
        text_resultado.insert("1.0", "Selecciona las variables para la regresión.")
        text_resultado.configure(state="disabled")
        return

    try:
        X = datos[variables_x]
        y = datos[variable_y]

        X_const = sm.add_constant(X)
        modelo_sm = sm.OLS(y, X_const).fit()
        resumen = modelo_sm.summary().as_text()

        regresion = LinearRegression()
        regresion.fit(X, y)
        y_pred = regresion.predict(X)

        # Mostrar resumen
        text_resultado.configure(state="normal")
        text_resultado.delete("1.0", "end")
        text_resultado.insert("1.0", resumen)
        text_resultado.configure(state="disabled")

        # Limpiar gráfica anterior
        for widget in frame_grafica.winfo_children():
            widget.destroy()

        # Crear nueva gráfica
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(y, y_pred, c='blue', label='Valores Predichos')
        ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', label='Línea Ideal')
        ax.set_xlabel("Valores Reales (Y)")
        ax.set_ylabel("Valores Predichos (Ŷ)")
        ax.set_title("Regresión Múltiple: Y vs Ŷ")
        ax.legend()
        ax.grid(True)

        figura_canvas = FigureCanvasTkAgg(fig, master=frame_grafica)
        figura_canvas.draw()
        figura_canvas.get_tk_widget().pack(fill="both", expand=True)

    except Exception as e:
        text_resultado.configure(state="normal")
        text_resultado.delete("1.0", "end")
        text_resultado.insert("1.0", f"Error en la regresión: {str(e)}")
        text_resultado.configure(state="disabled")

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

# Función para cerrar la aplicación limpiamente
def cerrar_aplicacion():
    try:
        if canvas_regresion:
            canvas_regresion.get_tk_widget().pack_forget()
        if figura_regresion:
            plt.close(figura_regresion)
        ventana.destroy()
    except Exception as e:
        print(f"Error al cerrar la aplicación: {e}")




ventana.protocol("WM_DELETE_WINDOW", cerrar_aplicacion)

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
tab_regresion_multiple = notebook.add("Regresion Multiple")

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

#Contenido de la pestaña Regresión Múltiple
frame_regresion_multiple_main = ctk.CTkFrame(tab_regresion_multiple)
frame_regresion_multiple_main.pack(fill="both", expand=True)

# ============================== [ Controles para regresión lineal ] ==============================

# Frame superior para selección de variables y botón
frame_seleccion = ctk.CTkFrame(frame_regresion_main)
frame_seleccion.pack(fill="x", padx=10, pady=(10, 0))

# Etiqueta y combobox para variable X
label_x = ctk.CTkLabel(frame_seleccion, text="Variable X:")
label_x.pack(side="left", padx=(0, 5))
combo_x = ctk.CTkComboBox(frame_seleccion, state="normal", width=200)
combo_x.pack(side="left", padx=(0, 15))

# Etiqueta y combobox para variable Y
label_y = ctk.CTkLabel(frame_seleccion, text="Variable Y:")
label_y.pack(side="left", padx=(0, 5))
combo_y = ctk.CTkComboBox(frame_seleccion, state="normal", width=200)
combo_y.pack(side="left", padx=(0, 15))

# Botón para calcular regresión
boton_regresion = ctk.CTkButton(frame_seleccion, text="Calcular Regresión", command=calcular_regresion_lineal)
boton_regresion.pack(side="left", padx=10)

# ========== NUEVA SECCIÓN COMBINADA: Texto y gráfico uno al lado del otro ==========

# Frame contenedor que se expandirá en vertical
frame_resultado_y_grafico = ctk.CTkFrame(frame_regresion_main)
frame_resultado_y_grafico.pack(fill="both", expand=True, padx=10, pady=10)

# Configuración del grid interno para dividir horizontalmente
frame_resultado_y_grafico.grid_rowconfigure(0, weight=1)
frame_resultado_y_grafico.grid_columnconfigure(0, weight=1)  # Para texto
frame_resultado_y_grafico.grid_columnconfigure(1, weight=1)  # Para gráfico

# Textbox a la izquierda
text_regresion = ctk.CTkTextbox(frame_resultado_y_grafico, wrap="word")
text_regresion.grid(row=0, column=0, sticky="nsew", padx=(0, 5), pady=5)

# Frame para el gráfico a la derecha
frame_regresion = ctk.CTkFrame(frame_resultado_y_grafico)
frame_regresion.grid(row=0, column=1, sticky="nsew", padx=(5, 0), pady=5)


#==============================multiple========================================

# Variables para almacenar las selecciones de las variables
variables_x = []  # Variables para las características independientes
variable_y = ""   # Variable para la característica dependiente
checkboxes_x_multiple = []  # Lista para almacenar los checkboxes

# =============================== [ Controles para regresión múltiple ] ==============================

# Frame superior para selección de variables
frame_seleccion_multiple = ctk.CTkFrame(frame_regresion_multiple_main, height=40)
frame_seleccion_multiple.pack(fill="x", padx=5, pady=(5, 0))
frame_seleccion_multiple.pack_propagate(False)  # Evita que se expanda

# Etiqueta para variables X
label_x_multiple = ctk.CTkLabel(frame_seleccion_multiple, text="Variables X (Independientes):")
label_x_multiple.grid(row=0, column=0, padx=5, pady=5, sticky="w")

# ScrollableFrame en lugar de Listbox
scroll_frame_x = ctk.CTkScrollableFrame(frame_seleccion_multiple, width=150, height=10)
scroll_frame_x.grid(row=1, column=0, pady=5)

# Etiqueta y combobox para variable Y (dependiente)
label_y_multiple = ctk.CTkLabel(frame_seleccion_multiple, text="Variable Y (Dependiente):")
label_y_multiple.grid(row=0, column=1, padx=5, pady=5, sticky="w")

combo_y_multiple = ctk.CTkComboBox(frame_seleccion_multiple, state="normal", width=200)
combo_y_multiple.grid(row=1, column=1, padx=5, pady=5)

# Botón para realizar la regresión múltiple
boton_regresion_multiple = ctk.CTkButton(
    frame_seleccion_multiple,
    text="Calcular Regresión Múltiple",
    command=lambda: seleccionar_variables_multiple()
)
boton_regresion_multiple.grid(row=1, column=2, columnspan=2, pady=10)

# ======================= [ ÁREA DE RESULTADOS EN LA PESTAÑA ] ========================

# Frame inferior para mostrar resultados
frame_resultados_multiple = ctk.CTkFrame(frame_regresion_multiple_main)
frame_resultados_multiple.pack(fill="both", expand=True, padx=10, pady=10)

# Textbox para mostrar el resumen
text_resultado = ctk.CTkTextbox(frame_resultados_multiple, wrap="word", font=("Consolas", 11))
text_resultado.pack(side="left", fill="both", expand=True, padx=(0, 5))

# Frame para la gráfica
frame_grafica = ctk.CTkFrame(frame_resultados_multiple)
frame_grafica.pack(side="right", fill="both", expand=True, padx=(5, 0))

# Variable para evitar superposición de gráficas
figura_canvas = None

# =================================== FUNCIONES =======================================

def crear_selector_multiple(frame, columnas):
    global checkboxes_x_multiple
    checkboxes_x_multiple = []

    for widget in frame.winfo_children():
        widget.destroy()

    for col in columnas:
        var = ctk.StringVar(value="0")
        checkbox = ctk.CTkCheckBox(frame, text=col, variable=var, onvalue="1", offvalue="0")
        checkbox.pack(anchor="w", padx=5, pady=2)
        checkboxes_x_multiple.append((col, var))

def actualizar_combo_variables():
    global datos

    if datos is not None:
        columnas = datos.columns.tolist()

        combo_y.configure(values=columnas)
        combo_x.configure(values=columnas)

        crear_selector_multiple(scroll_frame_x, columnas)
        combo_y_multiple.configure(values=columnas)

def seleccionar_variables_multiple():
    global variables_x, variable_y

    variables_x = [col for col, var in checkboxes_x_multiple if var.get() == "1"]
    variable_y = combo_y_multiple.get()

    calcular_regresion_multiple()


# Área de texto para mostrar los resultados de la regresión
#text_regresion_multiple = ctk.CTkTextbox(frame_regresion_multiple_main, wrap="word")
#text_regresion_multiple.pack(fill="both", expand=True, padx=10, pady=10)


# Etiqueta para mensajes
mensaje_label = ctk.CTkLabel(frame_calculos, textvariable=mensaje_var)
mensaje_label.pack(pady=5)

# ============================== [ Loop principal ] ==============================
ventana.mainloop()