# 🌸 Proyecto Matemática 4

Este proyecto utiliza distintas librerías de Python para realizar análisis de regresión sobre el **dataset Iris**.
Utilizamos un **entorno virtual** (`venv`) para mantener las dependencias organizadas.

---

## Configuración del entorno

### 1. Clonar el repositorio

```bash
git clone <URL_DEL_REPO>
cd Mate4
```

### 2. Instalar `venv`

* **Linux:**

```bash
sudo apt update
sudo apt install python3.12-venv -y
```

* **Windows:**
  Descarga Python desde: [python.org](https://www.python.org/downloads/)

### 3. Crear el entorno virtual

```bash
python3 -m venv venv
```

### 4. Activar el entorno

* **Linux:**

```bash
source venv/bin/activate
```

* **Windows (PowerShell):**

```powershell
.\venv\Scripts\Activate.ps1
```

### 5. Instalar dependencias

```bash
pip install -r requirements.txt
```

---

### Para desactivar el entorno

```bash
deactivate
```

## Ejecutar los scripts

| Script                  | Descripción                                    | Comando                               |
| ----------------------- | ---------------------------------------------- | ------------------------------------- |
| `regresion-lineal.py`   | Regresión lineal | `streamlit run regresion-lineal.py`   |
| `minimo-cuadrados.py`   | Regresión múltiple con mínimos cuadrados                   | `streamlit run minimo-cuadrados.py`   |
| `descenso-gradiente.py` | Regresión múltiple con descenso del gradiente  | `streamlit run descenso-gradiente.py` |
| `comparacion.py`        | Comparación entre regresión simple y múltiple  | `streamlit run comparacion.py`        |

---

## Autores

* Lautaro José Gubia
* Facundo Nicolás Mendoza
* Nicolás Agustin Muñoz
