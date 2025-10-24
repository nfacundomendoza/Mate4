# Proyecto Mate4

Este proyecto utiliza **Python 3.12**, **pandas** y **scikit-learn**.  
Para mantener las dependencias organizadas se utiliza un **entorno virtual** con `venv`.

---

## Pasos para levantar el entorno

1. Clonar el repositorio:
   ```bash
   git clone <URL_DEL_REPO>
   cd Mate4
   ```
2. Instalar el paquete venv:
    ```bash
    sudo apt update
    sudo apt install python3.12-venv -y
    ```
3. Crear entorno:
    ```bash
    python3 -m venv venv
    ```
4. Activar entorno: 
    - Linux:
        ```bash
        source venv/bin/activate
        ```
    - Windows:
        ```cmd
        .\venv\Scripts\Activate.ps1
        ```
5. Instalar dependencias:
    ```bash
    pip install -r requirements.txt
    ```
## Para desacativar el entorno:
    deactivate

## Ejecutar archivo python "regresion lineal" y guardar los resultados en un .txt: 
`streamlit run regresion-lineal.py`

## Ejecutar archivo python "grafico-dispersion": 
`streamlit run grafico-dispersion.py`

