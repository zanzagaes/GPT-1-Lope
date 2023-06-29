# Instalación
Clonar el repositorio. Para ello, ejecutar en una terminal

    git clone https://github.com/zanzagaes/GPT-1-Lope

en el directorio donde se quiera copiar el código. Trasladarse a la nueva carpeta creada:

    cd GPT-1-Lope

Instalar las dependencias (idealmente en un entorno nuevo para evitar conflictos)

	pip install -r requirements.txt

# Modelo
El modelo usado es una versión sencilla de GPT-1, con los siguientes parámetros:

- Bloques decodificador: 6
- Bloques de atención: 6
- Dimensión de las representaciones: 384
- Tamaño de vocabulario: 81
- Ratio de abandono (*dropout*): 0.2

# Ejecución
El fichero `sample.py` inicia el modelo, carga los pesos obtenidos del entrenamiento y lo utiliza para generar texto a partir de un texto de estímulo:

	python3 sample.py --prompt "[Texto de estímulo]"

Si no se proporciona el argumento `prompt` el texto de estímulo por defecto es `\n` (nueva línea).

Los argumentos para la creación de texto son:

- Temperatura: por defecto, 0.8
- Símbolos generados por muestra: por defecto, 500
- Muestras generadas: por defecto, 10

Todos los parámetros pueden alterarse en el archivo ``sample.py``.

## Observaciones
La semilla del generador de números pseudoaleatorios ha sido inicializada a `1337` para obtener resultados reproducibles.

Si se quiere un comportamiento aleatorio hay que comentar las líneas

	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)

del archivo ``sample.py``
