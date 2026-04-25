# CP Analizer

## Descripción general

CP Analizer es una herramienta de análisis de capacidad de proceso creada para ayudar a evaluar la calidad y la estabilidad de los datos de producción principalmente cuando tus datos no siguen una distribución normal.

Su objetivo es ofrecer una alternativa sencilla a funciones de software estadístico como Minitab la cual si simplifica el proceso pero aún requiere un trabajo manual y se complica cuando en la industria muchas veces se tienen producciones masivas de distintos lotes y se deben realizar estos analisis de capacidad de proceso de manera mas rápida.

### Ejemplo de resultados visuales

| Antes (Manual) | Después (Análisis Automático) |
| :--- | :--- |
| ![Antes](images/Antes.png) | ![Grafica1](images/despues_1.png) <br> ![Grafica2](images/despues_2.png) |

La aplicacion personal que le dí fue crear un backend para automatizar el proceso completo de Analis de capacidad, donde use cp_analizer para procesar los datos y posteriormente generar los PDFs automaticamente.

## Qué hace

- Analiza datos de subgrupos para calcular indicadores de capacidad como Cp, Cpk, Pp y Ppk.
- Además de la distribución normal, soporta análisis para datos que no siguen una distribución normal.
- Devuelve estadisticos (p-value y Anderson-Darling) que te permiten directamente tomar decisiones.
- Genera un histograma con los límites de especificación (USL y LSL).
- Genera un gráfico de promedio por subgrupo (gráfico X-bar).
- Entrega resultados estructurados que te permiten crear cualquier tipo de informe.

## Distribuciones incluidas

La herramienta realiza análisis con varias distribuciones para encontrar el mejor ajuste a los datos del proceso. Estas incluyen:

- Normal
- Weibull
- Lognormal
- Valor extremo menor (Smallest Extreme Value)
- Valor extremo mayor (Largest Extreme Value)
- Gamma
- Logística
- Log-logística
- Exponencial
- Weibull de 3 parámetros
- Lognormal de 3 parámetros
- Gamma de 3 parámetros
- Log-logística de 3 parámetros
- Exponencial de 2 parámetros
- Transformación Box-Cox
- Transformación Johnson
- Enfoque no paramétrico

## Formato de entrada

Los datos deben estar organizados en una matriz 2D donde cada fila sea un subgrupo y cada columna sea una medición dentro de ese subgrupo.

### Ejemplo de formato válido

```python
import numpy as np

data_subgroups = np.array([
    [100.5, 101.0, 99.8, 100.2, 100.9],
    [99.7, 100.1, 100.3, 100.0, 100.6],
    [100.2, 100.8, 100.0, 99.9, 100.4],
])
```

En otras palabras:

- Cada fila es un lote o subgrupo.
- Cada columna es una medición tomada dentro de ese lote.
- Los límites de especificación (`usl` y `lsl`) son los límites aceptables del proceso.
- `target_mean` es el valor objetivo deseado.

## Uso básico

```python
from data_analizer import ProcessCapabilityAnalizer

analyzer = ProcessCapabilityAnalizer(
    data_subgroups=data_subgroups,
    usl=105.0,
    lsl=95.0,
    target_mean=100.0,
)
```

### Método más importante: `run_full_analysis()`

El método clave de esta herramienta es `run_full_analysis()`. Con él se ejecutan todos los análisis disponibles y se obtiene un conjunto completo de resultados en una estructura de datos fácil de usar.

A partir de esa salida puedes hacer muchas cosas:

- Obtener los valores de Cp, Cpk, Pp y Ppk para cada distribución.
- Elegir la distribución que mejor ajuste tus datos.
- Generar reportes personalizados.
- Extraer métricas específicas de un análisis particular.

El método `report()` es solo un ejemplo de uso. Lo que hace es:

- ejecutar `run_full_analysis()`
- usar los resultados normales para construir gráficos y tablas
- devolver un histograma, un gráfico X-bar y tablas de resumen

Pero si lo deseas, puedes usar directamente `run_full_analysis()` para construir otros reportes o para analizar más a fondo los resultados.

```python
full_results = analyzer.run_full_analysis()
```

### Estructura de resultados

El método `run_full_analysis()` devuelve un diccionario donde cada clave es el nombre de una distribución y cada valor es un objeto `Results` de Pydantic.

| Campo | Descripción |
|---|---|
| `data` | Datos utilizados en el ajuste o análisis |
| `usl` | Límite superior de especificación |
| `lsl` | Límite inferior de especificación |
| `title` | Nombre de la distribución o método analizado |
| `ad` | Estadístico de Anderson-Darling (evaluación de ajuste) |
| `p_value` | Valor p del test de ajuste |
| `params` | Parámetros estimados de la distribución |
| `cp` | Índice Cp |
| `cpk` | Índice Cpk |
| `pp` | Índice Pp |
| `ppk` | Índice Ppk |
| `pdf_values` | Valores de la densidad de probabilidad calculada |

### Ejemplo de acceso a resultados

```python
# Obtener los resultados del análisis normal
normal_results = full_results["Normal"]
print(normal_results.cp)
print(normal_results.cpk)
print(normal_results.params)

# Usar otro análisis de distribución
weibull_results = full_results["Weibull"]
print(weibull_results.pp)
print(weibull_results.ppk)

# Ver la densidad de probabilidad calculada
pdf = normal_results.pdf_values
```

Esta estructura permite:

- Comparar fácilmente los índices entre distribuciones.
- Seleccionar la mejor distribución según el ajuste y los resultados.
- Construir reportes nuevos con las métricas que necesites.

## Ejecución de pruebas

Usa `uv` desde la carpeta del proyecto para ejecutar todas las pruebas:

```powershell
cd <ruta-del-proyecto>
uv run pytest -q
```

Para ejecutar sólo pruebas unitarias o funcionales:

```powershell
uv run pytest -q tests/unit
uv run pytest -q tests/functional
```

## Estructura de pruebas

- `tests/conftest.py` — configuración común de pytest.
- `tests/unit/test_data_analizer_unit.py` — pruebas unitarias de funciones individuales:
  - `plot_histogram()`: comprueba que genera figuras válidas.
  - `plot_xbar_chart()`: valida el gráfico de promedio por subgrupo.
  - `run_normal_analysis()`: verifica que devuelve un objeto `Results` válido.
- `tests/functional/test_data_analizer_functional.py` — pruebas funcionales:
  - Pruebas de `report()`: estructura, contenido y valores de tablas y gráficos.
  - Casos edge: datasets mínimos y datasets grandes.
  - Pruebas de `run_full_analysis()`: valida que:
    - Devuelve un diccionario de resultados.
    - Contiene todas las distribuciones esperadas.
    - Cada distribución tiene un objeto `Results` con campos requeridos.
    - Los valores de Cp, Cpk, Pp, Ppk están dentro de rangos válidos.
    - Las métricas no contienen valores NaN.