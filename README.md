# JVCRSP: MAHH con Thompson sampling

## Integrantes:
- Ignacio Chandia 🍉
- Francisco Correa 🐈
- Florencia Salazar 🌸


## Requisitos

- Python 3 o superior
- numpy 

## Archivos

| Archivo | Contenido |
|---|---|
| `thompson_hyper.py` | Punto de entrada. Método de aprendizaje: Thompson sampling (sección 4.2.2 del paper). |
| `mahh.py` | La hiperheurística (Algoritmo 4), los reinicios, la línea de comandos y los archivos de salida. |
| `heuristicas.py` | Representación de la solución, evaluación (asignación de tripulaciones) y las 34 heurísticas de bajo nivel (LLs). |
| `postproceso.py` | Pulido de la mejor solución de la MAHH. Nunca empeora el objetivo. |
| `solucion_inicial.py` | Lectura de instancias, solución inicial, verificador y salida JSON. |

## Uso

Una instancia:

    python3 thompson_hyper.py instancia.txt --time 60 --pp-frac 0.5

Todas las instancias de una carpeta, en paralelo:

    python3 thompson_hyper.py carpeta_instancias --time 300 --pp-frac 0.5 --workers 3

Solo la solución inicial:

    python3 solucion_inicial.py instancia.txt --out solucion.json

## Opciones

| Opción | Por defecto | Descripción |
|---|---|---|
| `--time` | 60 | Segundos de la MAHH por instancia. |
| `--pp-frac` | 0.10 | Tiempo del post-proceso como fracción de `--time`. Es adicional: con `--time 300 --pp-frac 0.5` cada instancia tarda unos 450 s. `0` = sin post-proceso. |
| `--seed` | 0 | Semilla. |
| `--restarts` | 3 | Corridas de la MAHH en que se reparte el tiempo; cada una parte de la mejor solución encontrada. `1` = una sola corrida. |
| `--no-chain` | no | Cada reinicio parte de la solución inicial en vez de la mejor. |
| `--max-seq` | 25 | Largo máximo de una secuencia de LLs. `0` = sin límite. |
| `--theta` | 0.035 | Tolerancia de aceptación. |
| `--rr` | on | `off` = solo acepta soluciones que no empeoran a la actual (Record to Record). |
| `--rounding` | ceil | Redondeo de distancias (`ceil` reproduce el anexo de Lam et al.). |
| `--workers` | núcleos − 1 | Procesos en paralelo al resolver una carpeta. |
| `--outdir` | `resultados_thompson_s<semilla>` | Carpeta de salida. |
| `--top` | 10 | Cuántas LLs mostrar en la tabla de uso por pantalla. |

## Algoritmo

1. **Solución inicial.** Rutas por inserción más barata: los pares se ordenan por
   la apertura de la ventana de la recogida y se insertan uno a uno en la posición
   factible de menor costo de vehículos.
2. **MAHH.** Thompson sampling elige la siguiente LL según la anterior y decide si
   aplicar la secuencia. La secuencia se aplica a la solución actual, se evalúa y
   se acepta según las ecuaciones (6) y (7), de la referencia. El aprendizaje se actualiza cuando la
   secuencia mejora la solución actual.
3. **Evaluación.** Las tripulaciones no se guardan en la solución: se calculan al
   evaluarla con tres métodos y se usa el de menor costo:
   - asignación greedy, con cambios de vehículo y caminos más cortos como pasajero;
   - asignación exacta, hasta `EXACT_MAX_LEGS` tramos;
   - programación dinámica por vehículo, que puede retener un vehículo para que
     una espera no cuente como conducción.

   En la salida final, con más tramos, la asignación exacta se resuelve como
   cobertura de conjuntos por ramificación y acotamiento.
4. **Post-proceso.** Pasadas de mejora local (esperas, políticas de tripulación,
   eliminación y fusión de vehículos, cambios dentro y entre rutas, horarios) con
   el tiempo adicional de `--pp-frac`.


## Archivos de salida

En la carpeta indicada por `--outdir`:

| Archivo | Contenido |
|---|---|
| `<instancia>_mahh.json` | La solución: rutas, horarios, tramos con su conductor y viajes de cada tripulación. |
| `<instancia>_evolucion.csv` | Mejor objetivo en el tiempo. |
| `<instancia>_heuristicas.csv` | Uso de cada LL. |
| `<instancia>_secuencias.csv` | Secuencias de LLs que encontraron una nueva mejor solución. |
| `resumen.csv` | Una fila por instancia: objetivo inicial, de la MAHH (`obj_mahh`) y final, vehículos, tripulaciones, iteraciones, tiempos y aporte del post-proceso. |
| `heuristicas_total.csv` | Uso de las LLs sumando todas las instancias de la carpeta. |

Si se vuelve a resolver una instancia en la misma carpeta, sus archivos y su fila
de `resumen.csv` se reemplazan; las demás se mantienen.
