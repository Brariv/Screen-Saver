# Screensaver Paralelo: Sistema Solar de Outer Wilds

Proyecto 1 — Computación Paralela y Distribuida
Universidad del Valle de Guatemala, Semestre 2, 2026

**Integrantes:** Javier Cifuentes (23079) · Brandon Rivera (23088)

---

## Descripción

Screensaver en C++ (OpenGL/GLUT) que simula el sistema solar de *Outer Wilds*: un
Sol central con halo pulsante orbitado por N cuerpos celestes, cada uno con
apariencia (manchas, anillos, estelas) y un **modelo físico distinto**, de modo
que el trabajo de actualizar la escena por cuadro es intencionalmente
**heterogéneo** entre cuerpos. Esto es lo que hace interesante paralelizar el
bucle con OpenMP: no todos los hilos hacen la misma cantidad de trabajo, así que
la elección de `schedule(static)` vs `schedule(dynamic)` sí importa.

El mismo archivo fuente (`src/screensaver.cpp`) compila en dos versiones —
secuencial y paralela (OpenMP) — mediante la macro `USE_OPENMP`, para poder
comparar tiempos de forma directa.

---

## Modelos físicos por cuerpo

Cada cuerpo celeste (`CuerpoCeleste`) lleva un `TipoFisica` que determina qué
rama del `switch` de `actualizarEscena()` ejecuta el hilo que lo atiende:

| Tipo | Cuerpo(s) inspirados en | Cálculo | Costo |
|---|---|---|---|
| `FISICA_CIRCULAR` | Planetas "normales" | Integración de Euler sobre una elipse a velocidad angular constante | Fijo, bajo |
| `FISICA_KEPLERIANA` | The Interloper | Ecuación de Kepler `M = E - e·sin(E)` resuelta por Newton-Raphson cada cuadro (velocidad variable: rápido en perihelio, lento en afelio) | Variable, según iteraciones para converger |
| `FISICA_BINARIA` | Ash Twin / Ember Twin (Hourglass Twins) | Par de cuerpos orbitando un baricentro común que a su vez orbita el Sol, más transferencia de "arena" sobre un contador compartido (`#pragma omp atomic`) | Medio |
| `FISICA_SATELITE` | The Attlerock | Posición dependiente de la de su cuerpo padre; se resuelve en una segunda fase, tras una barrera | Bajo |
| `FISICA_CUANTICA` | The Quantum Moon | Reposicionamiento discreto mediante una función de hash determinista (sin estado compartido, thread-safe) | Bajo, a saltos |
| `FISICA_OSCILANTE` | Dark Bramble | Órbita perturbada por la suma de K armónicos senoidales; K varía por cuerpo | Variable, crece con K — principal fuente de desbalance |

Esta heterogeneidad es deliberada: sirve para estudiar el balanceo de carga de
OpenMP frente a una carga de trabajo desigual entre hilos.

---

## Estructura del bucle de animación

Siguiendo el patrón *calcular en paralelo → sincronizar → renderizar en serie*:

- **`actualizarEscena(dt)` — fase de cálculo.** Recorre el arreglo plano de
  `CuerpoCeleste` e integra la física de cada uno según su `TipoFisica`. Cada
  iteración escribe únicamente en `cuerpos[i]` (salvo el contador compartido de
  arena de la física binaria, protegido con `atomic`), por lo que no hay
  condiciones de carrera. Es el bucle marcado con `#pragma omp parallel for`.
  Los satélites se resuelven en una fase posterior a la barrera implícita,
  porque dependen de la posición ya actualizada de su cuerpo padre.
- **`alDibujar()` — fase de renderizado.** Siempre en el hilo principal, ya que
  el contexto de OpenGL/GLUT no es *thread-safe*.

Otros detalles de implementación:

- **Independencia de los FPS.** La simulación avanza con el `dt` real medido con
  `std::chrono::steady_clock`, acotado a 50 ms para evitar saltos si la ventana
  se minimiza.
- **Nivel de detalle adaptativo.** Por encima de `LIMITE_DETALLE_FINO` (200
  cuerpos) el dibujo se simplifica a disco + halo, sin manchas ni anillos, para
  que el renderizado secuencial no se convierta en el cuello de botella al medir
  *speedup* con N grande.
- **Programación defensiva.** Todos los argumentos se validan con `strtol`
  (formato, desbordamiento y rango); ante cualquier error se imprime el uso y se
  termina con código de salida distinto de cero. No hay memoria cruda: los
  cuerpos viven en un `std::vector`.
- **Sin variables *hard-coded*:** N, ancho, alto y semilla se leen de los
  argumentos de la línea de comandos.

---

## Requisitos

| Dependencia | macOS |
|---|---|
| Compilador | `clang++` (Command Line Tools de Xcode: `xcode-select --install`) |
| OpenGL / GLUT | Frameworks del sistema, ya incluidos en macOS |
| OpenMP | `brew install libomp` (solo para la versión paralela) |

En macOS **no** se usa `<GL/freeglut.h>`. El código incluye las cabeceras
correctas de Apple:

```cpp
#include <GLUT/glut.h>
#include <OpenGL/gl.h>
#include <OpenGL/glu.h>
```

y selecciona automáticamente las de Linux mediante `#ifdef __APPLE__`, para que
el mismo archivo compile en ambos sistemas.

---

## Compilación

```bash
make            # versión secuencial  -> bin/screensaver_seq
make paralelo   # versión con OpenMP  -> bin/screensaver_omp
make todo       # ambas
make run        # compila y ejecuta la version secuencial
make run-omp    # compila y ejecuta la version paralela
make clean      # limpia bin/
make help       # lista de targets
```

La versión secuencial y la paralela son **el mismo archivo fuente**. Los
`#pragma omp` están protegidos con `#ifdef USE_OPENMP`, macro que solo define el
target `paralelo`.

---

## Ejecución

```bash
./bin/screensaver_seq [N] [ancho] [alto] [semilla]
./bin/screensaver_omp [N] [ancho] [alto] [semilla]
```

| Parámetro | Descripción | Rango | Por defecto |
|---|---|---|---|
| `N` | Cantidad de cuerpos celestes que orbitan al Sol | 1 – 2 000 000 | 8 |
| `ancho` | Ancho de la ventana en píxeles | 640 – 7680 | 1024 |
| `alto` | Alto de la ventana en píxeles | 480 – 4320 | 768 |
| `semilla` | Semilla del generador pseudoaleatorio | 0 – 2147483647 | reloj del sistema |

Ejemplos:

```bash
./bin/screensaver_seq                    # Sol + 8 cuerpos, mezcla de fisicas
./bin/screensaver_seq 500 1280 720       # 500 cuerpos
./bin/screensaver_omp 20000 1280 720 42  # escena reproducible con 20 000 cuerpos
OMP_NUM_THREADS=4 ./bin/screensaver_omp 20000 1280 720 42
./bin/screensaver_seq --help             # ayuda
```

Fijar la **semilla** genera exactamente la misma escena en cada corrida, lo cual
es necesario para comparar de forma justa los tiempos secuencial vs. paralelo.

### HUD en pantalla

La esquina superior izquierda muestra, en tiempo real:

- **FPS** (promediados cada 0.5 s) y modo activo (`secuencial` / `OpenMP`).
- **Composición de la escena** por modelo físico (`circular=x kepler=x ...`).
- Con `N ≤ 12`, la **asignación cuerpo → hilo** (`0:h2  1:h0  ...`), útil para
  observar el reparto de trabajo de OpenMP en vivo.

Los FPS también se imprimen en la terminal cada 0.5 s (`FPS= 59.87`).

---

## Controles

| Tecla | Acción |
|---|---|
| `ESC` / `q` / `Q` | Cerrar el screensaver |

---

## Estructura del proyecto

```
.
├── Makefile              # targets secuencial / paralelo
├── README.md
└── src/
    └── screensaver.cpp   # todo el programa (~1700 lineas, un solo archivo)
```
