# Screensaver Paralelo: Sistema Solar de Outer Wilds

Proyecto 1 — Computación Paralela y Distribuida
Universidad del Valle de Guatemala, Semestre 2, 2026

**Integrantes:** Javier Cifuentes (23079) · Brandon Rivera (23088)

---

## Estado actual: Entrega 2 — Prueba de concepto

Esta entrega valida que la herramienta gráfica seleccionada (**C++ + OpenGL/GLUT**)
funciona en el equipo de trabajo y deja iniciada la implementación del proyecto.
Cubre los dos puntos pedidos:

1. **Creación de la ventana / área gráfica** donde se mostrará el screensaver
   (ventana GLUT con doble buffer, proyección ortográfica 1 unidad = 1 píxel,
   redimensionable, cierre con `ESC` o `q`).
2. **Visualización de al menos un tipo de elemento de la propuesta:** el **Sol**
   central con su halo pulsante y **N cuerpos celestes** que lo orbitan sobre
   trayectorias elípticas, con color, tamaño y velocidad pseudoaleatorios.

Las animaciones locales de cada cuerpo (arena de los Hourglass Twins, grietas de
Brittle Hollow, zarcillos de Dark Bramble, cola de The Interloper, ciclo de
supernova) y la medición formal de *speedup* quedan para la entrega final.

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
make clean      # limpia bin/
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
| `N` | Cantidad de cuerpos celestes que orbitan al Sol | 1 – 2 000 000 | 1 |
| `ancho` | Ancho de la ventana en píxeles | ≥ 640 | 1024 |
| `alto` | Alto de la ventana en píxeles | ≥ 480 | 768 |
| `semilla` | Semilla del generador pseudoaleatorio | ≥ 0 | reloj del sistema |

Ejemplos:

```bash
./bin/screensaver_seq                  # Sol + 1 cuerpo orbitando
./bin/screensaver_seq 500 1280 720     # 500 cuerpos
./bin/screensaver_omp 20000 1280 720 42  # escena reproducible con 20 000 cuerpos
OMP_NUM_THREADS=4 ./bin/screensaver_omp 20000 1280 720 42
./bin/screensaver_seq --help           # ayuda
```

Fijar la **semilla** genera exactamente la misma escena en cada corrida, lo cual
es necesario para comparar de forma justa los tiempos secuencial vs. paralelo.

Los **FPS** se muestran en la esquina superior izquierda de la ventana y también
se imprimen en la terminal cada 0.5 s (`FPS= 59.87`).

---

## Estructura del proyecto

```
.
├── Makefile              # targets secuencial / paralelo
├── README.md
└── src/
    └── screensaver.cpp   # todo el programa (PoC de un solo archivo)
```

---

## Diseño relevante para la paralelización

El bucle de animación separa explícitamente dos fases, siguiendo el patrón
descrito en la propuesta (*calcular en paralelo → sincronizar → renderizar en
serie*):

- **`actualizarEscena(dt)` — fase de cálculo.** Recorre el arreglo plano de
  `CuerpoCeleste` e integra el ángulo orbital de cada cuerpo. Cada iteración
  escribe únicamente en `cuerpos[i]` y solo lee variables compartidas
  inmutables, por lo que no hay dependencias entre iteraciones ni condiciones de
  carrera. Es el bucle marcado con `#pragma omp parallel for schedule(static)`.
  La barrera implícita al cerrar la región paralela garantiza que todas las
  posiciones estén listas antes de dibujar.
- **`alDibujar()` — fase de renderizado.** Siempre en el hilo principal, ya que
  el contexto de OpenGL/GLUT no es *thread-safe*.

Otros detalles de implementación:

- **Física / trigonometría.** La posición sale de la ecuación paramétrica de la
  elipse (`x = cx + a·cos θ`, `y = cy + b·sin θ`) y la velocidad angular se
  deriva del radio orbital imitando la tercera ley de Kepler (`ω ∝ r^-3/2`), de
  modo que los cuerpos cercanos al Sol giran más rápido.
- **Independencia de los FPS.** La simulación avanza con el `dt` real medido con
  `std::chrono::steady_clock`, acotado a 50 ms para evitar saltos si la ventana
  se minimiza.
- **Programación defensiva.** Todos los argumentos se validan con `strtol`
  (formato, desbordamiento y rango) y ante cualquier error se imprime el uso y
  se termina con código de salida distinto de cero. No se usa memoria cruda:
  los cuerpos viven en un `std::vector`, por lo que no hay fugas.
- **Sin variables *hard-coded*:** N, ancho, alto y semilla se leen de los
  argumentos de la línea de comandos.

---

## Controles

| Tecla | Acción |
|---|---|
| `ESC` / `q` | Cerrar el screensaver |
