# ============================================================================
#  Makefile - Outer Wilds Screensaver (Prueba de concepto, Entrega 2)
#  UVG - Computacion Paralela y Distribuida
#
#  Targets:
#    make            -> compila la version secuencial  (bin/screensaver_seq)
#    make paralelo   -> compila la version con OpenMP  (bin/screensaver_omp)
#    make todo       -> compila ambas
#    make run        -> compila y ejecuta la version secuencial
#    make clean      -> borra los binarios y objetos
#
#  Nota macOS: GLUT y OpenGL son frameworks del sistema y estan marcados como
#  deprecados desde 10.14; -Wno-deprecated-declarations silencia ese ruido.
#  OpenMP requiere libomp de Homebrew:  brew install libomp
# ============================================================================

CXX       := clang++
CXXSTD    := -std=c++11
CXXFLAGS  := $(CXXSTD) -O2 -Wall -Wextra
SRC       := src/screensaver.cpp
BIN_DIR   := bin
BIN_SEQ   := $(BIN_DIR)/screensaver_seq
BIN_OMP   := $(BIN_DIR)/screensaver_omp

UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S),Darwin)
  # ----------------------------- macOS -------------------------------------
  CXXFLAGS  += -Wno-deprecated-declarations
  GL_LIBS   := -framework GLUT -framework OpenGL -framework Cocoa
  # Prefijo de libomp instalado con Homebrew (Apple Silicon o Intel).
  OMP_PREFIX := $(shell brew --prefix libomp 2>/dev/null)
  OMP_FLAGS  := -Xpreprocessor -fopenmp -I$(OMP_PREFIX)/include
  OMP_LIBS   := -L$(OMP_PREFIX)/lib -lomp
else
  # ------------------------- Linux (respaldo) -------------------------------
  GL_LIBS    := -lglut -lGLU -lGL -lm
  OMP_FLAGS  := -fopenmp
  OMP_LIBS   := -fopenmp
endif

.PHONY: all todo paralelo secuencial run run-omp clean help

all: secuencial

secuencial: $(BIN_SEQ)

paralelo: $(BIN_OMP)

todo: $(BIN_SEQ) $(BIN_OMP)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# Version secuencial: sin OpenMP, los pragmas quedan fuera por el #ifdef.
$(BIN_SEQ): $(SRC) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $(SRC) -o $@ $(GL_LIBS)

# Version paralela: define USE_OPENMP para activar los pragmas del codigo.
$(BIN_OMP): $(SRC) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -DUSE_OPENMP $(OMP_FLAGS) $(SRC) -o $@ $(GL_LIBS) $(OMP_LIBS)

run: $(BIN_SEQ)
	./$(BIN_SEQ) $(N) $(W) $(H)

run-omp: $(BIN_OMP)
	./$(BIN_OMP) $(N) $(W) $(H)

clean:
	rm -rf $(BIN_DIR)

help:
	@echo "make            compila la version secuencial"
	@echo "make paralelo   compila la version con OpenMP"
	@echo "make todo       compila ambas"
	@echo "make clean      borra los binarios"
	@echo ""
	@echo "Ejecucion: ./bin/screensaver_seq [N] [ancho] [alto] [semilla]"
