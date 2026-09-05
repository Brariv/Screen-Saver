// ============================================================================
//  Screensaver Paralelo: Sistema Solar de Outer Wilds
//  Universidad del Valle de Guatemala - Computacion Paralela y Distribuida
//  Proyecto 1 - Entrega 2: Prueba de concepto
//
//  Integrantes: Javier Cifuentes (23079), Brandon Rivera (23088)
//
//  Alcance de esta entrega (prueba de concepto):
//    1. Creacion de la ventana / area grafica donde correra el screensaver.
//    2. Visualizacion de al menos un tipo de elemento de la propuesta:
//       el Sol central (fijo, con pulso de brillo) y N cuerpos celestes que
//       orbitan a su alrededor siguiendo trayectorias elipticas.
//
//  La estructura de datos (arreglo plano de CuerpoCeleste) y la separacion
//  entre la fase de CALCULO y la fase de RENDERIZADO ya estan planteadas para
//  la paralelizacion con OpenMP descrita en la propuesta:
//      calcular en paralelo  ->  sincronizar  ->  renderizar en serie
//
//  Compilacion: ver Makefile / README.md
// ============================================================================

// ---------------------------------------------------------------------------
// Cabeceras de la libreria grafica.
// En macOS GLUT y OpenGL vienen como frameworks del sistema y sus rutas de
// include son distintas a las de Linux, por eso la seleccion condicional.
// ---------------------------------------------------------------------------
#ifdef __APPLE__
#include <GLUT/glut.h>
#include <OpenGL/gl.h>
#include <OpenGL/glu.h>
#else
#include <GL/freeglut.h>
#include <GL/gl.h>
#include <GL/glu.h>
#endif

// OpenMP solo se incluye cuando se compila el binario paralelo
// (make paralelo -> -DUSE_OPENMP -fopenmp).
#ifdef USE_OPENMP
#include <omp.h>
#endif

#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

// ============================================================================
//  CONSTANTES DE CONFIGURACION
// ============================================================================

static const int   ANCHO_MINIMO      = 640;    // requisito del enunciado (w)
static const int   ALTO_MINIMO       = 480;    // requisito del enunciado (h)
static const int   ANCHO_MAXIMO      = 7680;   // cota defensiva (8K)
static const int   ALTO_MAXIMO       = 4320;
static const long  N_MINIMO          = 1;      // al menos un cuerpo orbitando
static const long  N_MAXIMO          = 2000000;// cota defensiva de memoria
static const long  N_POR_DEFECTO     = 8;      // PoC: Sol + unos cuantos cuerpos
static const int   ANCHO_POR_DEFECTO = 1024;
static const int   ALTO_POR_DEFECTO  = 768;

static const int   SEGMENTOS_CIRCULO = 24;     // resolucion del poligono
static const float PI                = 3.14159265358979323846f;
static const float DT_MAXIMO         = 0.05f;  // clamp anti-salto (50 ms)
static const float INTERVALO_FPS     = 0.5f;   // ventana de promedio de FPS

// ============================================================================
//  ESTRUCTURAS DE DATOS
// ============================================================================

// Un cuerpo celeste que orbita al Sol. Cada instancia es INDEPENDIENTE de las
// demas: su nuevo estado depende unicamente de sus propios campos y del dt.
// Esa independencia es justamente lo que permitira repartir el arreglo entre
// varios hilos con #pragma omp parallel for sin condiciones de carrera.
struct CuerpoCeleste {
    float semiEjeMayor;     // radio horizontal de la orbita (pixeles)
    float semiEjeMenor;     // radio vertical de la orbita (pixeles)
    float anguloOrbital;    // angulo actual sobre la orbita (radianes)
    float velocidadAngular; // rad/s; derivada del radio (3a ley de Kepler)
    float radioCuerpo;      // radio dibujado del planeta (pixeles)
    float posX, posY;       // posicion resultante en coordenadas de ventana
    float colorR, colorG, colorB; // color pseudoaleatorio del cuerpo
};

// Estado global de la escena. GLUT trabaja con callbacks sin parametros de
// usuario, por lo que el estado debe ser accesible globalmente.
struct EstadoEscena {
    std::vector<CuerpoCeleste> cuerpos; // los N cuerpos que orbitan
    int   anchoVentana;
    int   altoVentana;
    float centroX, centroY;   // posicion del Sol (centro de la ventana)
    float radioSolBase;       // radio nominal del Sol
    float radioSolActual;     // radio con el pulso aplicado
    float tiempoAcumulado;    // segundos transcurridos desde el inicio

    // Medicion de FPS
    int   framesEnIntervalo;
    float tiempoEnIntervalo;
    float fpsActual;

    EstadoEscena()
        : anchoVentana(ANCHO_POR_DEFECTO), altoVentana(ALTO_POR_DEFECTO),
          centroX(0.0f), centroY(0.0f), radioSolBase(40.0f),
          radioSolActual(40.0f), tiempoAcumulado(0.0f),
          framesEnIntervalo(0), tiempoEnIntervalo(0.0f), fpsActual(0.0f) {}
};

static EstadoEscena g_escena;
static std::chrono::steady_clock::time_point g_marcaTiempoPrevia;

// ============================================================================
//  ARGUMENTOS DE LINEA DE COMANDOS (programacion defensiva)
// ============================================================================

// Imprime el modo de uso del programa. Se llama ante cualquier argumento
// invalido para que el usuario sepa como corregirlo.
static void imprimirUso(const char* nombrePrograma) {
    std::fprintf(stderr,
        "\nUso: %s [N] [ancho] [alto] [semilla]\n"
        "\n"
        "  N        Cantidad de cuerpos celestes que orbitan al Sol.\n"
        "           Entero en [%ld, %ld]. Por defecto: %ld\n"
        "  ancho    Ancho de la ventana en pixeles. Minimo %d. Por defecto: %d\n"
        "  alto     Alto de la ventana en pixeles.  Minimo %d. Por defecto: %d\n"
        "  semilla  Semilla del generador pseudoaleatorio (entero >= 0).\n"
        "           Por defecto se usa el reloj del sistema. Fijarla permite\n"
        "           repetir exactamente la misma escena entre corridas, algo\n"
        "           necesario para comparar tiempos secuencial vs paralelo.\n"
        "\n"
        "Ejemplos:\n"
        "  %s              -> Sol + 1 cuerpo, ventana %dx%d\n"
        "  %s 500 1280 720 -> Sol + 500 cuerpos en una ventana de 1280x720\n"
        "  %s 500 1280 720 42 -> misma escena reproducible (semilla 42)\n\n",
        nombrePrograma,
        N_MINIMO, N_MAXIMO, N_POR_DEFECTO,
        ANCHO_MINIMO, ANCHO_POR_DEFECTO,
        ALTO_MINIMO, ALTO_POR_DEFECTO,
        nombrePrograma, ANCHO_POR_DEFECTO, ALTO_POR_DEFECTO,
        nombrePrograma, nombrePrograma);
}

// Convierte una cadena a entero largo validando que sea numerica, que no
// desborde y que caiga dentro del rango [minimo, maximo].
// Devuelve true si la conversion fue exitosa; el valor queda en 'salida'.
static bool leerEnteroValidado(const char* texto, const char* nombreParametro,
                               long minimo, long maximo, long* salida) {
    if (texto == NULL || *texto == '\0') {
        std::fprintf(stderr, "Error: el parametro '%s' esta vacio.\n",
                     nombreParametro);
        return false;
    }

    errno = 0;
    char* finLectura = NULL;
    long valor = std::strtol(texto, &finLectura, 10);

    if (finLectura == texto || *finLectura != '\0') {
        std::fprintf(stderr,
                     "Error: '%s' no es un entero valido para el parametro "
                     "'%s'.\n", texto, nombreParametro);
        return false;
    }
    if (errno == ERANGE) {
        std::fprintf(stderr,
                     "Error: el valor '%s' del parametro '%s' esta fuera del "
                     "rango representable.\n", texto, nombreParametro);
        return false;
    }
    if (valor < minimo || valor > maximo) {
        std::fprintf(stderr,
                     "Error: el parametro '%s' debe estar entre %ld y %ld "
                     "(se recibio %ld).\n",
                     nombreParametro, minimo, maximo, valor);
        return false;
    }

    *salida = valor;
    return true;
}

// ============================================================================
//  INICIALIZACION DE LA ESCENA
// ============================================================================

// Genera los N cuerpos con parametros orbitales pseudoaleatorios.
// - Los radios de orbita se reparten entre un anillo interior y el borde de la
//   ventana para que ningun cuerpo quede tapado por el Sol ni salga del canvas.
// - La velocidad angular se deriva del radio imitando la tercera ley de Kepler
//   (w es proporcional a r^-1.5): los cuerpos cercanos giran mas rapido.
static void inicializarCuerpos(long cantidad, unsigned int semilla) {
    std::mt19937 generador(semilla);
    std::uniform_real_distribution<float> distUnitaria(0.0f, 1.0f);

    const float radioMaximoDisponible =
        0.5f * static_cast<float>(g_escena.anchoVentana < g_escena.altoVentana
                                      ? g_escena.anchoVentana
                                      : g_escena.altoVentana);
    const float radioOrbitaMinima = g_escena.radioSolBase * 3.0f;
    const float radioOrbitaMaxima = radioMaximoDisponible * 0.92f;

    g_escena.cuerpos.clear();
    g_escena.cuerpos.reserve(static_cast<size_t>(cantidad));

    for (long i = 0; i < cantidad; ++i) {
        CuerpoCeleste cuerpo;

        // Radio orbital: muestreo estratificado. En lugar de un valor
        // totalmente aleatorio, el rango disponible se divide en N franjas y
        // cada cuerpo cae dentro de la suya con un pequeno desplazamiento
        // aleatorio. Asi las orbitas quedan bien repartidas tanto para N=1
        // como para N grande, sin que todos los cuerpos se amontonen.
        const float centroFranja = (static_cast<float>(i) + 0.5f) /
                                   static_cast<float>(cantidad);
        const float desplazamiento =
            0.5f * (distUnitaria(generador) - 0.5f) /
            static_cast<float>(cantidad);
        const float fraccion = centroFranja + desplazamiento;
        const float radioOrbital = radioOrbitaMinima +
            fraccion * (radioOrbitaMaxima - radioOrbitaMinima);

        // Excentricidad suave: el semieje menor es entre 70% y 100% del mayor,
        // lo que produce orbitas elipticas como la de The Interloper.
        const float achatamiento = 0.70f + 0.30f * distUnitaria(generador);
        cuerpo.semiEjeMayor = radioOrbital;
        cuerpo.semiEjeMenor = radioOrbital * achatamiento;

        // Fase inicial aleatoria para que no arranquen todos alineados.
        cuerpo.anguloOrbital = distUnitaria(generador) * 2.0f * PI;

        // Tercera ley de Kepler (forma simplificada): w = k * r^(-3/2).
        // La constante esta calibrada para que un cuerpo a media distancia
        // complete una vuelta en ~30 s: un movimiento lento y contemplativo,
        // acorde a un screensaver (y al ritmo pausado del juego).
        // El signo alterna para que algunos cuerpos giren en sentido inverso.
        const float constanteGravitacional = 600.0f;
        const float magnitudVelocidad =
            constanteGravitacional / std::pow(radioOrbital, 1.5f);
        const float sentido = (distUnitaria(generador) < 0.15f) ? -1.0f : 1.0f;
        cuerpo.velocidadAngular = sentido * magnitudVelocidad;

        // Tamano del cuerpo: variacion pseudoaleatoria acotada para que ningun
        // planeta resulte invisible ni tape a los demas.
        cuerpo.radioCuerpo = 6.0f + 12.0f * distUnitaria(generador);

        // Color pseudoaleatorio con saturacion alta (se evita el gris apagado).
        cuerpo.colorR = 0.35f + 0.65f * distUnitaria(generador);
        cuerpo.colorG = 0.35f + 0.65f * distUnitaria(generador);
        cuerpo.colorB = 0.35f + 0.65f * distUnitaria(generador);

        // Posicion inicial coherente con el angulo inicial.
        cuerpo.posX = g_escena.centroX +
                      cuerpo.semiEjeMayor * std::cos(cuerpo.anguloOrbital);
        cuerpo.posY = g_escena.centroY +
                      cuerpo.semiEjeMenor * std::sin(cuerpo.anguloOrbital);

        g_escena.cuerpos.push_back(cuerpo);
    }
}

// ============================================================================
//  FASE DE CALCULO  (candidata a paralelizacion)
// ============================================================================

// Avanza la simulacion 'dt' segundos.
// Cada iteracion del bucle escribe UNICAMENTE en cuerpos[i], por lo que el
// bucle no tiene dependencias entre iteraciones y puede repartirse entre
// hilos. En esta prueba de concepto el pragma ya esta colocado para verificar
// que el toolchain de OpenMP funciona; la medicion formal de speedup queda
// para la siguiente entrega.
static void actualizarEscena(float dt) {
    // --- Sol: pulso de brillo/tamano (preludio del ciclo de supernova) ---
    g_escena.tiempoAcumulado += dt;
    g_escena.radioSolActual = g_escena.radioSolBase *
        (1.0f + 0.06f * std::sin(g_escena.tiempoAcumulado * 1.2f));

    // --- Cuerpos celestes: integracion del angulo orbital ---
    const int cantidad = static_cast<int>(g_escena.cuerpos.size());
    CuerpoCeleste* cuerpos = g_escena.cuerpos.data();
    const float centroX = g_escena.centroX;
    const float centroY = g_escena.centroY;

#ifdef USE_OPENMP
    // schedule(static): todas las iteraciones cuestan practicamente lo mismo
    // (dos llamadas trigonometricas), asi que un reparto en bloques iguales
    // minimiza el overhead de planificacion.
    // 'i' es privada por definicion; 'cuerpos', 'dt', 'centroX' y 'centroY'
    // son compartidas pero de solo lectura, salvo cuerpos[i] que cada hilo
    // escribe en exclusiva -> no hay condicion de carrera.
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < cantidad; ++i) {
        CuerpoCeleste& cuerpo = cuerpos[i];

        // Integracion explicita del angulo (Euler) y normalizacion a [0, 2PI)
        // para que el valor no crezca sin limite y pierda precision.
        cuerpo.anguloOrbital += cuerpo.velocidadAngular * dt;
        if (cuerpo.anguloOrbital >= 2.0f * PI) {
            cuerpo.anguloOrbital -= 2.0f * PI;
        } else if (cuerpo.anguloOrbital < 0.0f) {
            cuerpo.anguloOrbital += 2.0f * PI;
        }

        // Ecuacion parametrica de la elipse centrada en el Sol.
        cuerpo.posX = centroX + cuerpo.semiEjeMayor * std::cos(cuerpo.anguloOrbital);
        cuerpo.posY = centroY + cuerpo.semiEjeMenor * std::sin(cuerpo.anguloOrbital);
    }
    // Salida de la region paralela: barrera implicita de OpenMP. A partir de
    // aqui todas las posiciones estan actualizadas y el hilo principal puede
    // dibujar con seguridad.
}

// ============================================================================
//  FASE DE RENDERIZADO  (siempre secuencial, en el hilo principal)
// ============================================================================

// Dibuja un disco relleno mediante un abanico de triangulos.
static void dibujarDisco(float centroX, float centroY, float radio,
                         float r, float g, float b, float alfa) {
    glColor4f(r, g, b, alfa);
    glBegin(GL_TRIANGLE_FAN);
    glVertex2f(centroX, centroY); // vertice central del abanico
    for (int i = 0; i <= SEGMENTOS_CIRCULO; ++i) {
        const float angulo = 2.0f * PI * static_cast<float>(i) /
                             static_cast<float>(SEGMENTOS_CIRCULO);
        glVertex2f(centroX + radio * std::cos(angulo),
                   centroY + radio * std::sin(angulo));
    }
    glEnd();
}

// Dibuja la elipse de la orbita como una linea tenue, para dar contexto
// visual al movimiento (referencia del mapa del sistema solar).
static void dibujarOrbita(const CuerpoCeleste& cuerpo) {
    glColor4f(1.0f, 1.0f, 1.0f, 0.10f);
    glBegin(GL_LINE_LOOP);
    for (int i = 0; i < SEGMENTOS_CIRCULO * 3; ++i) {
        const float angulo = 2.0f * PI * static_cast<float>(i) /
                             static_cast<float>(SEGMENTOS_CIRCULO * 3);
        glVertex2f(g_escena.centroX + cuerpo.semiEjeMayor * std::cos(angulo),
                   g_escena.centroY + cuerpo.semiEjeMenor * std::sin(angulo));
    }
    glEnd();
}

// Dibuja el Sol: nucleo brillante mas varias capas de halo semitransparente.
static void dibujarSol() {
    const int capasHalo = 10;
    for (int capa = capasHalo; capa >= 1; --capa) {
        const float factor = 1.0f + 0.22f * static_cast<float>(capa);
        const float alfa   = 0.045f;
        dibujarDisco(g_escena.centroX, g_escena.centroY,
                     g_escena.radioSolActual * factor,
                     1.0f, 0.65f, 0.15f, alfa);
    }
    dibujarDisco(g_escena.centroX, g_escena.centroY, g_escena.radioSolActual,
                 1.0f, 0.85f, 0.35f, 1.0f);
}

// Dibuja una cadena de texto en coordenadas de ventana.
static void dibujarTexto(float x, float y, const std::string& texto) {
    glColor4f(1.0f, 1.0f, 1.0f, 0.85f);
    glRasterPos2f(x, y);
    for (size_t i = 0; i < texto.size(); ++i) {
        glutBitmapCharacter(GLUT_BITMAP_9_BY_15, texto[i]);
    }
}

// Callback de dibujo de GLUT.
static void alDibujar() {
    glClear(GL_COLOR_BUFFER_BIT);

    // Orbitas de referencia (solo si son pocas, para no saturar la pantalla).
    if (g_escena.cuerpos.size() <= 64) {
        for (size_t i = 0; i < g_escena.cuerpos.size(); ++i) {
            dibujarOrbita(g_escena.cuerpos[i]);
        }
    }

    dibujarSol();

    // Los cuerpos celestes: un halo tenue y el disco solido encima.
    for (size_t i = 0; i < g_escena.cuerpos.size(); ++i) {
        const CuerpoCeleste& cuerpo = g_escena.cuerpos[i];
        dibujarDisco(cuerpo.posX, cuerpo.posY, cuerpo.radioCuerpo * 1.8f,
                     cuerpo.colorR, cuerpo.colorG, cuerpo.colorB, 0.18f);
        dibujarDisco(cuerpo.posX, cuerpo.posY, cuerpo.radioCuerpo,
                     cuerpo.colorR, cuerpo.colorG, cuerpo.colorB, 1.0f);
    }

    // HUD: FPS y cantidad de cuerpos (requisito del enunciado).
    char lineaHud[128];
    std::snprintf(lineaHud, sizeof(lineaHud), "FPS: %6.2f   |   N = %zu",
                  g_escena.fpsActual, g_escena.cuerpos.size());
    dibujarTexto(10.0f, static_cast<float>(g_escena.altoVentana) - 22.0f,
                 std::string(lineaHud));

    glutSwapBuffers();
}

// ============================================================================
//  CALLBACKS DE GLUT
// ============================================================================

// Se invoca cuando la ventana cambia de tamano: reconstruye la proyeccion
// ortografica para que 1 unidad del mundo equivalga a 1 pixel.
static void alRedimensionar(int ancho, int alto) {
    if (ancho < 1) ancho = 1;
    if (alto  < 1) alto  = 1;

    g_escena.anchoVentana = ancho;
    g_escena.altoVentana  = alto;
    g_escena.centroX = static_cast<float>(ancho) * 0.5f;
    g_escena.centroY = static_cast<float>(alto)  * 0.5f;

    glViewport(0, 0, ancho, alto);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0.0, static_cast<double>(ancho), 0.0, static_cast<double>(alto));
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}

// Bucle de animacion: mide el tiempo real transcurrido (dt), actualiza la
// simulacion y solicita un nuevo cuadro. Usar dt real hace que la velocidad
// del screensaver sea independiente de los FPS de la maquina.
static void alEstarOcioso() {
    const std::chrono::steady_clock::time_point ahora =
        std::chrono::steady_clock::now();
    float dt = std::chrono::duration<float>(ahora - g_marcaTiempoPrevia).count();
    g_marcaTiempoPrevia = ahora;

    // Proteccion contra saltos grandes (ventana minimizada, breakpoint, etc.)
    if (dt < 0.0f)       dt = 0.0f;
    if (dt > DT_MAXIMO)  dt = DT_MAXIMO;

    actualizarEscena(dt);

    // Promedio de FPS sobre una ventana de tiempo fija, mas estable que 1/dt.
    g_escena.framesEnIntervalo += 1;
    g_escena.tiempoEnIntervalo += dt;
    if (g_escena.tiempoEnIntervalo >= INTERVALO_FPS) {
        g_escena.fpsActual = static_cast<float>(g_escena.framesEnIntervalo) /
                             g_escena.tiempoEnIntervalo;
        g_escena.framesEnIntervalo = 0;
        g_escena.tiempoEnIntervalo = 0.0f;
        std::printf("FPS= %.2f\n", g_escena.fpsActual);
        std::fflush(stdout);
    }

    glutPostRedisplay();
}

// Permite cerrar el screensaver con ESC o con la tecla 'q'.
static void alPresionarTecla(unsigned char tecla, int x, int y) {
    (void)x; (void)y; // parametros no usados
    if (tecla == 27 || tecla == 'q' || tecla == 'Q') {
        std::printf("Cerrando screensaver.\n");
        std::exit(EXIT_SUCCESS);
    }
}

// ============================================================================
//  PROGRAMA PRINCIPAL
// ============================================================================

int main(int argc, char** argv) {
    // --- Ayuda explicita ---
    // Se atiende ANTES de glutInit para que el usuario pueda consultarla
    // aunque no haya un servidor grafico disponible.
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-h") == 0 ||
            std::strcmp(argv[i], "--help") == 0) {
            imprimirUso(argv[0]);
            return EXIT_SUCCESS;
        }
    }

    // GLUT consume sus propios argumentos (-display, -geometry, ...) y deja el
    // resto en argv, por eso se inicializa antes de parsear los nuestros.
    glutInit(&argc, argv);

    // --- Valores por defecto ---
    long cantidadCuerpos = N_POR_DEFECTO;
    long anchoSolicitado = ANCHO_POR_DEFECTO;
    long altoSolicitado  = ALTO_POR_DEFECTO;
    unsigned int semilla = static_cast<unsigned int>(
        std::chrono::steady_clock::now().time_since_epoch().count());

    // --- Lectura defensiva de los argumentos ---
    if (argc > 5) {
        std::fprintf(stderr, "Error: demasiados argumentos (%d).\n", argc - 1);
        imprimirUso(argv[0]);
        return EXIT_FAILURE;
    }
    if (argc >= 2 && !leerEnteroValidado(argv[1], "N", N_MINIMO, N_MAXIMO,
                                         &cantidadCuerpos)) {
        imprimirUso(argv[0]);
        return EXIT_FAILURE;
    }
    if (argc >= 3 && !leerEnteroValidado(argv[2], "ancho", ANCHO_MINIMO,
                                         ANCHO_MAXIMO, &anchoSolicitado)) {
        imprimirUso(argv[0]);
        return EXIT_FAILURE;
    }
    if (argc >= 4 && !leerEnteroValidado(argv[3], "alto", ALTO_MINIMO,
                                         ALTO_MAXIMO, &altoSolicitado)) {
        imprimirUso(argv[0]);
        return EXIT_FAILURE;
    }
    if (argc >= 5) {
        long semillaLeida = 0;
        if (!leerEnteroValidado(argv[4], "semilla", 0, 2147483647,
                                &semillaLeida)) {
            imprimirUso(argv[0]);
            return EXIT_FAILURE;
        }
        semilla = static_cast<unsigned int>(semillaLeida);
    }

    // --- Creacion de la ventana / area grafica ---
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowSize(static_cast<int>(anchoSolicitado),
                       static_cast<int>(altoSolicitado));
    glutInitWindowPosition(80, 80);
    const int idVentana = glutCreateWindow(
        "Outer Wilds Screensaver - Prueba de concepto (UVG)");
    if (idVentana <= 0) {
        std::fprintf(stderr,
                     "Error: no se pudo crear la ventana OpenGL. Verifique que "
                     "exista un servidor grafico disponible.\n");
        return EXIT_FAILURE;
    }

    // Fondo azul muy oscuro (espacio) y mezcla alfa para los halos.
    glClearColor(0.02f, 0.02f, 0.07f, 1.0f);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_LINE_SMOOTH);

    // --- Estado inicial de la escena ---
    alRedimensionar(static_cast<int>(anchoSolicitado),
                    static_cast<int>(altoSolicitado));
    inicializarCuerpos(cantidadCuerpos, semilla);

    // --- Informacion de arranque ---
    std::printf("=== Outer Wilds Screensaver - Prueba de concepto ===\n");
    std::printf("Cuerpos celestes (N): %ld\n", cantidadCuerpos);
    std::printf("Ventana: %ldx%ld\n", anchoSolicitado, altoSolicitado);
    std::printf("Semilla: %u\n", semilla);
#ifdef USE_OPENMP
    std::printf("Version: PARALELA (OpenMP, hasta %d hilos)\n",
                omp_get_max_threads());
#else
    std::printf("Version: SECUENCIAL\n");
#endif
    std::printf("Presione ESC o 'q' sobre la ventana para salir.\n\n");
    std::fflush(stdout);

    // --- Registro de callbacks y arranque del bucle de eventos ---
    glutDisplayFunc(alDibujar);
    glutReshapeFunc(alRedimensionar);
    glutKeyboardFunc(alPresionarTecla);
    glutIdleFunc(alEstarOcioso);

    g_marcaTiempoPrevia = std::chrono::steady_clock::now();
    glutMainLoop();

    // glutMainLoop no retorna en la implementacion de macOS; la liberacion de
    // memoria queda a cargo del destructor de std::vector al terminar el
    // proceso (no se usa memoria cruda, por lo que no hay fugas).
    return EXIT_SUCCESS;
}