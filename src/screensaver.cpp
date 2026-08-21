

#include <GLUT/glut.h>
#include <OpenGL/gl.h>
#include <OpenGL/glu.h>

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
static const long  N_POR_DEFECTO     = 1;      // PoC: Sol + 1 cuerpo
static const int   ANCHO_POR_DEFECTO = 1024;
static const int   ALTO_POR_DEFECTO  = 768;

static const int   SEGMENTOS_CIRCULO = 24;     // resolucion del poligono
static const float PI                = 3.14159265358979323846f;
static const float DT_MAXIMO         = 0.05f;  // clamp anti-salto (50 ms)
static const float INTERVALO_FPS     = 0.5f;   // ventana de promedio de FPS

// ============================================================================
//  ESTRUCTURAS DE DATOS
// ============================================================================

struct CuerpoCeleste {
    float semiEjeMayor;     // radio horizontal de la orbita (pixeles)
    float semiEjeMenor;     // radio vertical de la orbita (pixeles)
    float anguloOrbital;    // angulo actual sobre la orbita (radianes)
    float velocidadAngular; // rad/s; derivada del radio (3a ley de Kepler)
    float radioCuerpo;      // radio dibujado del planeta (pixeles)
    float posX, posY;       // posicion resultante en coordenadas de ventana
    float colorR, colorG, colorB; // color pseudoaleatorio del cuerpo
};

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

        const float centroFranja = (static_cast<float>(i) + 0.5f) /
                                   static_cast<float>(cantidad);
        const float desplazamiento =
            0.5f * (distUnitaria(generador) - 0.5f) /
            static_cast<float>(cantidad);
        const float fraccion = centroFranja + desplazamiento;
        const float radioOrbital = radioOrbitaMinima +
            fraccion * (radioOrbitaMaxima - radioOrbitaMinima);

        const float achatamiento = 0.70f + 0.30f * distUnitaria(generador);
        cuerpo.semiEjeMayor = radioOrbital;
        cuerpo.semiEjeMenor = radioOrbital * achatamiento;

        cuerpo.anguloOrbital = distUnitaria(generador) * 2.0f * PI;

        const float constanteGravitacional = 90000.0f;
        const float magnitudVelocidad =
            constanteGravitacional / std::pow(radioOrbital, 1.5f);
        const float sentido = (distUnitaria(generador) < 0.15f) ? -1.0f : 1.0f;
        cuerpo.velocidadAngular = sentido * magnitudVelocidad;

        cuerpo.radioCuerpo = 6.0f + 12.0f * distUnitaria(generador);

        cuerpo.colorR = 0.35f + 0.65f * distUnitaria(generador);
        cuerpo.colorG = 0.35f + 0.65f * distUnitaria(generador);
        cuerpo.colorB = 0.35f + 0.65f * distUnitaria(generador);

        cuerpo.posX = g_escena.centroX +
                      cuerpo.semiEjeMayor * std::cos(cuerpo.anguloOrbital);
        cuerpo.posY = g_escena.centroY +
                      cuerpo.semiEjeMenor * std::sin(cuerpo.anguloOrbital);

        g_escena.cuerpos.push_back(cuerpo);
    }
}

// ============================================================================
//  FASE DE CALCULO
// ============================================================================

static void actualizarEscena(float dt) {
    // --- Sol: pulso de brillo/tamano ---
    g_escena.tiempoAcumulado += dt;
    g_escena.radioSolActual = g_escena.radioSolBase *
        (1.0f + 0.06f * std::sin(g_escena.tiempoAcumulado * 1.2f));

    // --- Cuerpos celestes: integracion del angulo orbital ---
    const int cantidad = static_cast<int>(g_escena.cuerpos.size());
    CuerpoCeleste* cuerpos = g_escena.cuerpos.data();
    const float centroX = g_escena.centroX;
    const float centroY = g_escena.centroY;

#ifdef USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < cantidad; ++i) {
        CuerpoCeleste& cuerpo = cuerpos[i];

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

// Dibuja la elipse de la orbita como una linea tenue
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

// Bucle de animacion
static void alEstarOcioso() {
    const std::chrono::steady_clock::time_point ahora =
        std::chrono::steady_clock::now();
    float dt = std::chrono::duration<float>(ahora - g_marcaTiempoPrevia).count();
    g_marcaTiempoPrevia = ahora;

    if (dt < 0.0f)       dt = 0.0f;
    if (dt > DT_MAXIMO)  dt = DT_MAXIMO;

    actualizarEscena(dt);

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

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-h") == 0 ||
            std::strcmp(argv[i], "--help") == 0) {
            imprimirUso(argv[0]);
            return EXIT_SUCCESS;
        }
    }

    glutInit(&argc, argv);

    long cantidadCuerpos = N_POR_DEFECTO;
    long anchoSolicitado = ANCHO_POR_DEFECTO;
    long altoSolicitado  = ALTO_POR_DEFECTO;
    unsigned int semilla = static_cast<unsigned int>(
        std::chrono::steady_clock::now().time_since_epoch().count());

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

    // --- Creacion de la ventana ---
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

    return EXIT_SUCCESS;
}
