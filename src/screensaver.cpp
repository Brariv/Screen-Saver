// ============================================================================
//  Screensaver Paralelo: Sistema Solar de Outer Wilds
//  Universidad del Valle de Guatemala - Computacion Paralela y Distribuida
//  Proyecto 1 - Entrega 2: Prueba de concepto
//
//  Integrantes: Javier Cifuentes (23079), Brandon Rivera (23088)
//
//  Alcance de esta entrega (prueba de concepto):
//    1. Creacion de la ventana / area grafica donde correra el screensaver.
//    2. Visualizacion de los elementos definidos en la propuesta: el Sol
//       central y N cuerpos celestes orbitando a su alrededor.
//    3. Cada cuerpo se mueve con un MODELO FISICO DISTINTO, de modo que el
//       hilo que atiende a un cuerpo ejecuta un calculo diferente al del hilo
//       vecino. Esto convierte el bucle de actualizacion en una carga de
//       trabajo HETEROGENEA, que es el caso interesante para estudiar el
//       balanceo de carga de OpenMP (static vs dynamic).
//
//  Modelos de fisica implementados (uno por cuerpo, ver enum TipoFisica):
//    - CIRCULAR   : integracion de Euler sobre una elipse a velocidad angular
//                   constante. Costo fijo y bajo. (planetas "normales")
//    - KEPLERIANA : orbita de dos cuerpos real. Resuelve la ecuacion de Kepler
//                   M = E - e*sin(E) por Newton-Raphson en cada cuadro, con lo
//                   que la velocidad varia a lo largo de la orbita (rapido en
//                   el perihelio, lento en el afelio). Costo VARIABLE segun
//                   cuantas iteraciones necesite converger. (The Interloper)
//    - BINARIA    : par de cuerpos girando alrededor de un baricentro comun
//                   que a su vez orbita al Sol, mas transferencia de "arena"
//                   sobre un CONTADOR COMPARTIDO protegido con #pragma omp
//                   atomic. (Hourglass Twins: Ash Twin y Ember Twin)
//    - SATELITE   : luna cuya posicion DEPENDE de la de su cuerpo padre. Se
//                   resuelve en una segunda fase, despues de una barrera.
//                   (The Attlerock)
//    - CUANTICA   : maquina de estados discreta; el cuerpo se reposiciona de
//                   golpe cada cierto intervalo usando una funcion de hash
//                   determinista (sin estado compartido, segura entre hilos).
//                   (The Quantum Moon)
//    - OSCILANTE  : orbita perturbada por una suma de K armonicos senoidales.
//                   Su costo crece con K, distinto en cada cuerpo, por lo que
//                   es la fuente principal de desbalance. (Dark Bramble)
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
static const long  N_POR_DEFECTO     = 8;      // Sol + unos cuantos cuerpos
static const int   ANCHO_POR_DEFECTO = 1024;
static const int   ALTO_POR_DEFECTO  = 768;

static const int   SEGMENTOS_CIRCULO = 24;     // resolucion del poligono
static const float PI                = 3.14159265358979323846f;
static const float DT_MAXIMO         = 0.05f;  // clamp anti-salto (50 ms)
static const float INTERVALO_FPS     = 0.5f;   // ventana de promedio de FPS

// Constante gravitacional del modelo (unidades de pantalla). Calibrada para
// que un cuerpo a media distancia complete una vuelta en unos 30 segundos:
// un movimiento lento y contemplativo, acorde a un screensaver.
static const float CONSTANTE_GRAVITACIONAL = 600.0f;

// Tolerancia y tope de iteraciones del solucionador de la ecuacion de Kepler.
static const float TOLERANCIA_KEPLER       = 1.0e-6f;
static const int   MAX_ITERACIONES_KEPLER  = 12;

// ============================================================================
//  TIPOS DE FISICA
// ============================================================================

// Cada cuerpo lleva uno de estos valores. El hilo que le toque ese cuerpo
// entrara a una rama distinta del switch de actualizarEscena() y por lo tanto
// ejecutara un calculo fisico diferente, con un costo diferente.
enum TipoFisica {
    FISICA_CIRCULAR = 0,   // Euler sobre elipse, velocidad angular constante
    FISICA_KEPLERIANA,     // ecuacion de Kepler resuelta por Newton-Raphson
    FISICA_BINARIA,        // par alrededor de un baricentro + recurso compartido
    FISICA_SATELITE,       // depende de la posicion de su cuerpo padre
    FISICA_CUANTICA,       // reposicionamiento discreto por hash determinista
    FISICA_OSCILANTE,      // orbita perturbada por K armonicos senoidales
    CANTIDAD_TIPOS_FISICA
};

static const char* NOMBRES_FISICA[CANTIDAD_TIPOS_FISICA] = {
    "circular", "kepler", "binaria", "satelite", "cuantica", "oscilante"
};

// ============================================================================
//  ESTRUCTURAS DE DATOS
// ============================================================================

// Un cuerpo celeste. Salvo el caso del satelite (que lee a su padre) y el de
// los gemelos (que comparten el contador de arena), cada instancia se actualiza
// leyendo y escribiendo UNICAMENTE sus propios campos, lo que permite repartir
// el arreglo entre hilos sin condiciones de carrera.
struct CuerpoCeleste {
    TipoFisica tipo;

    // --- Orbita base (la usan casi todos los tipos) ---
    float semiEjeMayor;      // radio horizontal de la orbita (px)
    float semiEjeMenor;      // radio vertical de la orbita (px)
    float anguloOrbital;     // angulo actual sobre la orbita (rad)
    float velocidadAngular;  // rad/s

    // --- FISICA_KEPLERIANA ---
    float excentricidad;      // e en [0,1)
    float anomaliaMedia;      // M (rad)
    float tasaAnomaliaMedia;  // n = 2*PI/T (rad/s)
    float argumentoPeriapsis; // rotacion de la elipse en el plano (rad)
    int   iteracionesNewton;  // iteraciones del ultimo cuadro (diagnostico)

    // --- FISICA_BINARIA ---
    float anguloBaricentro;     // angulo del baricentro alrededor del Sol
    float velocidadBaricentro;  // rad/s del baricentro
    float radioBaricentro;      // radio de la orbita del baricentro
    float anguloLocal;          // angulo del gemelo alrededor del baricentro
    float velocidadLocal;       // rad/s del giro local
    float radioLocal;           // separacion respecto al baricentro
    float desfaseLocal;         // 0 para un gemelo, PI para el otro
    bool  esDonanteDeArena;     // Ember cede arena, Ash la recibe

    // --- FISICA_SATELITE ---
    int   indicePadre;          // indice del cuerpo alrededor del cual gira

    // --- FISICA_CUANTICA ---
    float tiempoParaSalto;      // segundos restantes hasta el proximo salto
    float intervaloSalto;       // periodo entre saltos
    unsigned int contadorSaltos;// cuantos saltos lleva (entra al hash)

    // --- FISICA_OSCILANTE ---
    int   armonicos;            // K terminos de la serie (costo del cuerpo)
    float amplitudOscilacion;   // amplitud relativa de la perturbacion
    float faseOscilacion;       // desfase base de la serie

    // --- Comunes ---
    float radioBase;            // radio nominal dibujado (px)
    float radioCuerpo;          // radio efectivo del cuadro actual (px)
    float posX, posY;           // posicion resultante en coordenadas de ventana
    float colorR, colorG, colorB;
    int   hiloAsignado;         // id del hilo que lo actualizo (diagnostico)
};

// Estado global de la escena. GLUT trabaja con callbacks sin parametros de
// usuario, por lo que el estado debe ser accesible globalmente.
struct EstadoEscena {
    std::vector<CuerpoCeleste> cuerpos;

    // El arreglo se recorre en dos fases porque los satelites dependen de la
    // posicion ya actualizada de su padre. Guardar los indices separados evita
    // meter un 'if' dentro del bucle paralelo.
    std::vector<int> indicesPrimarios;  // todos los que NO son satelites
    std::vector<int> indicesSatelites;  // los que dependen de un padre

    int conteoPorTipo[CANTIDAD_TIPOS_FISICA];

    // RECURSO COMPARTIDO entre los gemelos: cuanta arena se ha transferido.
    // Varios hilos lo modifican en el mismo cuadro -> se protege con atomic.
    float arenaTransferida;

    int   anchoVentana;
    int   altoVentana;
    float centroX, centroY;   // posicion del Sol (centro de la ventana)
    float radioSolBase;
    float radioSolActual;
    float tiempoAcumulado;

    // Medicion de FPS
    int   framesEnIntervalo;
    float tiempoEnIntervalo;
    float fpsActual;

    EstadoEscena()
        : arenaTransferida(0.0f),
          anchoVentana(ANCHO_POR_DEFECTO), altoVentana(ALTO_POR_DEFECTO),
          centroX(0.0f), centroY(0.0f), radioSolBase(40.0f),
          radioSolActual(40.0f), tiempoAcumulado(0.0f),
          framesEnIntervalo(0), tiempoEnIntervalo(0.0f), fpsActual(0.0f) {
        for (int i = 0; i < CANTIDAD_TIPOS_FISICA; ++i) conteoPorTipo[i] = 0;
    }
};

static EstadoEscena g_escena;
static std::chrono::steady_clock::time_point g_marcaTiempoPrevia;
static int g_hilosDisponibles = 1;

// ============================================================================
//  UTILIDADES
// ============================================================================

// Mantiene un angulo dentro de [0, 2*PI) para que no crezca sin limite y
// termine perdiendo precision en punto flotante.
static inline float normalizarAngulo(float angulo) {
    while (angulo >= 2.0f * PI) angulo -= 2.0f * PI;
    while (angulo <  0.0f)      angulo += 2.0f * PI;
    return angulo;
}

// Funcion de mezcla de enteros (variante de MurmurHash3 finalizer).
// Se usa para el salto cuantico: produce numeros que parecen aleatorios pero
// son deterministas y NO requieren estado compartido, asi que cualquier hilo
// puede evaluarla sin sincronizacion y el resultado es reproducible.
static inline unsigned int mezclarEntero(unsigned int x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

// Convierte un entero mezclado en un flotante dentro de [0,1).
static inline float aleatorioDeterminista(unsigned int semilla) {
    return static_cast<float>(mezclarEntero(semilla) & 0x00FFFFFFu) /
           static_cast<float>(0x01000000u);
}

// ============================================================================
//  MODELOS DE FISICA - uno por tipo de cuerpo
//  Cada una de estas funciones es la que ejecuta el hilo al que le toco ese
//  cuerpo. Son deliberadamente distintas entre si, tanto en las ecuaciones
//  como en el costo computacional.
// ============================================================================

// ---------------------------------------------------------------------------
// 1) FISICA_CIRCULAR - integracion de Euler a velocidad angular constante.
//    Es el modelo mas barato: dos llamadas trigonometricas por cuadro.
// ---------------------------------------------------------------------------
static void fisicaCircular(CuerpoCeleste& c, float dt, float cx, float cy) {
    c.anguloOrbital = normalizarAngulo(c.anguloOrbital +
                                       c.velocidadAngular * dt);
    c.posX = cx + c.semiEjeMayor * std::cos(c.anguloOrbital);
    c.posY = cy + c.semiEjeMenor * std::sin(c.anguloOrbital);
    c.radioCuerpo = c.radioBase;
}

// ---------------------------------------------------------------------------
// 2) FISICA_KEPLERIANA - problema de dos cuerpos resuelto de verdad.
//
//    Se avanza la anomalia media M = M0 + n*t y se despeja la anomalia
//    excentrica E de la ecuacion trascendente de Kepler
//
//         M = E - e*sin(E)
//
//    mediante Newton-Raphson:  E <- E - (E - e*sin E - M) / (1 - e*cos E)
//
//    Con la E ya resuelta, la posicion respecto al foco (donde esta el Sol) es
//         x' = a*(cos E - e)
//         y' = a*sqrt(1 - e^2)*sin E
//    y se rota por el argumento del periapsis.
//
//    Consecuencia fisica: la velocidad NO es constante; el cuerpo acelera al
//    acercarse al Sol y se frena al alejarse (segunda ley de Kepler).
//    Consecuencia computacional: el numero de iteraciones depende de la
//    excentricidad y de la posicion en la orbita, asi que el costo de este
//    cuerpo VARIA de un cuadro a otro. Es el caso que rompe el balanceo
//    estatico de carga.
// ---------------------------------------------------------------------------
static void fisicaKepleriana(CuerpoCeleste& c, float dt, float cx, float cy) {
    c.anomaliaMedia = normalizarAngulo(c.anomaliaMedia +
                                       c.tasaAnomaliaMedia * dt);

    // Semilla clasica para orbitas excentricas.
    float anomaliaExcentrica = c.anomaliaMedia +
        c.excentricidad * std::sin(c.anomaliaMedia);

    int iteracion = 0;
    for (; iteracion < MAX_ITERACIONES_KEPLER; ++iteracion) {
        const float residuo = anomaliaExcentrica -
            c.excentricidad * std::sin(anomaliaExcentrica) - c.anomaliaMedia;
        const float derivada = 1.0f -
            c.excentricidad * std::cos(anomaliaExcentrica);
        const float paso = residuo / derivada;
        anomaliaExcentrica -= paso;
        if (std::fabs(paso) < TOLERANCIA_KEPLER) {
            break;
        }
    }
    c.iteracionesNewton = iteracion + 1;

    const float semiEjeB = c.semiEjeMayor *
        std::sqrt(1.0f - c.excentricidad * c.excentricidad);
    const float xLocal = c.semiEjeMayor *
        (std::cos(anomaliaExcentrica) - c.excentricidad);
    const float yLocal = semiEjeB * std::sin(anomaliaExcentrica);

    // Rotacion de la elipse por el argumento del periapsis.
    const float cosArg = std::cos(c.argumentoPeriapsis);
    const float senArg = std::sin(c.argumentoPeriapsis);
    c.posX = cx + xLocal * cosArg - yLocal * senArg;
    c.posY = cy + xLocal * senArg + yLocal * cosArg;
    c.radioCuerpo = c.radioBase;
}

// ---------------------------------------------------------------------------
// 3) FISICA_BINARIA - los Hourglass Twins.
//
//    Dos cuerpos giran alrededor de un baricentro comun, y ese baricentro a su
//    vez orbita al Sol. Ambos gemelos integran los MISMOS parametros de
//    baricentro con los mismos valores iniciales, por lo que llegan al mismo
//    resultado sin necesidad de comunicarse: no hay dependencia entre ellos
//    para la posicion.
//
//    Lo que si comparten es el contador de arena transferida entre ambos. Los
//    dos hilos lo modifican en el mismo cuadro, asi que es una autentica
//    seccion critica: se protege con #pragma omp atomic (update y read).
// ---------------------------------------------------------------------------
static void fisicaBinaria(CuerpoCeleste& c, float dt, float cx, float cy) {
    // (a) El baricentro del par recorre su propia orbita alrededor del Sol.
    c.anguloBaricentro = normalizarAngulo(c.anguloBaricentro +
                                          c.velocidadBaricentro * dt);
    const float bx = cx + c.radioBaricentro * std::cos(c.anguloBaricentro);
    const float by = cy + c.radioBaricentro * 0.85f *
                          std::sin(c.anguloBaricentro);

    // (b) Cada gemelo gira alrededor de ese baricentro, separados por PI.
    c.anguloLocal = normalizarAngulo(c.anguloLocal + c.velocidadLocal * dt);
    const float anguloEfectivo = c.anguloLocal + c.desfaseLocal;
    c.posX = bx + c.radioLocal * std::cos(anguloEfectivo);
    c.posY = by + c.radioLocal * std::sin(anguloEfectivo);

    // (c) Transferencia de arena sobre el contador COMPARTIDO.
    //     El donante aporta y el receptor descuenta; sin proteccion, las dos
    //     escrituras simultaneas se perderian (condicion de carrera clasica).
    const float aporte = (c.esDonanteDeArena ? 0.25f : -0.15f) * dt;
    float arenaLeida;
#ifdef USE_OPENMP
#pragma omp atomic update
    g_escena.arenaTransferida += aporte;
#pragma omp atomic read
    arenaLeida = g_escena.arenaTransferida;
#else
    g_escena.arenaTransferida += aporte;
    arenaLeida = g_escena.arenaTransferida;
#endif

    // El nivel de arena hace crecer a uno mientras el otro se encoge.
    const float fraccion = 0.5f + 0.5f * std::sin(arenaLeida);
    c.radioCuerpo = c.radioBase * (c.esDonanteDeArena
                                       ? (1.25f - 0.50f * fraccion)
                                       : (0.75f + 0.50f * fraccion));
}

// ---------------------------------------------------------------------------
// 4) FISICA_CUANTICA - The Quantum Moon.
//
//    No integra nada: es una maquina de estados. Cuando se agota el temporizador
//    el cuerpo desaparece y reaparece en otro punto, elegido con una funcion de
//    hash sobre (indice, numero de salto). Al ser una funcion pura, cualquier
//    hilo puede evaluarla sin estado compartido ni bloqueos, y la escena sigue
//    siendo reproducible entre la version secuencial y la paralela.
// ---------------------------------------------------------------------------
static void fisicaCuantica(CuerpoCeleste& c, int indice, float dt,
                           float cx, float cy, float radioMaximo) {
    c.tiempoParaSalto -= dt;
    if (c.tiempoParaSalto <= 0.0f) {
        c.contadorSaltos += 1u;
        const unsigned int base =
            static_cast<unsigned int>(indice) * 2654435761u ^
            (c.contadorSaltos * 40503u);

        c.anguloOrbital = aleatorioDeterminista(base) * 2.0f * PI;
        const float fraccionRadio = 0.35f + 0.60f * aleatorioDeterminista(base + 1u);
        c.semiEjeMayor = radioMaximo * fraccionRadio;
        c.semiEjeMenor = c.semiEjeMayor *
                         (0.70f + 0.30f * aleatorioDeterminista(base + 2u));
        c.tiempoParaSalto = c.intervaloSalto;
    }

    // Entre saltos permanece practicamente inmovil (solo deriva muy despacio).
    c.anguloOrbital = normalizarAngulo(c.anguloOrbital +
                                       c.velocidadAngular * 0.15f * dt);
    c.posX = cx + c.semiEjeMayor * std::cos(c.anguloOrbital);
    c.posY = cy + c.semiEjeMenor * std::sin(c.anguloOrbital);
    c.radioCuerpo = c.radioBase;
}

// ---------------------------------------------------------------------------
// 5) FISICA_OSCILANTE - Dark Bramble.
//
//    Orbita normal, pero el radio se perturba con una serie de K armonicos
//
//        r(t) = r0 * ( 1 + A * SUM_{k=1..K} sin(f_k * t + k*fase) / k )
//
//    que imita un ruido suave (los zarcillos ondulando). Como K es distinto en
//    cada cuerpo, el costo de este modelo varia mucho de un cuerpo a otro: es
//    la fuente principal de desbalance de carga entre hilos.
// ---------------------------------------------------------------------------
static void fisicaOscilante(CuerpoCeleste& c, float dt, float tiempo,
                            float cx, float cy) {
    c.anguloOrbital = normalizarAngulo(c.anguloOrbital +
                                       c.velocidadAngular * dt);

    float perturbacion = 0.0f;
    for (int k = 1; k <= c.armonicos; ++k) {
        const float frecuencia = 0.35f + 0.22f * static_cast<float>(k);
        perturbacion += std::sin(frecuencia * tiempo +
                                 static_cast<float>(k) * c.faseOscilacion) /
                        static_cast<float>(k);
    }

    const float factorRadio = 1.0f + c.amplitudOscilacion * perturbacion;
    c.posX = cx + c.semiEjeMayor * factorRadio * std::cos(c.anguloOrbital);
    c.posY = cy + c.semiEjeMenor * factorRadio * std::sin(c.anguloOrbital);
    c.radioCuerpo = c.radioBase;
}

// ---------------------------------------------------------------------------
// 6) FISICA_SATELITE - The Attlerock.
//
//    Unico modelo con DEPENDENCIA DE DATOS: necesita la posicion ya actualizada
//    de su cuerpo padre. Por eso se ejecuta en una segunda fase, despues de la
//    barrera que cierra la fase de los cuerpos primarios.
// ---------------------------------------------------------------------------
static void fisicaSatelite(CuerpoCeleste& c, const CuerpoCeleste& padre,
                           float dt) {
    c.anguloOrbital = normalizarAngulo(c.anguloOrbital +
                                       c.velocidadAngular * dt);
    c.posX = padre.posX + c.radioLocal * std::cos(c.anguloOrbital);
    c.posY = padre.posY + c.radioLocal * 0.9f * std::sin(c.anguloOrbital);
    c.radioCuerpo = c.radioBase;
}

// ============================================================================
//  FASE DE CALCULO  (region paralela)
// ============================================================================

// Avanza la simulacion 'dt' segundos.
//
// Estructura de la region paralela:
//
//    #pragma omp parallel
//    {
//        #pragma omp for schedule(dynamic,1)   <- cuerpos primarios
//        ...                                      (cada uno con SU fisica)
//        ---- barrera implicita del omp for ----
//        #pragma omp for schedule(static)      <- satelites (leen a su padre)
//    }
//
// Se usa schedule(dynamic,1) en la primera fase justamente porque el costo por
// cuerpo es muy desigual (un cuerpo oscilante con 30 armonicos cuesta ordenes
// de magnitud mas que uno circular); con reparto estatico un hilo terminaria
// mucho antes que otro. En la segunda fase, en cambio, todos los satelites
// cuestan lo mismo, asi que el reparto estatico es el adecuado.
static void actualizarEscena(float dt) {
    // --- Sol: pulso de brillo/tamano (preludio del ciclo de supernova) ---
    g_escena.tiempoAcumulado += dt;
    g_escena.radioSolActual = g_escena.radioSolBase *
        (1.0f + 0.06f * std::sin(g_escena.tiempoAcumulado * 1.2f));

    CuerpoCeleste* cuerpos      = g_escena.cuerpos.data();
    const int*     primarios    = g_escena.indicesPrimarios.data();
    const int      numPrimarios = static_cast<int>(g_escena.indicesPrimarios.size());
    const int*     satelites    = g_escena.indicesSatelites.data();
    const int      numSatelites = static_cast<int>(g_escena.indicesSatelites.size());

    const float cx     = g_escena.centroX;
    const float cy     = g_escena.centroY;
    const float tiempo = g_escena.tiempoAcumulado;
    const float radioMaximo = 0.46f * static_cast<float>(
        g_escena.anchoVentana < g_escena.altoVentana ? g_escena.anchoVentana
                                                     : g_escena.altoVentana);

#ifdef USE_OPENMP
#pragma omp parallel
#endif
    {
        // ---------------- Fase 1: cuerpos independientes -------------------
        // Cada iteracion entra a una rama distinta segun el tipo de fisica,
        // de modo que hilos distintos ejecutan calculos distintos.
#ifdef USE_OPENMP
#pragma omp for schedule(dynamic, 1)
#endif
        for (int k = 0; k < numPrimarios; ++k) {
            const int indice = primarios[k];
            CuerpoCeleste& cuerpo = cuerpos[indice];

#ifdef USE_OPENMP
            cuerpo.hiloAsignado = omp_get_thread_num();
#else
            cuerpo.hiloAsignado = 0;
#endif

            switch (cuerpo.tipo) {
                case FISICA_KEPLERIANA:
                    fisicaKepleriana(cuerpo, dt, cx, cy);
                    break;
                case FISICA_BINARIA:
                    fisicaBinaria(cuerpo, dt, cx, cy);
                    break;
                case FISICA_CUANTICA:
                    fisicaCuantica(cuerpo, indice, dt, cx, cy, radioMaximo);
                    break;
                case FISICA_OSCILANTE:
                    fisicaOscilante(cuerpo, dt, tiempo, cx, cy);
                    break;
                case FISICA_CIRCULAR:
                default:
                    fisicaCircular(cuerpo, dt, cx, cy);
                    break;
            }
        }
        // Barrera implicita al cerrar el 'omp for': a partir de aqui todos los
        // cuerpos primarios tienen su posicion definitiva del cuadro.

        // ---------------- Fase 2: satelites (dependientes) ------------------
#ifdef USE_OPENMP
#pragma omp for schedule(static)
#endif
        for (int k = 0; k < numSatelites; ++k) {
            const int indice = satelites[k];
            CuerpoCeleste& cuerpo = cuerpos[indice];

#ifdef USE_OPENMP
            cuerpo.hiloAsignado = omp_get_thread_num();
#else
            cuerpo.hiloAsignado = 0;
#endif
            fisicaSatelite(cuerpo, cuerpos[cuerpo.indicePadre], dt);
        }
    }
    // Salida de la region paralela: barrera final. El hilo principal ya puede
    // dibujar con seguridad.
}

// ============================================================================
//  INICIALIZACION DE LA ESCENA
// ============================================================================

// Reparte los tipos de fisica de forma ciclica sobre los N cuerpos, cuidando
// que las combinaciones que necesitan un companero o un padre sean validas.
static TipoFisica tipoParaIndice(long indice, long cantidad) {
    switch (indice % 7) {
        case 0: return FISICA_CIRCULAR;
        case 1: return FISICA_KEPLERIANA;
        // Un binario necesita dos cuerpos: si no cabe el companero, degrada.
        case 2: return (indice + 1 < cantidad) ? FISICA_BINARIA : FISICA_CIRCULAR;
        case 3: return FISICA_BINARIA;              // companero del anterior
        case 4: return FISICA_SATELITE;             // padre = indice - 1
        case 5: return FISICA_CUANTICA;
        default: return FISICA_OSCILANTE;
    }
}

// Genera los N cuerpos con sus parametros orbitales y su tipo de fisica.
// - Los radios se reparten por muestreo estratificado para que las orbitas
//   queden bien distribuidas tanto con N=1 como con N muy grande.
// - La velocidad angular se deriva del radio segun la tercera ley de Kepler
//   (w proporcional a r^-3/2): los cuerpos cercanos giran mas rapido.
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
    g_escena.indicesPrimarios.clear();
    g_escena.indicesSatelites.clear();
    g_escena.arenaTransferida = 0.0f;
    for (int t = 0; t < CANTIDAD_TIPOS_FISICA; ++t) g_escena.conteoPorTipo[t] = 0;

    for (long i = 0; i < cantidad; ++i) {
        CuerpoCeleste cuerpo;
        std::memset(&cuerpo, 0, sizeof(cuerpo));
        cuerpo.tipo = tipoParaIndice(i, cantidad);

        // --- Radio orbital por muestreo estratificado ---
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

        // Tercera ley de Kepler simplificada: w = k * r^(-3/2).
        const float magnitudVelocidad =
            CONSTANTE_GRAVITACIONAL / std::pow(radioOrbital, 1.5f);
        const float sentido = (distUnitaria(generador) < 0.15f) ? -1.0f : 1.0f;
        cuerpo.velocidadAngular = sentido * magnitudVelocidad;

        cuerpo.radioBase   = 6.0f + 12.0f * distUnitaria(generador);
        cuerpo.radioCuerpo = cuerpo.radioBase;
        cuerpo.colorR = 0.35f + 0.65f * distUnitaria(generador);
        cuerpo.colorG = 0.35f + 0.65f * distUnitaria(generador);
        cuerpo.colorB = 0.35f + 0.65f * distUnitaria(generador);
        cuerpo.indicePadre = -1;
        cuerpo.hiloAsignado = 0;

        // --- Parametros especificos de cada modelo de fisica ---
        switch (cuerpo.tipo) {
            case FISICA_KEPLERIANA: {
                // Orbita muy excentrica, como la de The Interloper. El semieje
                // se ajusta para que el afelio no se salga de la ventana.
                cuerpo.excentricidad = 0.55f + 0.30f * distUnitaria(generador);
                cuerpo.semiEjeMayor = radioOrbital / (1.0f + cuerpo.excentricidad);
                cuerpo.argumentoPeriapsis = distUnitaria(generador) * 2.0f * PI;
                cuerpo.anomaliaMedia = distUnitaria(generador) * 2.0f * PI;
                cuerpo.tasaAnomaliaMedia = CONSTANTE_GRAVITACIONAL /
                    std::pow(cuerpo.semiEjeMayor, 1.5f);
                cuerpo.iteracionesNewton = 0;
                break;
            }
            case FISICA_BINARIA: {
                const bool esSegundoGemelo = (i % 7 == 3);
                if (esSegundoGemelo && !g_escena.cuerpos.empty()) {
                    // Copia los parametros del baricentro de su companero para
                    // que ambos giren exactamente alrededor del mismo punto.
                    const CuerpoCeleste& companero =
                        g_escena.cuerpos[static_cast<size_t>(i - 1)];
                    cuerpo.radioBaricentro     = companero.radioBaricentro;
                    cuerpo.anguloBaricentro    = companero.anguloBaricentro;
                    cuerpo.velocidadBaricentro = companero.velocidadBaricentro;
                    cuerpo.radioLocal          = companero.radioLocal;
                    cuerpo.velocidadLocal      = companero.velocidadLocal;
                    cuerpo.anguloLocal         = companero.anguloLocal;
                    cuerpo.desfaseLocal        = PI;   // gemelo opuesto
                    cuerpo.esDonanteDeArena    = false; // Ash: recibe
                    cuerpo.radioBase           = companero.radioBase;
                } else {
                    cuerpo.radioBaricentro     = radioOrbital;
                    cuerpo.anguloBaricentro    = cuerpo.anguloOrbital;
                    cuerpo.velocidadBaricentro = cuerpo.velocidadAngular;
                    cuerpo.radioLocal          = 16.0f + 10.0f * distUnitaria(generador);
                    cuerpo.velocidadLocal      = 5.0f * magnitudVelocidad;
                    cuerpo.anguloLocal         = distUnitaria(generador) * 2.0f * PI;
                    cuerpo.desfaseLocal        = 0.0f;
                    cuerpo.esDonanteDeArena    = true;  // Ember: cede
                }
                break;
            }
            case FISICA_SATELITE: {
                cuerpo.indicePadre = static_cast<int>(i) - 1;
                if (cuerpo.indicePadre < 0) cuerpo.indicePadre = 0;
                cuerpo.radioBase   = 4.0f + 3.0f * distUnitaria(generador);
                cuerpo.radioCuerpo = cuerpo.radioBase;
                cuerpo.radioLocal  = 26.0f + 14.0f * distUnitaria(generador);
                // La luna gira bastante mas rapido que su planeta.
                cuerpo.velocidadAngular = 8.0f * magnitudVelocidad;
                break;
            }
            case FISICA_CUANTICA: {
                cuerpo.intervaloSalto  = 3.0f + 4.0f * distUnitaria(generador);
                cuerpo.tiempoParaSalto = cuerpo.intervaloSalto *
                                         distUnitaria(generador);
                cuerpo.contadorSaltos  = 0u;
                break;
            }
            case FISICA_OSCILANTE: {
                // K distinto por cuerpo: de aqui sale el desbalance de carga.
                cuerpo.armonicos = 8 + static_cast<int>(
                    24.0f * distUnitaria(generador));
                cuerpo.amplitudOscilacion = 0.05f + 0.07f * distUnitaria(generador);
                cuerpo.faseOscilacion     = distUnitaria(generador) * 2.0f * PI;
                break;
            }
            case FISICA_CIRCULAR:
            default:
                break;
        }

        // Posicion inicial coherente (se recalcula en el primer cuadro).
        cuerpo.posX = g_escena.centroX +
                      cuerpo.semiEjeMayor * std::cos(cuerpo.anguloOrbital);
        cuerpo.posY = g_escena.centroY +
                      cuerpo.semiEjeMenor * std::sin(cuerpo.anguloOrbital);

        g_escena.conteoPorTipo[cuerpo.tipo] += 1;
        g_escena.cuerpos.push_back(cuerpo);

        if (cuerpo.tipo == FISICA_SATELITE) {
            g_escena.indicesSatelites.push_back(static_cast<int>(i));
        } else {
            g_escena.indicesPrimarios.push_back(static_cast<int>(i));
        }
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

// Contorno circular sin relleno (se usa para marcar la luna cuantica).
static void dibujarContorno(float centroX, float centroY, float radio,
                            float r, float g, float b, float alfa) {
    glColor4f(r, g, b, alfa);
    glBegin(GL_LINE_LOOP);
    for (int i = 0; i < SEGMENTOS_CIRCULO; ++i) {
        const float angulo = 2.0f * PI * static_cast<float>(i) /
                             static_cast<float>(SEGMENTOS_CIRCULO);
        glVertex2f(centroX + radio * std::cos(angulo),
                   centroY + radio * std::sin(angulo));
    }
    glEnd();
}

// Traza la orbita de referencia. Cada tipo de fisica describe una curva
// distinta, asi que el dibujo tambien cambia segun el tipo.
static void dibujarOrbita(const CuerpoCeleste& cuerpo) {
    const int pasos = SEGMENTOS_CIRCULO * 3;
    glColor4f(1.0f, 1.0f, 1.0f, 0.10f);

    if (cuerpo.tipo == FISICA_KEPLERIANA) {
        // Elipse con el Sol en un FOCO (no en el centro) y rotada.
        const float semiEjeB = cuerpo.semiEjeMayor *
            std::sqrt(1.0f - cuerpo.excentricidad * cuerpo.excentricidad);
        const float cosArg = std::cos(cuerpo.argumentoPeriapsis);
        const float senArg = std::sin(cuerpo.argumentoPeriapsis);
        glBegin(GL_LINE_LOOP);
        for (int i = 0; i < pasos; ++i) {
            const float e = 2.0f * PI * static_cast<float>(i) /
                            static_cast<float>(pasos);
            const float xl = cuerpo.semiEjeMayor *
                             (std::cos(e) - cuerpo.excentricidad);
            const float yl = semiEjeB * std::sin(e);
            glVertex2f(g_escena.centroX + xl * cosArg - yl * senArg,
                       g_escena.centroY + xl * senArg + yl * cosArg);
        }
        glEnd();
        return;
    }

    if (cuerpo.tipo == FISICA_CUANTICA) {
        return; // salta de sitio: no tiene una orbita fija que trazar
    }

    // Circular, binaria (orbita del baricentro) y oscilante: elipse centrada.
    const float a = (cuerpo.tipo == FISICA_BINARIA) ? cuerpo.radioBaricentro
                                                    : cuerpo.semiEjeMayor;
    const float b = (cuerpo.tipo == FISICA_BINARIA) ? cuerpo.radioBaricentro * 0.85f
                                                    : cuerpo.semiEjeMenor;
    glBegin(GL_LINE_LOOP);
    for (int i = 0; i < pasos; ++i) {
        const float angulo = 2.0f * PI * static_cast<float>(i) /
                             static_cast<float>(pasos);
        glVertex2f(g_escena.centroX + a * std::cos(angulo),
                   g_escena.centroY + b * std::sin(angulo));
    }
    glEnd();
}

// Cola de hielo del cometa: crece al acercarse al Sol y apunta en direccion
// contraria a el, como una cola real empujada por el viento solar.
static void dibujarColaCometa(const CuerpoCeleste& cuerpo) {
    const float dx = cuerpo.posX - g_escena.centroX;
    const float dy = cuerpo.posY - g_escena.centroY;
    const float distancia = std::sqrt(dx * dx + dy * dy);
    if (distancia < 1.0f) return;

    const float afelio = cuerpo.semiEjeMayor * (1.0f + cuerpo.excentricidad);
    float cercania = 1.0f - (distancia / afelio);
    if (cercania < 0.0f) cercania = 0.0f;

    const float largo = 25.0f + 130.0f * cercania * cercania;
    const float ux = dx / distancia;
    const float uy = dy / distancia;

    const int pasos = 10;
    for (int k = 1; k <= pasos; ++k) {
        const float f = static_cast<float>(k) / static_cast<float>(pasos);
        dibujarDisco(cuerpo.posX + ux * largo * f,
                     cuerpo.posY + uy * largo * f,
                     cuerpo.radioCuerpo * (1.0f - 0.55f * f),
                     0.75f, 0.90f, 1.0f,
                     0.26f * (1.0f - f) * (0.35f + cercania));
    }
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
static void dibujarTexto(float x, float y, const std::string& texto,
                         float alfa) {
    glColor4f(1.0f, 1.0f, 1.0f, alfa);
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
            if (g_escena.cuerpos[i].tipo != FISICA_SATELITE) {
                dibujarOrbita(g_escena.cuerpos[i]);
            }
        }
    }

    dibujarSol();

    // Los cuerpos celestes, con el adorno propio de su tipo de fisica.
    for (size_t i = 0; i < g_escena.cuerpos.size(); ++i) {
        const CuerpoCeleste& cuerpo = g_escena.cuerpos[i];

        if (cuerpo.tipo == FISICA_KEPLERIANA) {
            dibujarColaCometa(cuerpo);
        }

        dibujarDisco(cuerpo.posX, cuerpo.posY, cuerpo.radioCuerpo * 1.8f,
                     cuerpo.colorR, cuerpo.colorG, cuerpo.colorB, 0.18f);
        dibujarDisco(cuerpo.posX, cuerpo.posY, cuerpo.radioCuerpo,
                     cuerpo.colorR, cuerpo.colorG, cuerpo.colorB, 1.0f);

        if (cuerpo.tipo == FISICA_CUANTICA) {
            dibujarContorno(cuerpo.posX, cuerpo.posY, cuerpo.radioCuerpo * 2.4f,
                            cuerpo.colorR, cuerpo.colorG, cuerpo.colorB, 0.55f);
        }
    }

    // ---------------------------- HUD ----------------------------------
    const float alto = static_cast<float>(g_escena.altoVentana);
    char linea[192];

    std::snprintf(linea, sizeof(linea), "FPS: %6.2f   |   N = %zu   |   %s",
                  g_escena.fpsActual, g_escena.cuerpos.size(),
#ifdef USE_OPENMP
                  "OpenMP"
#else
                  "secuencial"
#endif
                  );
    dibujarTexto(10.0f, alto - 22.0f, std::string(linea), 0.85f);

    // Composicion de la escena por modelo de fisica.
    std::string composicion = "fisica: ";
    for (int t = 0; t < CANTIDAD_TIPOS_FISICA; ++t) {
        if (g_escena.conteoPorTipo[t] == 0) continue;
        char parte[48];
        std::snprintf(parte, sizeof(parte), "%s=%d  ",
                      NOMBRES_FISICA[t], g_escena.conteoPorTipo[t]);
        composicion += parte;
    }
    dibujarTexto(10.0f, alto - 40.0f, composicion, 0.65f);

    // Con pocos cuerpos se muestra que hilo atendio a cada uno.
    if (g_escena.cuerpos.size() <= 12 && g_escena.cuerpos.size() > 0) {
        std::string asignacion = "cuerpo->hilo: ";
        for (size_t i = 0; i < g_escena.cuerpos.size(); ++i) {
            char parte[32];
            std::snprintf(parte, sizeof(parte), "%zu:h%d  ",
                          i, g_escena.cuerpos[i].hiloAsignado);
            asignacion += parte;
        }
        dibujarTexto(10.0f, alto - 58.0f, asignacion, 0.55f);
    }

    glutSwapBuffers();
}

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
        "Cada cuerpo recibe un modelo de fisica distinto (circular, kepler,\n"
        "binaria, satelite, cuantica u oscilante), por lo que el costo de\n"
        "actualizarlo varia de un cuerpo a otro.\n"
        "\n"
        "Ejemplos:\n"
        "  %s              -> Sol + %ld cuerpos, ventana %dx%d\n"
        "  %s 500 1280 720 -> Sol + 500 cuerpos en una ventana de 1280x720\n"
        "  %s 500 1280 720 42 -> misma escena reproducible (semilla 42)\n\n",
        nombrePrograma,
        N_MINIMO, N_MAXIMO, N_POR_DEFECTO,
        ANCHO_MINIMO, ANCHO_POR_DEFECTO,
        ALTO_MINIMO, ALTO_POR_DEFECTO,
        nombrePrograma, N_POR_DEFECTO, ANCHO_POR_DEFECTO, ALTO_POR_DEFECTO,
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

        // Diagnostico del reparto de trabajo: cuantos cuerpos atendio cada
        // hilo en el ultimo cuadro. Sirve como evidencia del balanceo de carga
        // para la bitacora de pruebas. El conteo se hace aqui, fuera de la
        // region paralela, para no interferir con la medicion.
        std::vector<int> cuerposPorHilo(static_cast<size_t>(g_hilosDisponibles), 0);
        for (size_t i = 0; i < g_escena.cuerpos.size(); ++i) {
            const int hilo = g_escena.cuerpos[i].hiloAsignado;
            if (hilo >= 0 && hilo < g_hilosDisponibles) {
                cuerposPorHilo[static_cast<size_t>(hilo)] += 1;
            }
        }
        std::printf("FPS= %.2f  |  reparto por hilo:", g_escena.fpsActual);
        for (int h = 0; h < g_hilosDisponibles; ++h) {
            std::printf(" h%d=%d", h, cuerposPorHilo[static_cast<size_t>(h)]);
        }
        std::printf("\n");
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

#ifdef USE_OPENMP
    g_hilosDisponibles = omp_get_max_threads();
#endif

    // --- Informacion de arranque ---
    std::printf("=== Outer Wilds Screensaver - Prueba de concepto ===\n");
    std::printf("Cuerpos celestes (N): %ld\n", cantidadCuerpos);
    std::printf("Ventana: %ldx%ld\n", anchoSolicitado, altoSolicitado);
    std::printf("Semilla: %u\n", semilla);
    std::printf("Composicion por modelo de fisica:\n");
    for (int t = 0; t < CANTIDAD_TIPOS_FISICA; ++t) {
        if (g_escena.conteoPorTipo[t] > 0) {
            std::printf("  %-10s %d\n", NOMBRES_FISICA[t],
                        g_escena.conteoPorTipo[t]);
        }
    }
    std::printf("Fase 1 (independientes): %zu cuerpos, schedule(dynamic,1)\n",
                g_escena.indicesPrimarios.size());
    std::printf("Fase 2 (satelites):      %zu cuerpos, schedule(static)\n",
                g_escena.indicesSatelites.size());
#ifdef USE_OPENMP
    std::printf("Version: PARALELA (OpenMP, hasta %d hilos)\n",
                g_hilosDisponibles);
#else
    std::printf("Version: SECUENCIAL (%d hilo)\n", g_hilosDisponibles);
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
