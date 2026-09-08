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

// --- Parametros de apariencia ---
static const int    MAX_MANCHAS            = 4;    // detalles por superficie
static const int    CANTIDAD_ESTRELLAS     = 260;  // estrellas de fondo
static const float  DURACION_ESTELA        = 0.8f; // estela del salto cuantico
static const int    RAYOS_CORONA           = 28;   // fulgores del Sol
// Por encima de este N el dibujo se simplifica a disco + halo. El detalle fino
// (manchas, anillos, zarcillos) solo tiene sentido cuando los cuerpos se ven
// suficientemente grandes, y evita que el renderizado secuencial se vuelva el
// cuello de botella en las pruebas de carga.
static const size_t LIMITE_DETALLE_FINO    = 200;

// --- Nave exploradora ---
static const float  NAVE_RAPIDEZ_MAXIMA    = 230.0f; // px/s
static const float  NAVE_AGILIDAD          = 2.0f;   // suavizado del viraje
static const float  NAVE_ESPERA_EN_DESTINO = 2.5f;   // s parada en cada planeta
static const size_t NAVE_LARGO_ESTELA      = 34;     // muestras de la estela

// --- Secuencia final (asteroide -> supernova -> cierre) ---
static const float  FINAL_GM_ASTEROIDE     = 900000.0f; // atraccion al Sol
static const float  FINAL_DURACION_ONDA    = 2.3f;      // s de la explosion
static const float  FINAL_DURACION_APAGADO = 1.1f;      // s de fundido a negro
static const float  FINAL_VELOCIDAD_ONDA   = 950.0f;    // px/s del frente
static const int    FINAL_CANTIDAD_ESCOMBROS = 260;

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
    FISICA_ESTACION,       // Sun Station: orbita rasante al Sol + giro propio
    FISICA_NAVE,           // nave: navegacion vectorial con llegada suave
    CANTIDAD_TIPOS_FISICA
};

static const char* NOMBRES_FISICA[CANTIDAD_TIPOS_FISICA] = {
    "circular", "kepler", "binaria", "satelite", "cuantica", "oscilante",
    "estacion", "nave"
};

// Etapas de la secuencia de cierre que se dispara con ESC o 'q'.
enum EstadoFinal {
    FINAL_INACTIVO = 0,  // funcionamiento normal
    FINAL_ASTEROIDE,     // un asteroide cae hacia el Sol
    FINAL_EXPLOSION,     // impacto: supernova y onda expansiva
    FINAL_APAGADO        // fundido a negro antes de cerrar
};

// ============================================================================
//  ESTRUCTURAS DE DATOS
// ============================================================================

// Detalle sobre la superficie de un cuerpo (continente, mar, crater, mancha
// ciclonica...). Se guarda en coordenadas esfericas para que, al girar el
// cuerpo sobre su eje, el detalle se desplace y desaparezca por el borde como
// lo haria sobre una esfera real.
struct ManchaSuperficie {
    float longitud;  // posicion a lo largo del ecuador (rad)
    float latitud;   // -PI/2 .. PI/2
    float tamano;    // radio relativo al del cuerpo (0..1)
};

// Estrella del fondo. Se guarda en coordenadas normalizadas [0,1] para que
// sobreviva a los cambios de tamano de la ventana.
struct Estrella {
    float x, y;        // posicion normalizada
    float brillo;      // 0..1
    float frecuencia;  // velocidad del parpadeo
    float fase;        // desfase del parpadeo
};

// Fragmento lanzado por la supernova durante la secuencia de cierre.
struct Escombro {
    float x, y;
    float velX, velY;
    float radio;
    float r, g, b;
    float vida;     // segundos restantes
    float vidaTotal;
};

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
    float posEstelaX, posEstelaY; // ultima posicion antes de saltar
    float tiempoEstela;           // segundos que le quedan a la estela

    // --- FISICA_OSCILANTE ---
    int   armonicos;            // K terminos de la serie (costo del cuerpo)
    float amplitudOscilacion;   // amplitud relativa de la perturbacion
    float faseOscilacion;       // desfase base de la serie

    // --- FISICA_ESTACION ---
    float radioAnillo;          // radio del anillo estructural (px)

    // --- FISICA_NAVE ---
    float velX, velY;           // velocidad lineal (px/s)
    float rumbo;                // orientacion, derivada de la velocidad (rad)
    int   indiceObjetivo;       // cuerpo hacia el que viaja
    float tiempoEnDestino;      // segundos que lleva junto a su objetivo
    float empuje;               // 0..1, intensidad de la llama del motor
    unsigned int contadorViajes;// cuantos destinos lleva visitados

    // --- Secuencia final ---
    float velFinalX, velFinalY; // velocidad al ser barrido por la onda
    bool  empujadoPorOnda;      // ya lo alcanzo el frente de la explosion
    float opacidad;             // 1 normal, baja mientras se desintegra

    // --- Comunes ---
    float radioBase;            // radio nominal dibujado (px)
    float radioCuerpo;          // radio efectivo del cuadro actual (px)
    float posX, posY;           // posicion resultante en coordenadas de ventana
    float colorR, colorG, colorB;
    int   hiloAsignado;         // id del hilo que lo actualizo (diagnostico)

    // --- Apariencia ---
    float anguloRotacion;       // rotacion propia sobre su eje (rad)
    float velocidadRotacion;    // rad/s de la rotacion propia
    int   cantidadManchas;
    ManchaSuperficie manchas[MAX_MANCHAS];
    float manchaR, manchaG, manchaB;  // color de los detalles de superficie
    bool  tieneAnillo;
    float inclinacionAnillo;    // achatamiento vertical del anillo (0.15-0.40)
    int   indiceCompanero;      // gemelo binario asociado (-1 si no aplica)
};

// Estado global de la escena. GLUT trabaja con callbacks sin parametros de
// usuario, por lo que el estado debe ser accesible globalmente.
struct EstadoEscena {
    std::vector<CuerpoCeleste> cuerpos;

    // El arreglo se recorre en dos fases porque los satelites dependen de la
    // posicion ya actualizada de su padre. Guardar los indices separados evita
    // meter un 'if' dentro del bucle paralelo.
    std::vector<int> indicesPrimarios;    // no dependen de nadie
    std::vector<int> indicesDependientes; // satelites y nave: leen a otro cuerpo
    std::vector<int> indicesDestinoNave;  // planetas a los que puede viajar
    int indiceNave;                       // -1 si no hay nave en la escena

    // Estela de la nave (posiciones recientes). Vive aqui y no en el cuerpo
    // porque se actualiza durante el dibujado, que es secuencial, y porque un
    // arreglo por cuerpo multiplicaria la memoria con N grande.
    std::vector<float> estelaNaveX;
    std::vector<float> estelaNaveY;
    size_t estelaCursor;
    size_t estelaLlenas;

    int conteoPorTipo[CANTIDAD_TIPOS_FISICA];
    long cuerposSolicitados;  // el N que pidio el usuario (sin estacion ni nave)

    std::vector<Estrella> estrellas;    // fondo estelar (coordenadas 0..1)

    // RECURSO COMPARTIDO entre los gemelos: cuanta arena se ha transferido.
    // Varios hilos lo modifican en el mismo cuadro -> se protege con atomic.
    float arenaTransferida;

    int   anchoVentana;
    int   altoVentana;
    float centroX, centroY;   // posicion del Sol (centro de la ventana)
    float radioSolBase;
    float radioSolActual;
    float tiempoAcumulado;

    // --- Secuencia de cierre ---
    EstadoFinal estadoFinal;
    float tiempoEnEtapa;       // segundos dentro de la etapa actual
    float asteroideX, asteroideY;
    float asteroideVelX, asteroideVelY;
    float asteroideRadio;
    float asteroideGiro;
    float radioOnda;           // radio del frente de la explosion
    float destello;            // 0..1, fogonazo blanco del impacto
    std::vector<Escombro> escombros;

    // Medicion de FPS
    int   framesEnIntervalo;
    float tiempoEnIntervalo;
    float fpsActual;

    EstadoEscena()
        : indiceNave(-1), estelaCursor(0), estelaLlenas(0),
          cuerposSolicitados(0), arenaTransferida(0.0f),
          anchoVentana(ANCHO_POR_DEFECTO), altoVentana(ALTO_POR_DEFECTO),
          centroX(0.0f), centroY(0.0f), radioSolBase(40.0f),
          radioSolActual(40.0f), tiempoAcumulado(0.0f),
          estadoFinal(FINAL_INACTIVO), tiempoEnEtapa(0.0f),
          asteroideX(0.0f), asteroideY(0.0f),
          asteroideVelX(0.0f), asteroideVelY(0.0f),
          asteroideRadio(14.0f), asteroideGiro(0.0f),
          radioOnda(0.0f), destello(0.0f),
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
    if (c.tiempoEstela > 0.0f) {
        c.tiempoEstela -= dt;
        if (c.tiempoEstela < 0.0f) c.tiempoEstela = 0.0f;
    }

    c.tiempoParaSalto -= dt;
    if (c.tiempoParaSalto <= 0.0f) {
        // Deja una estela que se desvanece en el punto donde estaba: da la
        // lectura visual de "desaparecio de aqui y aparecio alla".
        c.posEstelaX  = c.posX;
        c.posEstelaY  = c.posY;
        c.tiempoEstela = DURACION_ESTELA;
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
// 6) FISICA_ESTACION - Sun Station.
//
//    Orbita rasante y muy rapida pegada al Sol, practicamente circular. Como
//    esta tan cerca, se le anade una precesion: el radio late ligeramente por
//    el tiron de marea del Sol. Ademas la estructura gira sobre si misma a un
//    ritmo distinto del orbital, lo que se aprecia en el anillo.
// ---------------------------------------------------------------------------
static void fisicaEstacion(CuerpoCeleste& c, float dt, float tiempo,
                           float cx, float cy, float radioSol) {
    c.anguloOrbital = normalizarAngulo(c.anguloOrbital +
                                       c.velocidadAngular * dt);

    // El radio se referencia al Sol actual para que la estacion siga pegada a
    // el aunque el Sol este pulsando (o creciendo en la secuencia final).
    const float radioBaseOrbita = radioSol * 1.75f;
    const float marea = 1.0f + 0.05f * std::sin(tiempo * 2.3f);
    const float radio = radioBaseOrbita * marea;

    c.posX = cx + radio * std::cos(c.anguloOrbital);
    c.posY = cy + radio * 0.92f * std::sin(c.anguloOrbital);
    c.radioCuerpo = c.radioBase;
}

// ---------------------------------------------------------------------------
// 7) FISICA_NAVE - la nave exploradora.
//
//    Es el unico cuerpo que NO sigue una orbita: navega. Usa un modelo de
//    direccion vectorial con "llegada suave" (arrival steering):
//
//        rapidezDeseada = min(rapidezMaxima, distancia * factorFrenado)
//        velDeseada     = direccionAlObjetivo * rapidezDeseada
//        vel           += (velDeseada - vel) * min(1, agilidad * dt)
//        pos           += vel * dt
//
//    El termino de frenado hace que desacelere sola al acercarse en vez de
//    pasarse de largo. Cuando lleva un rato junto al planeta elige otro destino
//    con la misma funcion de hash determinista que la luna cuantica, asi que
//    tampoco necesita un generador con estado compartido.
//
//    Depende de la posicion ya actualizada de su objetivo, asi que se resuelve
//    en la segunda fase, despues de la barrera.
// ---------------------------------------------------------------------------
static void fisicaNave(CuerpoCeleste& c, const CuerpoCeleste* cuerpos,
                       const int* destinos, int cantidadDestinos, float dt) {
    if (cantidadDestinos <= 0) {
        return;  // no hay a donde ir: la nave se queda a la deriva
    }

    // Elige destino si no tiene uno valido todavia.
    if (c.indiceObjetivo < 0) {
        const unsigned int semilla =
            static_cast<unsigned int>(c.contadorViajes) * 2246822519u + 7919u;
        c.indiceObjetivo = destinos[mezclarEntero(semilla) %
                                    static_cast<unsigned int>(cantidadDestinos)];
    }

    const CuerpoCeleste& objetivo = cuerpos[c.indiceObjetivo];
    const float dx = objetivo.posX - c.posX;
    const float dy = objetivo.posY - c.posY;
    const float distancia = std::sqrt(dx * dx + dy * dy);

    // Radio a partir del cual se considera que "llego" al planeta.
    const float radioLlegada = objetivo.radioCuerpo * 2.2f + 22.0f;

    if (distancia < radioLlegada) {
        c.tiempoEnDestino += dt;
        if (c.tiempoEnDestino >= NAVE_ESPERA_EN_DESTINO) {
            c.tiempoEnDestino = 0.0f;
            c.contadorViajes += 1u;
            const unsigned int semilla =
                static_cast<unsigned int>(c.contadorViajes) * 2246822519u + 7919u;
            c.indiceObjetivo = destinos[mezclarEntero(semilla) %
                                        static_cast<unsigned int>(cantidadDestinos)];
        }
    } else {
        c.tiempoEnDestino = 0.0f;
    }

    // Velocidad deseada con frenado proporcional a la distancia restante.
    float rapidezDeseada = distancia * 1.3f;
    if (rapidezDeseada > NAVE_RAPIDEZ_MAXIMA) {
        rapidezDeseada = NAVE_RAPIDEZ_MAXIMA;
    }
    float deseadaX = 0.0f;
    float deseadaY = 0.0f;
    if (distancia > 1.0e-3f) {
        deseadaX = (dx / distancia) * rapidezDeseada;
        deseadaY = (dy / distancia) * rapidezDeseada;
    }

    // Ademas de perseguir, arrastra la velocidad orbital del planeta destino
    // para poder acompanarlo en vez de quedarse atras.
    float mezcla = NAVE_AGILIDAD * dt;
    if (mezcla > 1.0f) mezcla = 1.0f;
    const float correccionX = (deseadaX - c.velX) * mezcla;
    const float correccionY = (deseadaY - c.velY) * mezcla;
    c.velX += correccionX;
    c.velY += correccionY;

    c.posX += c.velX * dt;
    c.posY += c.velY * dt;

    // El motor se enciende en proporcion a cuanto tuvo que corregir el rumbo.
    const float magnitudCorreccion =
        std::sqrt(correccionX * correccionX + correccionY * correccionY);
    c.empuje = magnitudCorreccion / (NAVE_RAPIDEZ_MAXIMA * 0.35f);
    if (c.empuje > 1.0f) c.empuje = 1.0f;

    // La nave siempre apunta hacia donde se mueve.
    const float rapidez = std::sqrt(c.velX * c.velX + c.velY * c.velY);
    if (rapidez > 4.0f) {
        c.rumbo = std::atan2(c.velY, c.velX);
    }
    c.radioCuerpo = c.radioBase;
}

// ---------------------------------------------------------------------------
// 8) FISICA_SATELITE - The Attlerock.
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
//  SECUENCIA DE CIERRE: asteroide -> supernova -> apagado
// ============================================================================

// Lanza el asteroide desde fuera de la ventana, con una velocidad inicial
// dirigida al Sol. A partir de ahi lo gobierna la gravedad, asi que llega
// acelerando (atraccion inversa al cuadrado de la distancia).
static void iniciarSecuenciaFinal() {
    if (g_escena.estadoFinal != FINAL_INACTIVO) {
        return;  // ya esta en marcha; una segunda pulsacion fuerza la salida
    }

    g_escena.estadoFinal   = FINAL_ASTEROIDE;
    g_escena.tiempoEnEtapa = 0.0f;
    g_escena.radioOnda     = 0.0f;
    g_escena.destello      = 0.0f;

    // Entra por una esquina elegida con el reloj, para que no sea siempre igual.
    const unsigned int semilla = static_cast<unsigned int>(
        g_escena.tiempoAcumulado * 1000.0f);
    const float anguloEntrada = aleatorioDeterminista(semilla) * 2.0f * PI;
    // Justo fuera de la ventana: la diagonal completa por 0.6 deja margen
    // suficiente para que entre en cuadro ya acelerando.
    const float diagonal = std::sqrt(
        static_cast<float>(g_escena.anchoVentana) *
        static_cast<float>(g_escena.anchoVentana) +
        static_cast<float>(g_escena.altoVentana) *
        static_cast<float>(g_escena.altoVentana));
    const float distanciaEntrada = 0.60f * diagonal;

    g_escena.asteroideX = g_escena.centroX +
                          distanciaEntrada * std::cos(anguloEntrada);
    g_escena.asteroideY = g_escena.centroY +
                          distanciaEntrada * std::sin(anguloEntrada);
    g_escena.asteroideRadio = 15.0f;
    g_escena.asteroideGiro  = 0.0f;

    // Velocidad inicial hacia el Sol, con un ligero sesgo lateral para que la
    // trayectoria se vea como una caida y no como una linea recta.
    const float haciaSolX = g_escena.centroX - g_escena.asteroideX;
    const float haciaSolY = g_escena.centroY - g_escena.asteroideY;
    const float d = std::sqrt(haciaSolX * haciaSolX + haciaSolY * haciaSolY);
    const float rapidezInicial = 300.0f;
    const float sesgoLateral   = 140.0f;
    g_escena.asteroideVelX = (haciaSolX / d) * rapidezInicial -
                             (haciaSolY / d) * sesgoLateral;
    g_escena.asteroideVelY = (haciaSolY / d) * rapidezInicial +
                             (haciaSolX / d) * sesgoLateral;

    std::printf("\n>> Asteroide en curso de colision con el Sol...\n");
    std::fflush(stdout);
}

// Genera los fragmentos que salen despedidos del Sol en el impacto.
static void dispararEscombros() {
    g_escena.escombros.clear();
    g_escena.escombros.reserve(FINAL_CANTIDAD_ESCOMBROS);

    for (int i = 0; i < FINAL_CANTIDAD_ESCOMBROS; ++i) {
        const unsigned int base = static_cast<unsigned int>(i) * 2654435761u;
        Escombro fragmento;

        const float angulo = aleatorioDeterminista(base) * 2.0f * PI;
        const float rapidez = 180.0f + 780.0f * aleatorioDeterminista(base + 1u);
        fragmento.x = g_escena.centroX;
        fragmento.y = g_escena.centroY;
        fragmento.velX = std::cos(angulo) * rapidez;
        fragmento.velY = std::sin(angulo) * rapidez;
        fragmento.radio = 1.5f + 4.5f * aleatorioDeterminista(base + 2u);

        // Paleta del Sol: del blanco incandescente al naranja profundo.
        const float calor = aleatorioDeterminista(base + 3u);
        fragmento.r = 1.0f;
        fragmento.g = 0.45f + 0.50f * calor;
        fragmento.b = 0.10f + 0.55f * calor * calor;

        fragmento.vidaTotal = 1.2f + 1.6f * aleatorioDeterminista(base + 4u);
        fragmento.vida = fragmento.vidaTotal;
        g_escena.escombros.push_back(fragmento);
    }
}

// Avanza la parte secuencial de la secuencia final: asteroide, onda, escombros
// y temporizadores de etapa. El empuje de los cuerpos se hace aparte, en
// paralelo, dentro de actualizarEscena().
static void actualizarSecuenciaFinal(float dt) {
    g_escena.tiempoEnEtapa += dt;

    if (g_escena.estadoFinal == FINAL_ASTEROIDE) {
        // Atraccion gravitatoria del Sol: a = GM / r^2, dirigida al centro.
        const float dx = g_escena.centroX - g_escena.asteroideX;
        const float dy = g_escena.centroY - g_escena.asteroideY;
        float distanciaCuadrado = dx * dx + dy * dy;
        if (distanciaCuadrado < 1.0f) distanciaCuadrado = 1.0f;
        const float distancia = std::sqrt(distanciaCuadrado);

        const float radialX = dx / distancia;
        const float radialY = dy / distancia;

        const float aceleracion = FINAL_GM_ASTEROIDE / distanciaCuadrado;
        g_escena.asteroideVelX += radialX * aceleracion * dt;
        g_escena.asteroideVelY += radialY * aceleracion * dt;

        // Amortiguamiento de la componente tangencial. Con gravedad pura y algo
        // de velocidad lateral el asteroide describiria una hiperbola y pasaria
        // de largo; aqui interesa que SIEMPRE impacte, asi que se le resta
        // momento angular de forma progresiva. Es una espiral de caida, no una
        // trayectoria balistica: esto es una animacion de cierre, no una
        // simulacion, y se documenta como tal.
        const float componenteRadial = g_escena.asteroideVelX * radialX +
                                       g_escena.asteroideVelY * radialY;
        float tangencialX = g_escena.asteroideVelX - componenteRadial * radialX;
        float tangencialY = g_escena.asteroideVelY - componenteRadial * radialY;
        float amortiguado = 1.0f - 1.7f * dt;
        if (amortiguado < 0.0f) amortiguado = 0.0f;
        tangencialX *= amortiguado;
        tangencialY *= amortiguado;
        g_escena.asteroideVelX = componenteRadial * radialX + tangencialX;
        g_escena.asteroideVelY = componenteRadial * radialY + tangencialY;

        g_escena.asteroideX += g_escena.asteroideVelX * dt;
        g_escena.asteroideY += g_escena.asteroideVelY * dt;
        g_escena.asteroideGiro = normalizarAngulo(g_escena.asteroideGiro + 3.2f * dt);

        // El asteroide se calienta y se hincha al acercarse.
        const float proximidad = 1.0f -
            distancia / (0.75f * static_cast<float>(g_escena.anchoVentana));
        g_escena.asteroideRadio = 15.0f +
            10.0f * (proximidad > 0.0f ? proximidad : 0.0f);

        // Impacto. La condicion de tiempo es una red de seguridad: pase lo que
        // pase con la trayectoria, el screensaver nunca se queda colgado sin
        // poder cerrarse.
        if (distancia <= g_escena.radioSolActual + g_escena.asteroideRadio ||
            g_escena.tiempoEnEtapa > 6.0f) {
            g_escena.estadoFinal   = FINAL_EXPLOSION;
            g_escena.tiempoEnEtapa = 0.0f;
            g_escena.radioOnda     = g_escena.radioSolActual;
            g_escena.destello      = 0.80f;
            dispararEscombros();
            std::printf(">> IMPACTO. El Sol entra en supernova.\n");
            std::fflush(stdout);
        }
        return;
    }

    if (g_escena.estadoFinal == FINAL_EXPLOSION) {
        // Frente de la onda expansiva y fogonazo que se apaga.
        g_escena.radioOnda += FINAL_VELOCIDAD_ONDA * dt;
        g_escena.destello -= dt * 4.5f;   // fogonazo breve, no un velo largo
        if (g_escena.destello < 0.0f) g_escena.destello = 0.0f;

        // El Sol se hincha de golpe y luego se desvanece.
        // El nucleo se hincha de golpe y despues COLAPSA: la estrella queda
        // destruida. Lo que sigue viendose es el cascaron en expansion, no un
        // disco gigante ocupando el centro de la pantalla.
        float avance = g_escena.tiempoEnEtapa / FINAL_DURACION_ONDA;
        if (avance > 1.0f) avance = 1.0f;

        float hinchazon;
        if (avance < 0.25f) {
            hinchazon = 1.0f + 8.8f * avance;              // 1x -> 3.2x
        } else {
            float restante = 1.0f - (avance - 0.25f) / 0.45f;
            if (restante < 0.0f) restante = 0.0f;
            hinchazon = 3.2f * restante;                    // 3.2x -> 0
        }
        g_escena.radioSolActual = g_escena.radioSolBase * hinchazon;

        // Escombros: movimiento inercial con un leve frenado.
        for (size_t i = 0; i < g_escena.escombros.size(); ++i) {
            Escombro& fragmento = g_escena.escombros[i];
            if (fragmento.vida <= 0.0f) continue;
            fragmento.x += fragmento.velX * dt;
            fragmento.y += fragmento.velY * dt;
            fragmento.velX *= (1.0f - 0.55f * dt);
            fragmento.velY *= (1.0f - 0.55f * dt);
            fragmento.vida -= dt;
        }

        if (g_escena.tiempoEnEtapa >= FINAL_DURACION_ONDA) {
            g_escena.estadoFinal   = FINAL_APAGADO;
            g_escena.tiempoEnEtapa = 0.0f;
        }
        return;
    }

    if (g_escena.estadoFinal == FINAL_APAGADO) {
        for (size_t i = 0; i < g_escena.escombros.size(); ++i) {
            Escombro& fragmento = g_escena.escombros[i];
            if (fragmento.vida <= 0.0f) continue;
            fragmento.x += fragmento.velX * dt;
            fragmento.y += fragmento.velY * dt;
            fragmento.vida -= dt;
        }
        if (g_escena.tiempoEnEtapa >= FINAL_DURACION_APAGADO) {
            std::printf(">> Fin del ciclo. Cerrando screensaver.\n");
            std::fflush(stdout);
            std::exit(EXIT_SUCCESS);
        }
    }
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
    g_escena.tiempoAcumulado += dt;

    CuerpoCeleste* cuerpos      = g_escena.cuerpos.data();
    const int      numCuerpos   = static_cast<int>(g_escena.cuerpos.size());
    const int*     primarios    = g_escena.indicesPrimarios.data();
    const int      numPrimarios = static_cast<int>(g_escena.indicesPrimarios.size());
    const int*     dependientes = g_escena.indicesDependientes.data();
    const int      numDependientes =
        static_cast<int>(g_escena.indicesDependientes.size());
    const int*     destinos     = g_escena.indicesDestinoNave.data();
    const int      numDestinos  = static_cast<int>(g_escena.indicesDestinoNave.size());

    // ------------------------------------------------------------------
    // Secuencia de cierre. Durante la caida del asteroide la escena sigue
    // animandose con normalidad; a partir del impacto los cuerpos dejan de
    // orbitar y salen despedidos por la onda expansiva.
    // ------------------------------------------------------------------

    // Pulso normal del Sol. Durante la explosion, actualizarSecuenciaFinal()
    // sobrescribe este valor con la hinchazon de la supernova.
    g_escena.radioSolActual = g_escena.radioSolBase *
        (1.0f + 0.06f * std::sin(g_escena.tiempoAcumulado * 1.2f));

    if (g_escena.estadoFinal != FINAL_INACTIVO) {
        actualizarSecuenciaFinal(dt);

        if (g_escena.estadoFinal == FINAL_EXPLOSION ||
            g_escena.estadoFinal == FINAL_APAGADO) {
            const float cxOnda = g_escena.centroX;
            const float cyOnda = g_escena.centroY;
            const float frente = g_escena.radioOnda;

            // El barrido de la onda tambien se reparte entre hilos: cada cuerpo
            // solo lee el radio del frente y escribe su propio estado.
#ifdef USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
            for (int i = 0; i < numCuerpos; ++i) {
                CuerpoCeleste& cuerpo = cuerpos[i];
                const float dx = cuerpo.posX - cxOnda;
                const float dy = cuerpo.posY - cyOnda;
                const float distancia = std::sqrt(dx * dx + dy * dy);

                if (!cuerpo.empujadoPorOnda && frente >= distancia) {
                    // La onda lo alcanza: sale despedido radialmente, mas
                    // rapido cuanto mas cerca estaba del Sol.
                    const float atenuacion = 260.0f / (60.0f + distancia);
                    const float impulso = 240.0f + 900.0f * atenuacion;
                    const float nx = (distancia > 1.0e-3f) ? dx / distancia : 1.0f;
                    const float ny = (distancia > 1.0e-3f) ? dy / distancia : 0.0f;
                    cuerpo.velFinalX = nx * impulso - ny * impulso * 0.25f;
                    cuerpo.velFinalY = ny * impulso + nx * impulso * 0.25f;
                    cuerpo.empujadoPorOnda = true;
                }

                if (cuerpo.empujadoPorOnda) {
                    cuerpo.posX += cuerpo.velFinalX * dt;
                    cuerpo.posY += cuerpo.velFinalY * dt;
                    cuerpo.anguloRotacion = normalizarAngulo(
                        cuerpo.anguloRotacion + 6.0f * dt);
                    cuerpo.opacidad -= dt * 0.85f;
                    if (cuerpo.opacidad < 0.0f) cuerpo.opacidad = 0.0f;
                }
            }
            return;  // no se ejecuta la fisica orbital normal
        }
    }

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

            // Rotacion propia sobre su eje: comun a todos los modelos y lo que
            // hace que los detalles de superficie se desplacen por el disco.
            cuerpo.anguloRotacion = normalizarAngulo(
                cuerpo.anguloRotacion + cuerpo.velocidadRotacion * dt);

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
                case FISICA_ESTACION:
                    fisicaEstacion(cuerpo, dt, tiempo, cx, cy,
                                   g_escena.radioSolActual);
                    break;
                case FISICA_CIRCULAR:
                default:
                    fisicaCircular(cuerpo, dt, cx, cy);
                    break;
            }
        }
        // Barrera implicita al cerrar el 'omp for': a partir de aqui todos los
        // cuerpos primarios tienen su posicion definitiva del cuadro.

        // ------------- Fase 2: cuerpos dependientes (leen a otro) -----------
        // Satelites (leen a su padre) y la nave (lee a su planeta destino).
        // Sus objetivos son siempre cuerpos de la fase 1, ya actualizados.
#ifdef USE_OPENMP
#pragma omp for schedule(static)
#endif
        for (int k = 0; k < numDependientes; ++k) {
            const int indice = dependientes[k];
            CuerpoCeleste& cuerpo = cuerpos[indice];

#ifdef USE_OPENMP
            cuerpo.hiloAsignado = omp_get_thread_num();
#else
            cuerpo.hiloAsignado = 0;
#endif
            cuerpo.anguloRotacion = normalizarAngulo(
                cuerpo.anguloRotacion + cuerpo.velocidadRotacion * dt);

            if (cuerpo.tipo == FISICA_NAVE) {
                fisicaNave(cuerpo, cuerpos, destinos, numDestinos, dt);
            } else {
                fisicaSatelite(cuerpo, cuerpos[cuerpo.indicePadre], dt);
            }
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

// Genera el fondo estelar. Las posiciones se guardan normalizadas para que la
// escena siga siendo valida si el usuario redimensiona la ventana.
static void inicializarEstrellas(unsigned int semilla) {
    std::mt19937 generador(semilla);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    g_escena.estrellas.clear();
    g_escena.estrellas.reserve(CANTIDAD_ESTRELLAS);
    for (int i = 0; i < CANTIDAD_ESTRELLAS; ++i) {
        Estrella estrella;
        estrella.x = dist(generador);
        estrella.y = dist(generador);
        // Distribucion sesgada hacia estrellas tenues: unas pocas destacan.
        const float u = dist(generador);
        estrella.brillo     = 0.18f + 0.72f * u * u;
        estrella.frecuencia = 0.6f + 2.4f * dist(generador);
        estrella.fase       = dist(generador) * 2.0f * PI;
        g_escena.estrellas.push_back(estrella);
    }
}

// Reparte unas cuantas manchas sobre la superficie de un cuerpo. Se colocan en
// coordenadas esfericas (longitud, latitud) para que giren con el cuerpo.
static void generarManchas(CuerpoCeleste& cuerpo, int cantidad,
                           float tamanoMin, float tamanoMax,
                           std::mt19937& generador) {
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    if (cantidad > MAX_MANCHAS) cantidad = MAX_MANCHAS;
    cuerpo.cantidadManchas = cantidad;
    for (int m = 0; m < cantidad; ++m) {
        cuerpo.manchas[m].longitud = dist(generador) * 2.0f * PI;
        // Se evitan los polos: ahi la mancha se veria deformada.
        cuerpo.manchas[m].latitud  = (dist(generador) - 0.5f) * 1.4f;
        cuerpo.manchas[m].tamano   = tamanoMin +
            (tamanoMax - tamanoMin) * dist(generador);
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

    // Con pocos cuerpos conviene dibujarlos mas grandes: se aprecian los
    // detalles de superficie, los anillos y los zarcillos. Con N grande se
    // reduce el tamano para que la escena no se convierta en una mancha.
    const float escalaCuerpo = (cantidad <= 12) ? 1.75f
                             : (cantidad <= 40) ? 1.30f
                             : 1.00f;

    g_escena.cuerpos.clear();
    g_escena.cuerpos.reserve(static_cast<size_t>(cantidad));
    g_escena.indicesPrimarios.clear();
    g_escena.indicesDependientes.clear();
    g_escena.indicesDestinoNave.clear();
    g_escena.indiceNave = -1;
    g_escena.arenaTransferida = 0.0f;
    g_escena.estelaNaveX.assign(NAVE_LARGO_ESTELA, 0.0f);
    g_escena.estelaNaveY.assign(NAVE_LARGO_ESTELA, 0.0f);
    g_escena.estelaCursor = 0;
    g_escena.estelaLlenas = 0;
    g_escena.cuerposSolicitados = cantidad;
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

        cuerpo.radioBase   = (6.0f + 12.0f * distUnitaria(generador)) * escalaCuerpo;
        cuerpo.radioCuerpo = cuerpo.radioBase;
        cuerpo.colorR = 0.35f + 0.65f * distUnitaria(generador);
        cuerpo.colorG = 0.35f + 0.65f * distUnitaria(generador);
        cuerpo.colorB = 0.35f + 0.65f * distUnitaria(generador);
        cuerpo.indicePadre = -1;
        cuerpo.indiceCompanero = -1;
        cuerpo.hiloAsignado = 0;

        // --- Apariencia comun ---
        // Rotacion propia: independiente de la orbital, con sentido y ritmo
        // propios. Es la que arrastra los detalles de superficie.
        cuerpo.velocidadRotacion = (distUnitaria(generador) < 0.2f ? -1.0f : 1.0f) *
                                   (0.10f + 0.35f * distUnitaria(generador));
        cuerpo.anguloRotacion = distUnitaria(generador) * 2.0f * PI;
        // Los detalles se pintan en una version mas oscura y desaturada del
        // color del cuerpo, para que se lean como relieve y no como manchas.
        cuerpo.manchaR = cuerpo.colorR * 0.44f + 0.04f;
        cuerpo.manchaG = cuerpo.colorG * 0.44f + 0.04f;
        cuerpo.manchaB = cuerpo.colorB * 0.44f + 0.08f;
        cuerpo.tieneAnillo = false;
        cuerpo.inclinacionAnillo = 0.0f;
        cuerpo.cantidadManchas = 0;
        cuerpo.opacidad = 1.0f;
        cuerpo.empujadoPorOnda = false;
        cuerpo.indiceObjetivo = -1;

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
                // Cometa de hielo: nucleo palido y azulado, y gira deprisa.
                cuerpo.colorR = 0.72f;
                cuerpo.colorG = 0.86f;
                cuerpo.colorB = 0.98f;
                cuerpo.manchaR = 0.50f;
                cuerpo.manchaG = 0.66f;
                cuerpo.manchaB = 0.85f;
                cuerpo.velocidadRotacion *= 2.5f;
                generarManchas(cuerpo, 2, 0.20f, 0.32f, generador);
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
                    // Ash Twin: se va cubriendo de arena, tono claro y calido.
                    cuerpo.colorR = 0.88f; cuerpo.colorG = 0.72f; cuerpo.colorB = 0.42f;
                    cuerpo.manchaR = 0.68f; cuerpo.manchaG = 0.50f; cuerpo.manchaB = 0.28f;
                    cuerpo.indiceCompanero = static_cast<int>(i) - 1;
                    // Se cierra el enlace en los dos sentidos.
                    g_escena.cuerpos[static_cast<size_t>(i - 1)].indiceCompanero =
                        static_cast<int>(i);
                    generarManchas(cuerpo, 3, 0.16f, 0.26f, generador);
                } else {
                    cuerpo.radioBaricentro     = radioOrbital;
                    cuerpo.anguloBaricentro    = cuerpo.anguloOrbital;
                    cuerpo.velocidadBaricentro = cuerpo.velocidadAngular;
                    cuerpo.radioLocal          = (16.0f + 10.0f *
                                                  distUnitaria(generador)) *
                                                 escalaCuerpo;
                    cuerpo.velocidadLocal      = 5.0f * magnitudVelocidad;
                    cuerpo.anguloLocal         = distUnitaria(generador) * 2.0f * PI;
                    cuerpo.desfaseLocal        = 0.0f;
                    cuerpo.esDonanteDeArena    = true;  // Ember: cede
                    // Ember Twin: roca volcanica oscura y rojiza.
                    cuerpo.colorR = 0.76f; cuerpo.colorG = 0.36f; cuerpo.colorB = 0.26f;
                    cuerpo.manchaR = 0.45f; cuerpo.manchaG = 0.16f; cuerpo.manchaB = 0.14f;
                    generarManchas(cuerpo, 3, 0.18f, 0.30f, generador);
                }
                break;
            }
            case FISICA_SATELITE: {
                cuerpo.indicePadre = static_cast<int>(i) - 1;
                if (cuerpo.indicePadre < 0) cuerpo.indicePadre = 0;
                cuerpo.radioBase   = (4.0f + 3.0f * distUnitaria(generador)) *
                                     escalaCuerpo;
                cuerpo.radioCuerpo = cuerpo.radioBase;
                cuerpo.radioLocal  = (26.0f + 14.0f * distUnitaria(generador)) *
                                     escalaCuerpo;
                // La luna gira bastante mas rapido que su planeta.
                cuerpo.velocidadAngular = 8.0f * magnitudVelocidad;
                // Luna rocosa gris con crateres bien marcados.
                cuerpo.colorR = 0.66f; cuerpo.colorG = 0.64f; cuerpo.colorB = 0.62f;
                cuerpo.manchaR = 0.42f; cuerpo.manchaG = 0.40f; cuerpo.manchaB = 0.40f;
                generarManchas(cuerpo, 3, 0.20f, 0.34f, generador);
                break;
            }
            case FISICA_CUANTICA: {
                cuerpo.intervaloSalto  = 3.0f + 4.0f * distUnitaria(generador);
                cuerpo.tiempoParaSalto = cuerpo.intervaloSalto *
                                         distUnitaria(generador);
                cuerpo.contadorSaltos  = 0u;
                cuerpo.tiempoEstela    = 0.0f;
                // Luna cuantica: turquesa palido, casi etereo.
                cuerpo.colorR = 0.62f; cuerpo.colorG = 0.88f; cuerpo.colorB = 0.86f;
                cuerpo.manchaR = 0.38f; cuerpo.manchaG = 0.62f; cuerpo.manchaB = 0.64f;
                generarManchas(cuerpo, 2, 0.22f, 0.30f, generador);
                break;
            }
            case FISICA_OSCILANTE: {
                // K distinto por cuerpo: de aqui sale el desbalance de carga.
                cuerpo.armonicos = 8 + static_cast<int>(
                    24.0f * distUnitaria(generador));
                cuerpo.amplitudOscilacion = 0.05f + 0.07f * distUnitaria(generador);
                cuerpo.faseOscilacion     = distUnitaria(generador) * 2.0f * PI;
                // Dark Bramble: verde oscuro y brumoso, con niebla interior.
                cuerpo.colorR = 0.30f; cuerpo.colorG = 0.48f; cuerpo.colorB = 0.44f;
                cuerpo.manchaR = 0.16f; cuerpo.manchaG = 0.30f; cuerpo.manchaB = 0.30f;
                generarManchas(cuerpo, 3, 0.22f, 0.36f, generador);
                break;
            }
            case FISICA_CIRCULAR:
            default:
                // Planetas "normales": continentes y, en algunos, un anillo.
                generarManchas(cuerpo, 2 + static_cast<int>(
                                   2.0f * distUnitaria(generador)),
                               0.18f, 0.34f, generador);
                if (distUnitaria(generador) < 0.50f) {
                    cuerpo.tieneAnillo = true;
                    cuerpo.inclinacionAnillo = 0.16f + 0.22f * distUnitaria(generador);
                }
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
            g_escena.indicesDependientes.push_back(static_cast<int>(i));
        } else {
            g_escena.indicesPrimarios.push_back(static_cast<int>(i));
            // Los cuerpos de la fase 1 son destinos validos para la nave: al
            // estar ya actualizados cuando la nave los lee, no hay carrera.
            g_escena.indicesDestinoNave.push_back(static_cast<int>(i));
        }
    }

    // ------------------------------------------------------------------
    // Elementos fijos de la escena, siempre presentes y ajenos al parametro N:
    // la Sun Station y la nave exploradora. Se anaden al mismo arreglo para que
    // participen del reparto de trabajo entre hilos como cualquier otro cuerpo.
    // ------------------------------------------------------------------

    // --- Sun Station: anillo en orbita rasante al Sol ---
    {
        CuerpoCeleste estacion;
        std::memset(&estacion, 0, sizeof(estacion));
        estacion.tipo = FISICA_ESTACION;
        estacion.anguloOrbital    = distUnitaria(generador) * 2.0f * PI;
        estacion.velocidadAngular = 0.85f;   // vuelta completa en ~7 s
        estacion.radioBase        = 9.0f * (escalaCuerpo > 1.0f ? 1.35f : 1.0f);
        estacion.radioCuerpo      = estacion.radioBase;
        estacion.radioAnillo      = estacion.radioBase * 2.1f;
        estacion.velocidadRotacion = 1.4f;   // el anillo gira sobre si mismo
        estacion.anguloRotacion   = 0.0f;
        estacion.colorR = 0.82f; estacion.colorG = 0.80f; estacion.colorB = 0.72f;
        estacion.manchaR = 0.45f; estacion.manchaG = 0.43f; estacion.manchaB = 0.40f;
        estacion.indicePadre = -1;
        estacion.indiceCompanero = -1;
        estacion.indiceObjetivo = -1;
        estacion.opacidad = 1.0f;
        estacion.empujadoPorOnda = false;

        g_escena.conteoPorTipo[FISICA_ESTACION] += 1;
        g_escena.indicesPrimarios.push_back(
            static_cast<int>(g_escena.cuerpos.size()));
        g_escena.cuerpos.push_back(estacion);
        // Nota: la estacion NO se anade a indicesDestinoNave; enviar la nave a
        // una orbita rasante al Sol seria un viaje de ida.
    }

    // --- Nave exploradora ---
    {
        CuerpoCeleste nave;
        std::memset(&nave, 0, sizeof(nave));
        nave.tipo = FISICA_NAVE;
        nave.radioBase   = 7.0f * (escalaCuerpo > 1.0f ? 1.4f : 1.0f);
        nave.radioCuerpo = nave.radioBase;
        nave.posX = g_escena.centroX + radioOrbitaMaxima * 0.6f;
        nave.posY = g_escena.centroY;
        nave.velX = 0.0f;
        nave.velY = 60.0f;
        nave.rumbo = PI * 0.5f;
        nave.indiceObjetivo  = -1;   // elige destino en el primer cuadro
        nave.tiempoEnDestino = 0.0f;
        nave.contadorViajes  = 0u;
        nave.empuje = 0.0f;
        nave.colorR = 0.90f; nave.colorG = 0.86f; nave.colorB = 0.76f;
        nave.manchaR = 0.42f; nave.manchaG = 0.32f; nave.manchaB = 0.24f;
        nave.indicePadre = -1;
        nave.indiceCompanero = -1;
        nave.velocidadRotacion = 0.0f;  // no gira: apunta hacia donde vuela
        nave.opacidad = 1.0f;
        nave.empujadoPorOnda = false;

        g_escena.conteoPorTipo[FISICA_NAVE] += 1;
        g_escena.indiceNave = static_cast<int>(g_escena.cuerpos.size());
        g_escena.indicesDependientes.push_back(g_escena.indiceNave);
        g_escena.cuerpos.push_back(nave);
    }

    // El fondo estelar usa una semilla derivada para que no quede acoplado al
    // sorteo de los cuerpos (cambiar N no reordena las estrellas).
    inicializarEstrellas(semilla ^ 0x9E3779B9u);
}

// ============================================================================
//  FASE DE RENDERIZADO  (siempre secuencial, en el hilo principal)
// ============================================================================

// Dibuja un disco relleno mediante un abanico de triangulos.
static void dibujarDisco(float centroX, float centroY, float radio,
                         float r, float g, float b, float alfa) {
    // Resolucion adaptativa: un disco pequeno no necesita muchos lados, pero el
    // Sol hinchado de la supernova ocupa media pantalla y con pocos segmentos se
    // le notaria la forma de poligono.
    int segmentos = static_cast<int>(radio * 0.55f);
    if (segmentos < 12)  segmentos = 12;
    if (segmentos > 128) segmentos = 128;

    glColor4f(r, g, b, alfa);
    glBegin(GL_TRIANGLE_FAN);
    glVertex2f(centroX, centroY); // vertice central del abanico
    for (int i = 0; i <= segmentos; ++i) {
        const float angulo = 2.0f * PI * static_cast<float>(i) /
                             static_cast<float>(segmentos);
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

// ---------------------------------------------------------------------------
// Disco ILUMINADO: en vez de un circulo plano, el color de cada vertice del
// borde se modula segun si esa parte del cuerpo mira o no hacia el Sol. El
// resultado es un terminador (la linea dia/noche) que apunta siempre al centro
// de la escena, igual que en un planeta real.
// ---------------------------------------------------------------------------
static void dibujarDiscoIluminado(float px, float py, float radio,
                                  float r, float g, float b, float alfa) {
    float dirSolX = g_escena.centroX - px;
    float dirSolY = g_escena.centroY - py;
    const float distancia = std::sqrt(dirSolX * dirSolX + dirSolY * dirSolY);
    if (distancia > 1.0e-3f) {
        dirSolX /= distancia;
        dirSolY /= distancia;
    } else {
        dirSolX = 0.0f;
        dirSolY = 1.0f;
    }

    glBegin(GL_TRIANGLE_FAN);
    glColor4f(r * 0.80f, g * 0.80f, b * 0.80f, alfa);
    glVertex2f(px, py);
    for (int i = 0; i <= SEGMENTOS_CIRCULO; ++i) {
        const float angulo = 2.0f * PI * static_cast<float>(i) /
                             static_cast<float>(SEGMENTOS_CIRCULO);
        const float nx = std::cos(angulo);
        const float ny = std::sin(angulo);
        // Producto punto entre la normal del borde y la direccion al Sol.
        const float incidencia = nx * dirSolX + ny * dirSolY;
        float luz = 0.28f + 0.88f * (0.5f + 0.5f * incidencia);
        if (luz > 1.15f) luz = 1.15f;
        glColor4f(r * luz, g * luz, b * luz, alfa);
        glVertex2f(px + radio * nx, py + radio * ny);
    }
    glEnd();
}

// ---------------------------------------------------------------------------
// Detalles de superficie. Cada mancha esta anclada a una longitud/latitud del
// cuerpo, asi que al girar este sobre su eje la mancha se desplaza por el disco
// y desaparece al pasar al lado oculto. La proyeccion es la de una esfera:
//     x = R * sin(longitud + rotacion) * cos(latitud)
//     y = R * sin(latitud)
// y el ancho se estrecha con cos(longitud + rotacion) (escorzo hacia el borde).
// ---------------------------------------------------------------------------
static void dibujarManchas(const CuerpoCeleste& cuerpo) {
    for (int m = 0; m < cuerpo.cantidadManchas; ++m) {
        const ManchaSuperficie& mancha = cuerpo.manchas[m];
        const float anguloVisible = mancha.longitud + cuerpo.anguloRotacion;
        const float profundidad = std::cos(anguloVisible);
        if (profundidad <= 0.12f) {
            continue;  // esta en la cara oculta del cuerpo
        }

        const float cosLatitud = std::cos(mancha.latitud);
        const float x = cuerpo.posX + cuerpo.radioCuerpo *
                        std::sin(anguloVisible) * cosLatitud;
        const float y = cuerpo.posY + cuerpo.radioCuerpo *
                        std::sin(mancha.latitud);
        const float radioMancha = mancha.tamano * cuerpo.radioCuerpo *
                                  profundidad;
        if (radioMancha < 0.6f) continue;

        // La mancha tambien se oscurece si cae en el lado nocturno.
        float dirSolX = g_escena.centroX - cuerpo.posX;
        float dirSolY = g_escena.centroY - cuerpo.posY;
        const float d = std::sqrt(dirSolX * dirSolX + dirSolY * dirSolY);
        float luz = 1.0f;
        if (d > 1.0e-3f) {
            const float nx = (x - cuerpo.posX) / cuerpo.radioCuerpo;
            const float ny = (y - cuerpo.posY) / cuerpo.radioCuerpo;
            const float incidencia = nx * (dirSolX / d) + ny * (dirSolY / d);
            luz = 0.32f + 0.80f * (0.5f + 0.5f * incidencia);
        }

        dibujarDisco(x, y, radioMancha,
                     cuerpo.manchaR * luz, cuerpo.manchaG * luz,
                     cuerpo.manchaB * luz, 0.85f);
    }
}

// ---------------------------------------------------------------------------
// Anillo planetario. Se dibuja en dos mitades: la de atras antes del cuerpo y
// la de adelante despues, para que el planeta quede correctamente intercalado.
// ---------------------------------------------------------------------------
static void dibujarMitadAnillo(const CuerpoCeleste& cuerpo, bool mitadTrasera) {
    if (!cuerpo.tieneAnillo) return;

    const int pasos = SEGMENTOS_CIRCULO;
    const float inicio = mitadTrasera ? 0.0f : PI;
    for (int capa = 0; capa < 2; ++capa) {
        const float radioX = cuerpo.radioCuerpo * (1.85f + 0.45f * capa);
        const float radioY = radioX * cuerpo.inclinacionAnillo;
        glColor4f(cuerpo.manchaR + 0.28f, cuerpo.manchaG + 0.26f,
                  cuerpo.manchaB + 0.22f, mitadTrasera ? 0.30f : 0.50f);
        glBegin(GL_LINE_STRIP);
        for (int i = 0; i <= pasos; ++i) {
            const float angulo = inicio + PI * static_cast<float>(i) /
                                 static_cast<float>(pasos);
            glVertex2f(cuerpo.posX + radioX * std::cos(angulo),
                       cuerpo.posY + radioY * std::sin(angulo));
        }
        glEnd();
    }
}

// ---------------------------------------------------------------------------
// Zarcillos de Dark Bramble: filamentos que salen del cuerpo y ondulan con la
// misma serie de armonicos que perturba su orbita, de modo que el adorno visual
// y el modelo fisico cuentan la misma historia.
// ---------------------------------------------------------------------------
static void dibujarZarcillos(const CuerpoCeleste& cuerpo, float tiempo) {
    const int cantidadZarcillos = 7;
    const int segmentos = 9;

    for (int z = 0; z < cantidadZarcillos; ++z) {
        const float anguloBase = 2.0f * PI * static_cast<float>(z) /
                                 static_cast<float>(cantidadZarcillos) +
                                 cuerpo.anguloRotacion;
        glBegin(GL_LINE_STRIP);
        for (int s = 0; s <= segmentos; ++s) {
            const float avance = static_cast<float>(s) /
                                 static_cast<float>(segmentos);
            const float largo = cuerpo.radioCuerpo * (1.0f + 2.6f * avance);
            const float ondulacion = 0.5f * avance *
                std::sin(2.2f * tiempo + static_cast<float>(z) * 1.7f +
                         avance * 4.5f + cuerpo.faseOscilacion);
            const float angulo = anguloBase + ondulacion;
            glColor4f(cuerpo.colorR + 0.18f, cuerpo.colorG + 0.22f,
                      cuerpo.colorB + 0.18f, 0.55f * (1.0f - avance));
            glVertex2f(cuerpo.posX + largo * std::cos(angulo),
                       cuerpo.posY + largo * std::sin(angulo));
        }
        glEnd();
    }
}

// ---------------------------------------------------------------------------
// Columna de arena entre los Hourglass Twins: hace visible el recurso
// compartido que los dos hilos modifican con #pragma omp atomic. Los granos
// avanzan del donante al receptor.
// ---------------------------------------------------------------------------
static void dibujarColumnaDeArena(const CuerpoCeleste& donante,
                                  const CuerpoCeleste& receptor,
                                  float tiempo) {
    const int granos = 16;
    const float dx = receptor.posX - donante.posX;
    const float dy = receptor.posY - donante.posY;
    const float distancia = std::sqrt(dx * dx + dy * dy);
    if (distancia < 1.0f) return;

    // Vector perpendicular, para dispersar ligeramente los granos.
    const float perpX = -dy / distancia;
    const float perpY =  dx / distancia;
    // Desplazamiento continuo: da la sensacion de flujo hacia el receptor.
    const float corrimiento = std::fmod(tiempo * 0.45f, 1.0f);

    for (int k = 0; k < granos; ++k) {
        float avance = (static_cast<float>(k) + corrimiento) /
                       static_cast<float>(granos);
        if (avance > 1.0f) avance -= 1.0f;
        // Se recorta cerca de los extremos para que la arena salga y entre
        // "desde" la superficie de cada gemelo, no desde su centro.
        if (avance < 0.12f || avance > 0.88f) continue;

        const float dispersion = (aleatorioDeterminista(
            static_cast<unsigned int>(k) * 2246822519u) - 0.5f) * 6.0f;
        const float x = donante.posX + dx * avance + perpX * dispersion;
        const float y = donante.posY + dy * avance + perpY * dispersion;
        // Se afina en el centro del chorro, como un reloj de arena.
        const float grosor = 1.1f + 1.5f * std::fabs(avance - 0.5f);
        dibujarDisco(x, y, grosor, 0.95f, 0.82f, 0.50f, 0.75f);
    }
}

// ---------------------------------------------------------------------------
// Sun Station: no es un disco, es una estructura. Se dibuja como un anillo
// visto en escorzo con radios que lo unen a un nucleo central, mas dos modulos
// en los extremos. El conjunto gira sobre si mismo con anguloRotacion.
// ---------------------------------------------------------------------------
static void dibujarEstacionSolar(const CuerpoCeleste& c, float alfa) {
    const float radioAnillo = c.radioAnillo;
    const float escorzo = 0.45f;   // achatamiento vertical del anillo
    const float giro = c.anguloRotacion;
    const float cosGiro = std::cos(giro);
    const float senGiro = std::sin(giro);

    // Transforma un punto del plano del anillo a coordenadas de pantalla.
    // El anillo se ve inclinado, asi que la componente vertical se comprime y
    // luego todo se rota por el angulo de giro de la estructura.
    #define ESTACION_PUNTO(ang, radio, sx, sy)                                  \
        do {                                                                    \
            const float px_ = (radio) * std::cos(ang);                          \
            const float py_ = (radio) * std::sin(ang) * escorzo;                \
            (sx) = c.posX + px_ * cosGiro - py_ * senGiro;                       \
            (sy) = c.posY + px_ * senGiro + py_ * cosGiro;                       \
        } while (0)

    // Halo tenue: la estacion esta al rojo por la cercania al Sol.
    dibujarDisco(c.posX, c.posY, radioAnillo * 1.25f, 1.0f, 0.72f, 0.35f, 0.12f);

    // Anillo exterior (dos trazos concentricos para darle grosor).
    const int pasos = SEGMENTOS_CIRCULO * 2;
    for (int capa = 0; capa < 2; ++capa) {
        const float radio = radioAnillo * (1.0f - 0.14f * capa);
        glColor4f(c.colorR, c.colorG, c.colorB, alfa * (0.9f - 0.25f * capa));
        glBegin(GL_LINE_LOOP);
        for (int i = 0; i < pasos; ++i) {
            const float ang = 2.0f * PI * static_cast<float>(i) /
                              static_cast<float>(pasos);
            float sx, sy;
            ESTACION_PUNTO(ang, radio, sx, sy);
            glVertex2f(sx, sy);
        }
        glEnd();
    }

    // Cuatro radios que unen el anillo con el nucleo.
    glColor4f(c.colorR, c.colorG, c.colorB, alfa * 0.75f);
    glBegin(GL_LINES);
    for (int i = 0; i < 4; ++i) {
        const float ang = PI * 0.5f * static_cast<float>(i);
        float sx, sy;
        ESTACION_PUNTO(ang, radioAnillo, sx, sy);
        glVertex2f(c.posX, c.posY);
        glVertex2f(sx, sy);
    }
    glEnd();

    // Modulos en dos extremos opuestos del anillo.
    for (int i = 0; i < 2; ++i) {
        const float ang = PI * static_cast<float>(i);
        float sx, sy;
        ESTACION_PUNTO(ang, radioAnillo, sx, sy);
        dibujarDisco(sx, sy, c.radioCuerpo * 0.42f,
                     c.manchaR + 0.25f, c.manchaG + 0.22f, c.manchaB + 0.18f,
                     alfa);
    }

    // Nucleo central, con una luz de aviso que parpadea.
    dibujarDisco(c.posX, c.posY, c.radioCuerpo * 0.55f,
                 c.colorR, c.colorG, c.colorB, alfa);
    const float baliza = 0.45f + 0.55f * std::fabs(std::sin(giro * 2.0f));
    dibujarDisco(c.posX, c.posY, c.radioCuerpo * 0.22f,
                 1.0f, 0.45f, 0.30f, alfa * baliza);

    #undef ESTACION_PUNTO
}

// ---------------------------------------------------------------------------
// Nave exploradora: casco triangular alargado, dos alerones, cabina iluminada
// y llama del motor proporcional al empuje. Todo se orienta segun el rumbo.
// ---------------------------------------------------------------------------
static void dibujarNave(const CuerpoCeleste& c, float tiempo, float alfa) {
    const float largo = c.radioCuerpo * 1.9f;
    const float ancho = c.radioCuerpo * 0.85f;
    const float cosR = std::cos(c.rumbo);
    const float senR = std::sin(c.rumbo);

    // Pasa un punto del sistema local de la nave (x hacia proa) a pantalla.
    #define NAVE_PUNTO(lx, ly, sx, sy)                     \
        do {                                                \
            (sx) = c.posX + (lx) * cosR - (ly) * senR;      \
            (sy) = c.posY + (lx) * senR + (ly) * cosR;      \
        } while (0)

    // --- Llama del motor (detras del casco) ---
    if (c.empuje > 0.02f) {
        const float parpadeo = 0.72f + 0.28f * std::sin(tiempo * 38.0f);
        const float largoLlama = largo * (0.85f + 1.5f * c.empuje) * parpadeo;
        float x1, y1, x2, y2, x3, y3;
        NAVE_PUNTO(-largo * 0.55f, -ancho * 0.42f, x1, y1);
        NAVE_PUNTO(-largo * 0.55f,  ancho * 0.42f, x2, y2);
        NAVE_PUNTO(-largo * 0.55f - largoLlama, 0.0f, x3, y3);
        glBegin(GL_TRIANGLES);
        glColor4f(1.0f, 0.78f, 0.35f, alfa * 0.85f);
        glVertex2f(x1, y1);
        glVertex2f(x2, y2);
        glColor4f(1.0f, 0.35f, 0.10f, 0.0f);
        glVertex2f(x3, y3);
        glEnd();
    }

    // --- Alerones ---
    glColor4f(c.manchaR + 0.10f, c.manchaG + 0.08f, c.manchaB + 0.06f, alfa);
    glBegin(GL_TRIANGLES);
    {
        float ax, ay, bx, by, cx2, cy2;
        NAVE_PUNTO(-largo * 0.25f, 0.0f, ax, ay);
        NAVE_PUNTO(-largo * 0.62f, -ancho * 1.15f, bx, by);
        NAVE_PUNTO(-largo * 0.62f, -ancho * 0.15f, cx2, cy2);
        glVertex2f(ax, ay); glVertex2f(bx, by); glVertex2f(cx2, cy2);

        NAVE_PUNTO(-largo * 0.25f, 0.0f, ax, ay);
        NAVE_PUNTO(-largo * 0.62f,  ancho * 1.15f, bx, by);
        NAVE_PUNTO(-largo * 0.62f,  ancho * 0.15f, cx2, cy2);
        glVertex2f(ax, ay); glVertex2f(bx, by); glVertex2f(cx2, cy2);
    }
    glEnd();

    // --- Casco ---
    glBegin(GL_TRIANGLES);
    {
        float proaX, proaY, popaIzqX, popaIzqY, popaDerX, popaDerY;
        NAVE_PUNTO(largo, 0.0f, proaX, proaY);
        NAVE_PUNTO(-largo * 0.55f, -ancho, popaIzqX, popaIzqY);
        NAVE_PUNTO(-largo * 0.55f,  ancho, popaDerX, popaDerY);
        glColor4f(c.colorR, c.colorG, c.colorB, alfa);
        glVertex2f(proaX, proaY);
        glColor4f(c.colorR * 0.65f, c.colorG * 0.65f, c.colorB * 0.65f, alfa);
        glVertex2f(popaIzqX, popaIzqY);
        glVertex2f(popaDerX, popaDerY);
    }
    glEnd();

    // --- Cabina ---
    {
        float cabinaX, cabinaY;
        NAVE_PUNTO(largo * 0.30f, 0.0f, cabinaX, cabinaY);
        dibujarDisco(cabinaX, cabinaY, c.radioCuerpo * 0.32f,
                     0.55f, 0.85f, 1.0f, alfa);
    }

    #undef NAVE_PUNTO
}

// Estela de la nave: las posiciones recientes, desvaneciendose hacia atras.
static void dibujarEstelaNave(float alfa) {
    if (g_escena.estelaLlenas < 2) return;

    glBegin(GL_LINE_STRIP);
    for (size_t k = 0; k < g_escena.estelaLlenas; ++k) {
        // Se recorre del mas antiguo al mas reciente.
        const size_t indice = (g_escena.estelaCursor + NAVE_LARGO_ESTELA -
                               g_escena.estelaLlenas + k) % NAVE_LARGO_ESTELA;
        const float f = static_cast<float>(k) /
                        static_cast<float>(g_escena.estelaLlenas);
        glColor4f(0.65f, 0.82f, 1.0f, alfa * 0.42f * f * f);
        glVertex2f(g_escena.estelaNaveX[indice], g_escena.estelaNaveY[indice]);
    }
    glEnd();
}

// ---------------------------------------------------------------------------
// Elementos de la secuencia final.
// ---------------------------------------------------------------------------

// Asteroide: roca irregular (radio modulado por armonicos) con una estela
// incandescente que apunta en direccion contraria a su movimiento.
static void dibujarAsteroide() {
    const float x = g_escena.asteroideX;
    const float y = g_escena.asteroideY;
    const float radio = g_escena.asteroideRadio;

    // Estela de entrada.
    const float rapidez = std::sqrt(g_escena.asteroideVelX * g_escena.asteroideVelX +
                                    g_escena.asteroideVelY * g_escena.asteroideVelY);
    if (rapidez > 1.0f) {
        const float ux = -g_escena.asteroideVelX / rapidez;
        const float uy = -g_escena.asteroideVelY / rapidez;
        const int pasos = 14;
        for (int k = 1; k <= pasos; ++k) {
            const float f = static_cast<float>(k) / static_cast<float>(pasos);
            dibujarDisco(x + ux * radio * 9.0f * f, y + uy * radio * 9.0f * f,
                         radio * (1.0f - 0.75f * f),
                         1.0f, 0.55f - 0.35f * f, 0.15f,
                         0.42f * (1.0f - f));
        }
    }

    // Cuerpo irregular.
    glBegin(GL_TRIANGLE_FAN);
    glColor4f(0.55f, 0.44f, 0.38f, 1.0f);
    glVertex2f(x, y);
    const int vertices = 14;
    for (int i = 0; i <= vertices; ++i) {
        const float ang = 2.0f * PI * static_cast<float>(i) /
                          static_cast<float>(vertices) + g_escena.asteroideGiro;
        // Radio irregular: dos armonicos le quitan la forma de circulo perfecto.
        const float irregular = radio * (1.0f + 0.22f * std::sin(3.0f * ang) +
                                                0.13f * std::sin(7.0f * ang));
        glColor4f(0.42f, 0.33f, 0.29f, 1.0f);
        glVertex2f(x + irregular * std::cos(ang), y + irregular * std::sin(ang));
    }
    glEnd();

    // Cara caliente, la que mira al Sol.
    float haciaSolX = g_escena.centroX - x;
    float haciaSolY = g_escena.centroY - y;
    const float d = std::sqrt(haciaSolX * haciaSolX + haciaSolY * haciaSolY);
    if (d > 1.0e-3f) {
        haciaSolX /= d;
        haciaSolY /= d;
        dibujarDisco(x + haciaSolX * radio * 0.35f,
                     y + haciaSolY * radio * 0.35f,
                     radio * 0.55f, 1.0f, 0.62f, 0.25f, 0.75f);
    }
}

// Cascaron de la supernova: la banda de material incandescente que se expande
// detras del frente. Se dibuja como un anillo relleno con degradado, brillante
// por dentro y transparente por fuera; es lo que hace legible la explosion una
// vez que el nucleo ya colapso.
static void dibujarCascaronExplosion() {
    const float radioExterior = g_escena.radioOnda;
    if (radioExterior <= 2.0f) return;

    const float grosor = 55.0f + radioExterior * 0.40f;
    float radioInterior = radioExterior - grosor;
    if (radioInterior < 0.0f) radioInterior = 0.0f;

    float avance = g_escena.tiempoEnEtapa / FINAL_DURACION_ONDA;
    if (avance > 1.0f) avance = 1.0f;
    const float alfa = 0.60f * (1.0f - avance) * (1.0f - avance);

    const int segmentos = 96;
    glBegin(GL_TRIANGLE_STRIP);
    for (int i = 0; i <= segmentos; ++i) {
        const float angulo = 2.0f * PI * static_cast<float>(i) /
                             static_cast<float>(segmentos);
        const float cosA = std::cos(angulo);
        const float senA = std::sin(angulo);

        // Borde interior: brillante. Borde exterior: se funde con el fondo.
        glColor4f(1.0f, 0.86f, 0.55f, alfa);
        glVertex2f(g_escena.centroX + radioInterior * cosA,
                   g_escena.centroY + radioInterior * senA);
        glColor4f(1.0f, 0.42f, 0.10f, 0.0f);
        glVertex2f(g_escena.centroX + radioExterior * cosA,
                   g_escena.centroY + radioExterior * senA);
    }
    glEnd();
}

// Onda expansiva: varios frentes concentricos que se persiguen y se apagan.
static void dibujarOndaExpansiva() {
    for (int capa = 0; capa < 3; ++capa) {
        const float radio = g_escena.radioOnda -
                            static_cast<float>(capa) * 34.0f;
        if (radio <= 0.0f) continue;

        const float desvanecido = 1.0f -
            g_escena.tiempoEnEtapa / FINAL_DURACION_ONDA;
        const float alfa = (desvanecido > 0.0f ? desvanecido : 0.0f) *
                           (0.85f - 0.22f * static_cast<float>(capa));

        glLineWidth(3.0f - static_cast<float>(capa));
        dibujarContorno(g_escena.centroX, g_escena.centroY, radio,
                        1.0f, 0.80f - 0.15f * static_cast<float>(capa), 0.45f,
                        alfa);
    }
    glLineWidth(1.0f);
}

// Fragmentos incandescentes lanzados por la supernova.
static void dibujarEscombros() {
    for (size_t i = 0; i < g_escena.escombros.size(); ++i) {
        const Escombro& fragmento = g_escena.escombros[i];
        if (fragmento.vida <= 0.0f) continue;
        const float f = fragmento.vida / fragmento.vidaTotal;
        dibujarDisco(fragmento.x, fragmento.y, fragmento.radio * (0.4f + 0.6f * f),
                     fragmento.r, fragmento.g, fragmento.b, f);
    }
}

// Capa de color a pantalla completa: el fogonazo blanco del impacto y el
// fundido a negro del cierre.
static void dibujarVelo(float r, float g, float b, float alfa) {
    if (alfa <= 0.0f) return;
    if (alfa > 1.0f) alfa = 1.0f;
    glColor4f(r, g, b, alfa);
    glBegin(GL_QUADS);
    glVertex2f(0.0f, 0.0f);
    glVertex2f(static_cast<float>(g_escena.anchoVentana), 0.0f);
    glVertex2f(static_cast<float>(g_escena.anchoVentana),
               static_cast<float>(g_escena.altoVentana));
    glVertex2f(0.0f, static_cast<float>(g_escena.altoVentana));
    glEnd();
}

// ---------------------------------------------------------------------------
// Fondo estelar. Las estrellas parpadean con una senoidal de frecuencia propia.
// ---------------------------------------------------------------------------
static void dibujarEstrellas(float tiempo) {
    const float ancho = static_cast<float>(g_escena.anchoVentana);
    const float alto  = static_cast<float>(g_escena.altoVentana);

    glPointSize(2.0f);
    glBegin(GL_POINTS);
    for (size_t i = 0; i < g_escena.estrellas.size(); ++i) {
        const Estrella& estrella = g_escena.estrellas[i];
        const float parpadeo = 0.75f + 0.25f *
            std::sin(tiempo * estrella.frecuencia + estrella.fase);
        const float intensidad = estrella.brillo * parpadeo;
        glColor4f(0.86f, 0.90f, 1.0f, intensidad);
        glVertex2f(estrella.x * ancho, estrella.y * alto);
    }
    glEnd();
    glPointSize(1.0f);
}

// ---------------------------------------------------------------------------
// Corona del Sol: fulgores que laten con periodos distintos, insinuando ya la
// inestabilidad que terminara en supernova.
// ---------------------------------------------------------------------------
static void dibujarCoronaSolar(float tiempo, float alfaGlobal) {
    if (alfaGlobal <= 0.01f) return;
    const float radio = g_escena.radioSolActual;
    for (int i = 0; i < RAYOS_CORONA; ++i) {
        const float angulo = 2.0f * PI * static_cast<float>(i) /
                             static_cast<float>(RAYOS_CORONA);
        const float pulso = 0.5f + 0.5f *
            std::sin(tiempo * (1.1f + 0.37f * static_cast<float>(i % 5)) +
                     static_cast<float>(i) * 2.3f);
        const float largo = radio * (1.25f + 0.75f * pulso);
        const float mediaBase = 0.10f;

        const float dirX = std::cos(angulo);
        const float dirY = std::sin(angulo);
        const float ladoX = std::cos(angulo + mediaBase);
        const float ladoY = std::sin(angulo + mediaBase);
        const float otroX = std::cos(angulo - mediaBase);
        const float otroY = std::sin(angulo - mediaBase);

        glBegin(GL_TRIANGLES);
        glColor4f(1.0f, 0.72f, 0.28f, 0.22f * alfaGlobal);
        glVertex2f(g_escena.centroX + radio * ladoX,
                   g_escena.centroY + radio * ladoY);
        glVertex2f(g_escena.centroX + radio * otroX,
                   g_escena.centroY + radio * otroY);
        glColor4f(1.0f, 0.45f, 0.08f, 0.0f);
        glVertex2f(g_escena.centroX + largo * dirX,
                   g_escena.centroY + largo * dirY);
        glEnd();
    }
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

// Dibuja el Sol: corona de fulgores, capas de halo y nucleo brillante.
// Durante la supernova el nucleo se blanquea y se va disipando, para que el
// fundido a negro del cierre no caiga sobre un disco amarillo plano.
static void dibujarSol(float tiempo) {
    float alfaSol   = 1.0f;
    float blanqueo  = 0.0f;

    if (g_escena.estadoFinal == FINAL_EXPLOSION ||
        g_escena.estadoFinal == FINAL_APAGADO) {
        float avance = (g_escena.estadoFinal == FINAL_APAGADO)
                           ? 1.0f
                           : g_escena.tiempoEnEtapa / FINAL_DURACION_ONDA;
        if (avance > 1.0f) avance = 1.0f;

        blanqueo = avance * 1.3f;
        if (blanqueo > 1.0f) blanqueo = 1.0f;

        // El nucleo se disipa pronto: lo que debe quedar en pantalla es el
        // cascaron y los escombros, no un disco plano ocupando el centro.
        const float inicioDisipacion = 0.25f;
        const float finDisipacion    = 0.70f;
        if (avance > inicioDisipacion) {
            alfaSol = 1.0f - (avance - inicioDisipacion) /
                             (finDisipacion - inicioDisipacion);
            if (alfaSol < 0.0f) alfaSol = 0.0f;
        }
    }

    dibujarCoronaSolar(tiempo, alfaSol);

    const int capasHalo = 10;
    for (int capa = capasHalo; capa >= 1; --capa) {
        const float factor = 1.0f + 0.22f * static_cast<float>(capa);
        dibujarDisco(g_escena.centroX, g_escena.centroY,
                     g_escena.radioSolActual * factor,
                     1.0f, 0.65f + 0.30f * blanqueo, 0.15f + 0.75f * blanqueo,
                     0.045f * alfaSol);
    }
    dibujarDisco(g_escena.centroX, g_escena.centroY, g_escena.radioSolActual,
                 1.0f, 0.85f + 0.15f * blanqueo, 0.35f + 0.60f * blanqueo,
                 alfaSol);
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

    const float tiempo = g_escena.tiempoAcumulado;
    const bool enExplosion = (g_escena.estadoFinal == FINAL_EXPLOSION ||
                              g_escena.estadoFinal == FINAL_APAGADO);
    // Nivel de detalle: con muchos cuerpos el adorno fino ni se distingue y el
    // renderizado (que es secuencial) se volveria el cuello de botella. Durante
    // la explosion tampoco se dibuja: los cuerpos ya van desintegrandose.
    const bool detalleFino = (g_escena.cuerpos.size() <= LIMITE_DETALLE_FINO) &&
                             !enExplosion;

    dibujarEstrellas(tiempo);

    // Orbitas de referencia (solo si son pocas, para no saturar la pantalla).
    // Durante la explosion no se dibujan: los cuerpos ya no las siguen.
    if (g_escena.cuerpos.size() <= 64 && !enExplosion) {
        for (size_t i = 0; i < g_escena.cuerpos.size(); ++i) {
            if (g_escena.cuerpos[i].tipo != FISICA_SATELITE) {
                dibujarOrbita(g_escena.cuerpos[i]);
            }
        }
    }

    dibujarSol(tiempo);

    // La columna de arena va por debajo de los gemelos, para que estos la
    // tapen en los extremos y parezca que sale de su superficie.
    if (detalleFino) {
        for (size_t i = 0; i < g_escena.cuerpos.size(); ++i) {
            const CuerpoCeleste& cuerpo = g_escena.cuerpos[i];
            if (cuerpo.tipo == FISICA_BINARIA && cuerpo.esDonanteDeArena &&
                cuerpo.indiceCompanero >= 0) {
                dibujarColumnaDeArena(
                    cuerpo,
                    g_escena.cuerpos[static_cast<size_t>(cuerpo.indiceCompanero)],
                    tiempo);
            }
        }
    }

    // Estela de la nave: se muestrea aqui, en el hilo de dibujado, para no
    // meter un arreglo por cuerpo (con N grande multiplicaria la memoria).
    if (g_escena.indiceNave >= 0) {
        const CuerpoCeleste& nave =
            g_escena.cuerpos[static_cast<size_t>(g_escena.indiceNave)];
        g_escena.estelaNaveX[g_escena.estelaCursor] = nave.posX;
        g_escena.estelaNaveY[g_escena.estelaCursor] = nave.posY;
        g_escena.estelaCursor = (g_escena.estelaCursor + 1) % NAVE_LARGO_ESTELA;
        if (g_escena.estelaLlenas < NAVE_LARGO_ESTELA) {
            g_escena.estelaLlenas += 1;
        }
        dibujarEstelaNave(nave.opacidad);
    }

    // Los cuerpos celestes, con el adorno propio de su tipo de fisica.
    for (size_t i = 0; i < g_escena.cuerpos.size(); ++i) {
        const CuerpoCeleste& cuerpo = g_escena.cuerpos[i];
        const float alfa = cuerpo.opacidad;
        if (alfa <= 0.0f) continue;   // ya se desintegro

        // --- Cuerpos con forma propia: no son discos ---
        if (cuerpo.tipo == FISICA_ESTACION) {
            dibujarEstacionSolar(cuerpo, alfa);
            continue;
        }
        if (cuerpo.tipo == FISICA_NAVE) {
            dibujarNave(cuerpo, tiempo, alfa);
            continue;
        }

        // --- Adornos que van DETRAS del cuerpo ---
        if (cuerpo.tipo == FISICA_KEPLERIANA && !enExplosion) {
            dibujarColaCometa(cuerpo);
        }
        if (detalleFino && cuerpo.tipo == FISICA_OSCILANTE) {
            dibujarZarcillos(cuerpo, tiempo);
        }
        if (detalleFino && cuerpo.tipo == FISICA_CUANTICA &&
            cuerpo.tiempoEstela > 0.0f) {
            // Estela del ultimo salto, desvaneciendose donde estaba antes.
            const float resto = cuerpo.tiempoEstela / DURACION_ESTELA;
            dibujarDisco(cuerpo.posEstelaX, cuerpo.posEstelaY,
                         cuerpo.radioCuerpo * (1.0f + 0.9f * (1.0f - resto)),
                         cuerpo.colorR, cuerpo.colorG, cuerpo.colorB,
                         0.40f * resto);
            dibujarContorno(cuerpo.posEstelaX, cuerpo.posEstelaY,
                            cuerpo.radioCuerpo * 2.4f,
                            cuerpo.colorR, cuerpo.colorG, cuerpo.colorB,
                            0.35f * resto);
        }
        if (detalleFino) {
            dibujarMitadAnillo(cuerpo, true);   // mitad trasera del anillo
        }

        // --- Atmosfera y cuerpo ---
        dibujarDisco(cuerpo.posX, cuerpo.posY, cuerpo.radioCuerpo * 1.42f,
                     cuerpo.colorR, cuerpo.colorG, cuerpo.colorB, 0.15f * alfa);
        dibujarDiscoIluminado(cuerpo.posX, cuerpo.posY, cuerpo.radioCuerpo,
                              cuerpo.colorR, cuerpo.colorG, cuerpo.colorB, alfa);

        // --- Adornos que van ENCIMA del cuerpo ---
        if (detalleFino) {
            dibujarManchas(cuerpo);
            dibujarMitadAnillo(cuerpo, false);  // mitad delantera del anillo

            if (cuerpo.tipo == FISICA_KEPLERIANA) {
                // Nucleo helado: un punto muy brillante en el centro.
                dibujarDisco(cuerpo.posX, cuerpo.posY, cuerpo.radioCuerpo * 0.45f,
                             0.97f, 0.99f, 1.0f, 0.85f);
            }
        }

        if (cuerpo.tipo == FISICA_CUANTICA) {
            dibujarContorno(cuerpo.posX, cuerpo.posY, cuerpo.radioCuerpo * 2.4f,
                            cuerpo.colorR, cuerpo.colorG, cuerpo.colorB,
                            0.55f * alfa);
        }
    }

    // ------------------- Secuencia de cierre -------------------
    if (g_escena.estadoFinal == FINAL_ASTEROIDE) {
        dibujarAsteroide();
    } else if (enExplosion) {
        dibujarCascaronExplosion();
        dibujarEscombros();
        dibujarOndaExpansiva();
        // Fogonazo blanco del impacto y, al final, fundido a negro.
        dibujarVelo(1.0f, 0.97f, 0.90f, g_escena.destello);
        if (g_escena.estadoFinal == FINAL_APAGADO) {
            dibujarVelo(0.0f, 0.0f, 0.0f,
                        g_escena.tiempoEnEtapa / FINAL_DURACION_APAGADO);
        }
    }

    // ---------------------------- HUD ----------------------------------
    // Se atenua en cuanto arranca la secuencia final: a partir de ahi lo que
    // interesa ver es la escena, no las metricas.
    float alfaHud = 1.0f;
    if (g_escena.estadoFinal == FINAL_ASTEROIDE) {
        alfaHud = 1.0f - g_escena.tiempoEnEtapa * 0.8f;
    } else if (g_escena.estadoFinal != FINAL_INACTIVO) {
        alfaHud = 0.0f;
    }
    if (alfaHud < 0.0f) alfaHud = 0.0f;

    if (alfaHud <= 0.01f) {
        glutSwapBuffers();
        return;
    }

    const float alto = static_cast<float>(g_escena.altoVentana);
    char linea[192];

    std::snprintf(linea, sizeof(linea),
                  "FPS: %6.2f   |   N = %ld (+estacion +nave)   |   %s",
                  g_escena.fpsActual, g_escena.cuerposSolicitados,
#ifdef USE_OPENMP
                  "OpenMP"
#else
                  "Distribuido"
#endif
                  );
    dibujarTexto(10.0f, alto - 22.0f, std::string(linea), 0.85f * alfaHud);

    // Composicion de la escena por modelo de fisica.
    std::string composicion = "fisica: ";
    for (int t = 0; t < CANTIDAD_TIPOS_FISICA; ++t) {
        if (g_escena.conteoPorTipo[t] == 0) continue;
        char parte[48];
        std::snprintf(parte, sizeof(parte), "%s=%d  ",
                      NOMBRES_FISICA[t], g_escena.conteoPorTipo[t]);
        composicion += parte;
    }
    dibujarTexto(10.0f, alto - 40.0f, composicion, 0.65f * alfaHud);

    // Con pocos cuerpos se muestra que hilo atendio a cada uno.
    if (g_escena.cuerpos.size() <= 12 && g_escena.cuerpos.size() > 0) {
        std::string asignacion = "cuerpo->hilo: ";
        for (size_t i = 0; i < g_escena.cuerpos.size(); ++i) {
            char parte[32];
            std::snprintf(parte, sizeof(parte), "%zu:h%d  ",
                          i, g_escena.cuerpos[i].hiloAsignado);
            asignacion += parte;
        }
        dibujarTexto(10.0f, alto - 58.0f, asignacion, 0.55f * alfaHud);
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
    // dtReal es el tiempo de pared que tardo el cuadro: es el que debe usarse
    // para medir FPS. dt es el paso de simulacion, recortado para que la fisica
    // no de un salto si la ventana se minimiza o el proceso se detiene.
    // Confundirlos hace que el contador de FPS se sature (nunca bajaria de
    // 1/DT_MAXIMO), justo en los valores de N que interesa medir.
    float dtReal = std::chrono::duration<float>(ahora - g_marcaTiempoPrevia).count();
    g_marcaTiempoPrevia = ahora;
    if (dtReal < 0.0f) dtReal = 0.0f;

    float dt = dtReal;
    if (dt > DT_MAXIMO) dt = DT_MAXIMO;

    actualizarEscena(dt);

    // Promedio de FPS sobre una ventana de tiempo fija, mas estable que 1/dt.
    g_escena.framesEnIntervalo += 1;
    g_escena.tiempoEnIntervalo += dtReal;
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
        // std::printf("FPS= %.2f  |  reparto por hilo:", g_escena.fpsActual);
        for (int h = 0; h < g_hilosDisponibles; ++h) {
            std::printf(" h%d=%d", h, cuerposPorHilo[static_cast<size_t>(h)]);
        }
        std::printf("\n");
        std::fflush(stdout);
    }

    glutPostRedisplay();
}

// ESC o 'q' no cierran de golpe: disparan la secuencia final (un asteroide cae
// sobre el Sol, este entra en supernova y la escena se desintegra) y el
// programa termina al acabar la animacion. Una segunda pulsacion salta la
// animacion y sale de inmediato, por si el usuario tiene prisa.
static void alPresionarTecla(unsigned char tecla, int x, int y) {
    (void)x; (void)y; // parametros no usados
    if (tecla == 27 || tecla == 'q' || tecla == 'Q') {
        if (g_escena.estadoFinal == FINAL_INACTIVO) {
            iniciarSecuenciaFinal();
        } else {
            std::printf("Secuencia final omitida. Cerrando screensaver.\n");
            std::exit(EXIT_SUCCESS);
        }
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
    std::printf("Fase 2 (dependientes):   %zu cuerpos, schedule(static)\n",
                g_escena.indicesDependientes.size());
#ifdef USE_OPENMP
    std::printf("Version: PARALELA (OpenMP, hasta %d hilos)\n",
                g_hilosDisponibles);
#else
    std::printf("Version: SECUENCIAL (%d hilo)\n", g_hilosDisponibles);
#endif
    std::printf("Ademas de los N cuerpos: Sun Station + nave exploradora.\n");
    std::printf("Presione ESC o 'q' sobre la ventana para iniciar la secuencia\n"
                "final (asteroide -> supernova); una segunda pulsacion la omite.\n\n");
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
