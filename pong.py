import pygame
import sys

# Inicializar Pygame
pygame.init()

# Definir colores
BLANCO = (255, 255, 255)
NEGRO = (0, 0, 0)
AZUL = (0, 0, 255)
ROJO = (255, 0, 0)

# Definir tamaños
ANCHO = 800
ALTO = 600
RADIO_BOLA = 15
ANCHO_RAQUETA = 15
ALTO_RAQUETA = 100

# Crear la ventana
pantalla = pygame.display.set_mode((ANCHO, ALTO))
pygame.display.set_caption("Pong")

# Reloj para controlar el FPS
reloj = pygame.time.Clock()

# Posición inicial de las raquetas y la pelota
raqueta_izquierda = pygame.Rect(50, ALTO // 2 - ALTO_RAQUETA // 2, ANCHO_RAQUETA, ALTO_RAQUETA)
raqueta_derecha = pygame.Rect(ANCHO - 50 - ANCHO_RAQUETA, ALTO // 2 - ALTO_RAQUETA // 2, ANCHO_RAQUETA, ALTO_RAQUETA)
bola = pygame.Rect(ANCHO // 2 - RADIO_BOLA // 2, ALTO // 2 - RADIO_BOLA // 2, RADIO_BOLA, RADIO_BOLA)

# Velocidades de la bola
velocidad_bola_x = 5
velocidad_bola_y = 5

# Velocidades de las raquetas
velocidad_raqueta = 10

# Variables de puntuación
puntuacion_izquierda = 0
puntuacion_derecha = 0

# Fuente para la puntuación
fuente = pygame.font.SysFont("Arial", 30)

def dibujar():
    # Rellenar la pantalla de blanco
    pantalla.fill(NEGRO)

    # Dibujar las raquetas y la bola
    pygame.draw.rect(pantalla, AZUL, raqueta_izquierda)
    pygame.draw.rect(pantalla, ROJO, raqueta_derecha)
    pygame.draw.ellipse(pantalla, BLANCO, bola)



    # Dibujar las puntuaciones
    texto_izquierda = fuente.render(str(puntuacion_izquierda), True, BLANCO)
    texto_derecha = fuente.render(str(puntuacion_derecha), True, BLANCO)
    pantalla.blit(texto_izquierda, (ANCHO // 4, 20))
    pantalla.blit(texto_derecha, (ANCHO - ANCHO // 4 - texto_derecha.get_width(), 20))

    # Actualizar pantalla
    pygame.display.flip()

def mover_raquetas():
    keys = pygame.key.get_pressed()

    if keys[pygame.K_w] and raqueta_izquierda.top > 0:
        raqueta_izquierda.y -= velocidad_raqueta
    if keys[pygame.K_s] and raqueta_izquierda.bottom < ALTO:
        raqueta_izquierda.y += velocidad_raqueta
    if keys[pygame.K_UP] and raqueta_derecha.top > 0:
        raqueta_derecha.y -= velocidad_raqueta
    if keys[pygame.K_DOWN] and raqueta_derecha.bottom < ALTO:
        raqueta_derecha.y += velocidad_raqueta

def mover_bola():
    global velocidad_bola_x, velocidad_bola_y, puntuacion_izquierda, puntuacion_derecha

    bola.x += velocidad_bola_x
    bola.y += velocidad_bola_y

    # Colisiones con las paredes superior e inferior
    if bola.top <= 0 or bola.bottom >= ALTO:
        velocidad_bola_y = -velocidad_bola_y

    # Colisiones con las raquetas
    if bola.colliderect(raqueta_izquierda) or bola.colliderect(raqueta_derecha):
        velocidad_bola_x = -velocidad_bola_x
        
    # Puntos cuando la bola sale por los laterales
    if bola.left <= 0:
        puntuacion_derecha += 1
        bola.x = ANCHO // 2 - RADIO_BOLA // 2
        bola.y = ALTO // 2 - RADIO_BOLA // 2
        velocidad_bola_x = -velocidad_bola_x

    if bola.right >= ANCHO:
        puntuacion_izquierda += 1
        bola.x = ANCHO // 2 - RADIO_BOLA // 2
        bola.y = ALTO // 2 - RADIO_BOLA // 2
        velocidad_bola_x = -velocidad_bola_x

# Bucle principal
while True:
    for evento in pygame.event.get():
        if evento.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    mover_raquetas()
    mover_bola()
    dibujar()
    reloj.tick(60)