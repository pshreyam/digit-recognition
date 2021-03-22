import pygame
import sys

import tensorflow as tf
import numpy as np
import cv2 as cv
import os

model = tf.keras.models.load_model('models/model.h5')

pygame.init()
SCREEN_SIZE = 400
screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
pygame.display.set_caption('Guess the number')
clock = pygame.time.Clock()

FPS = 70

pen_color = (0, 0, 0)
screen_bgcolor = (255, 255, 255)

screen.fill(screen_bgcolor)

def draw(x, y):
    point = pygame.rect.Rect(x - 10, y - 10, 20, 20)
    pygame.draw.ellipse(screen, pen_color, point)

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        
        if event.type == pygame.MOUSEBUTTONDOWN:
            x, y = event.pos
            draw(x, y)
    
        if event.type == pygame.MOUSEMOTION:
            if pygame.mouse.get_pressed()[0]:
                x, y = event.pos
                draw(x, y)

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                screen.fill(screen_bgcolor)
        
            if event.key == pygame.K_p:
                pygame.image.save(screen, "test-image.png")

                img = cv.imread("test-image.png")[:,:,0]
                img = cv.resize(img, (28, 28))
                img = np.invert(np.array([img]))
                prediction = model.predict(img)
                if os.name == "posix":
                    os.system("clear")
                elif os.name == "nt":
                    os.system("cls")
                print(f"Prediction: {np.argmax(prediction)} with confidence: {np.amax(prediction) * 100} %")

    pygame.display.update()
    clock.tick(FPS)