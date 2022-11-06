import pandas as pd
import numpy as np
import pygame

BLACK = (  0,   0,   0)
WHITE = (255, 255, 255)

df = pd.read_csv('../data/result.csv')

preds = df['Predicted Angles']
true = df['Actual Angles']
filenames = list(df['File'])
images = []

pygame.init()
size = (640, 320)
pygame.display.set_caption("Data viewer")
screen = pygame.display.set_mode(size, pygame.DOUBLEBUF)
myfont = pygame.font.SysFont("monospace", 25)

for i in range(len(filenames)):

    angle = preds.iloc[i]
    true_angle = true.iloc[i]

    img = pygame.image.load('../data/test/'+filenames[i])
    screen.blit(img, (0, 0))

    pred_txt = myfont.render("Prediction:" + str(round(angle* 57.2958, 3)), 1, WHITE)
    true_txt = myfont.render("True angle:" + str(round(true_angle* 57.2958, 3)), 1, WHITE)
    screen.blit(pred_txt, (10, 275))
    screen.blit(true_txt, (10, 300))

    # draw steering wheel
    radius = 50
    pygame.draw.circle(screen, WHITE, [320, 300], radius, 2)

    # draw cricle for true angle
    x = radius * np.cos(np.pi/2 + true_angle)
    y = radius * np.sin(np.pi/2 + true_angle)
    pygame.draw.circle(screen, WHITE, [320 + int(x), 300 - int(y)], 7)

    # draw cricle for predicted angle
    x = radius * np.cos(np.pi/2 + angle)
    y = radius * np.sin(np.pi/2 + angle)
    pygame.draw.circle(screen, BLACK, [320 + int(x), 300 - int(y)], 5)

    #pygame.display.update()
    pygame.display.flip()
    pygame.time.wait(25)
    pygame.event.pump()
