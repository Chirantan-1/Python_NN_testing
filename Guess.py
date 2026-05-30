import pygame
import numpy as np
from TheNN import NeuralNetwork

nn = NeuralNetwork(save_file="model.pkl")

nn.load()

grid_size = 28
square_size = 20
margin = 20

button_width = 140
button_height = 50

width = (grid_size * square_size) + (margin * 2)
height = (grid_size * square_size) + (margin * 3) + button_height + 60

white = (255, 255, 255)
black = (0, 0, 0)
gray = (120, 120, 120)
green = (0, 255, 0)

pygame.init()

screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Digit Recognition")

font = pygame.font.SysFont("Arial", 30)

pixels = np.zeros((28, 28))

result = ""

running = True

while running:

    screen.fill(white)

    for row in range(28):
        for col in range(28):

            color_value = int(pixels[row][col] * 255)

            pygame.draw.rect(
                screen,
                (color_value, color_value, color_value),
                (
                    margin + col * square_size,
                    margin + row * square_size,
                    square_size,
                    square_size
                )
            )

    guess_rect = pygame.Rect(
        margin,
        height - button_height - margin,
        button_width,
        button_height
    )

    clear_rect = pygame.Rect(
        margin + button_width + 20,
        height - button_height - margin,
        button_width,
        button_height
    )

    pygame.draw.rect(screen, gray, guess_rect)
    pygame.draw.rect(screen, gray, clear_rect)

    guess_text = font.render("GUESS", True, white)
    clear_text = font.render("CLEAR", True, white)

    screen.blit(
        guess_text,
        (
            guess_rect.x + 18,
            guess_rect.y + 10
        )
    )

    screen.blit(
        clear_text,
        (
            clear_rect.x + 20,
            clear_rect.y + 10
        )
    )

    result_text = font.render(result, True, green)

    screen.blit(
        result_text,
        (
            margin,
            height - button_height - margin - 50
        )
    )

    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            running = False

        if pygame.mouse.get_pressed()[0]:

            mx, my = pygame.mouse.get_pos()

            col = (mx - margin) // square_size
            row = (my - margin) // square_size

            if 0 <= row < 28 and 0 <= col < 28:

                pixels[row][col] = 1

                for dy in range(-1, 2):
                    for dx in range(-1, 2):

                        ny = row + dy
                        nx = col + dx

                        if 0 <= ny < 28 and 0 <= nx < 28:
                            pixels[ny][nx] = max(
                                pixels[ny][nx],
                                0.5
                            )

        if pygame.mouse.get_pressed()[2]:

            mx, my = pygame.mouse.get_pos()

            col = (mx - margin) // square_size
            row = (my - margin) // square_size

            if 0 <= row < 28 and 0 <= col < 28:
                pixels[row][col] = 0

        if event.type == pygame.MOUSEBUTTONDOWN:

            mx, my = pygame.mouse.get_pos()

            if guess_rect.collidepoint(mx, my):

                inp = pixels.reshape(1, 784)

                activations, _ = nn.forward(inp)

                prediction = np.argmax(activations[-1])

                result = f"Prediction: {prediction}"

            if clear_rect.collidepoint(mx, my):

                pixels = np.zeros((28, 28))
                result = ""

    pygame.display.flip()

pygame.quit()
