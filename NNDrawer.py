import pygame
import numpy as np

class NNDrawer:

    def __init__(self, nn, input_size, screen_size, output_size, output_labels, rows=None, cols=None):

        self.nn = nn
        self.input_size = input_size

        if rows is None and cols is None:
            rows = int(np.sqrt(input_size))
            cols = int(np.ceil(input_size / rows))
        elif rows is None:
            rows = int(np.ceil(input_size / cols))
        elif cols is None:
            cols = int(np.ceil(input_size / rows))

        while rows * cols < input_size:
            cols += 1

        self.rows = rows
        self.cols = cols

        self.square_size = min(screen_size // self.cols, screen_size // self.rows)

        self.output_size = output_size
        self.output_labels = output_labels

        self.margin = 20
        self.button_width = 140
        self.button_height = 50

        self.width = (self.cols * self.square_size) + (self.margin * 2)
        self.height = (self.rows * self.square_size) + (self.margin * 3) + self.button_height + 60

        self.white = (255, 255, 255)
        self.black = (0, 0, 0)
        self.gray = (120, 120, 120)
        self.green = (0, 255, 0)

        pygame.init()

        self.screen = pygame.display.set_mode((self.width, self.height))
        self.font = pygame.font.SysFont("Arial", 30)

        self.pixels = np.zeros((self.rows, self.cols))

        self.result = ""

    def predict(self):

        inp = self.pixels.flatten()

        if len(inp) > self.input_size:
            inp = inp[:self.input_size]
        elif len(inp) < self.input_size:
            inp = np.pad(inp, (0, self.input_size - len(inp)))

        inp = inp.reshape(1, self.input_size)

        activations, _ = self.nn.forward(inp)

        prediction = np.argmax(activations[-1])

        if prediction < len(self.output_labels):
            self.result = f"Prediction: {self.output_labels[prediction]}"
        else:
            self.result = f"Prediction: {prediction}"

    def clear(self):

        self.pixels = np.zeros((self.rows, self.cols))
        self.result = ""

    def draw_grid(self):

        for row in range(self.rows):
            for col in range(self.cols):

                color_value = int(self.pixels[row][col] * 255)

                pygame.draw.rect(self.screen, (color_value, color_value, color_value), (self.margin + col * self.square_size, self.margin + row * self.square_size, self.square_size, self.square_size))

    def paint(self):

        mx, my = pygame.mouse.get_pos()

        col = (mx - self.margin) // self.square_size
        row = (my - self.margin) // self.square_size

        if 0 <= row < self.rows and 0 <= col < self.cols:

            self.pixels[row][col] = 1

            for dy in range(-1, 2):
                for dx in range(-1, 2):

                    ny = row + dy
                    nx = col + dx

                    if 0 <= ny < self.rows and 0 <= nx < self.cols:
                        self.pixels[ny][nx] = max(self.pixels[ny][nx], 0.5)

    def erase(self):

        mx, my = pygame.mouse.get_pos()

        col = (mx - self.margin) // self.square_size
        row = (my - self.margin) // self.square_size

        if 0 <= row < self.rows and 0 <= col < self.cols:
            self.pixels[row][col] = 0

    def run(self, title="NN Drawer"):

        pygame.display.set_caption(title)

        running = True

        while running:

            self.screen.fill(self.white)

            self.draw_grid()

            guess_rect = pygame.Rect(self.margin, self.height - self.button_height - self.margin, self.button_width, self.button_height)
            clear_rect = pygame.Rect(self.margin + self.button_width + 20, self.height - self.button_height - self.margin, self.button_width, self.button_height)

            pygame.draw.rect(self.screen, self.gray, guess_rect)
            pygame.draw.rect(self.screen, self.gray, clear_rect)

            guess_text = self.font.render("GUESS", True, self.white)
            clear_text = self.font.render("CLEAR", True, self.white)

            self.screen.blit(guess_text, (guess_rect.x + 18, guess_rect.y + 10))
            self.screen.blit(clear_text, (clear_rect.x + 20, clear_rect.y + 10))

            result_text = self.font.render(self.result, True, self.green)

            self.screen.blit(result_text, (self.margin, self.height - self.button_height - self.margin - 50))

            for event in pygame.event.get():

                if event.type == pygame.QUIT:
                    running = False

                if pygame.mouse.get_pressed()[0]:
                    self.paint()

                if pygame.mouse.get_pressed()[2]:
                    self.erase()

                if event.type == pygame.MOUSEBUTTONDOWN:

                    mx, my = pygame.mouse.get_pos()

                    if guess_rect.collidepoint(mx, my):
                        self.predict()

                    if clear_rect.collidepoint(mx, my):
                        self.clear()

            pygame.display.flip()

        pygame.quit()
