import os
import pygame
import random
import neat
import pickle
import time

# INIT:
pygame.init()
window = pygame.display.set_mode((400, 600))
pygame.display.set_caption("Flappy Bird AI")
# PICTURES:
basePic = pygame.transform.scale2x(pygame.image.load("Images/base.png").convert_alpha())
bgPic = pygame.transform.scale(pygame.image.load("Images/bg.png").convert_alpha(), (400, 600))
pipePic = pygame.image.load("Images/pipe.png")
birdPic = [pygame.image.load("Images/bird1.png"), pygame.image.load("Images/bird2.png"),
           pygame.image.load("Images/bird3.png")]
# birdPic = [pygame.transform.scale2x(pygame.image.load("Images/bird" + str(x) + ".png")) for x in range(1, 4)]  !!! F E T T S A C K BIRDMODUS !!!
#
clock = pygame.time.Clock()
gen = 0
STAT_FONT = pygame.font.SysFont("comicsans", 50)


#
class Bird:
    BIRD_WIDTH = 34
    BIRD_HEIGHT = 24

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.tick_count = 0
        self.velocity = 12
        self.tilt = 0
        self.tilt_up = True
        self.c = 0
        self.img = birdPic[0]

    def move(self):
        self.y = self.y - self.velocity + 2 * (self.tick_count ** 2)

    def fly(self):
        self.tick_count = 0
        self.tilt_up = True

    def draw(self):
        self.c += 1
        if self.c <= 2:
            self.img = birdPic[0]
        elif self.c <= 4:
            self.img = birdPic[1]
        elif self.c < 6:
            self.img = birdPic[2]
        elif self.c < 8:
            self.img = birdPic[1]
        elif self.c == 8:
            self.c = 0
        if self.tilt_up:
            self.img = pygame.transform.rotate(self.img, self.tilt)
            if self.tilt < 30:
                self.tilt += 10

        elif not self.tilt_up:
            self.img = pygame.transform.rotate(self.img, self.tilt)
            if self.tilt > -79:
                self.tilt -= 5

        window.blit(self.img, (self.x, self.y))

    def hit(self, pipe_x, pipe_boty):
        if self.x + self.BIRD_WIDTH - 10 > pipe_x and self.x <= pipe_x + Pipe.PIPE_WIDTH - 10:  ### man könnte daraus ne eigene Funktion, " hit " machen
            if self.y + self.BIRD_HEIGHT >= pipe_boty + 10 or self.y <= pipe_boty - Pipe.pipe_gap - 10:  ### wenn bird in die pipe fliegt wird das spiel neugestartet
                return True
        return False


class Pipe:
    boty = 0
    pipe_gap = 7 * Bird.BIRD_HEIGHT
    x = 0
    PIPE_BOT = pygame.transform.scale2x(pygame.image.load("Images/pipe.png").convert_alpha())
    PIPE_TOP = pygame.transform.flip(PIPE_BOT, False, True)
    PIPE_WIDTH = 104
    PIPE_HEIGHT = 640
    PIPE_VELOCITY = 4

    def __init__(self, boty, x):
        self.x = x
        self.boty = boty

    def draw(self):
        window.blit(self.PIPE_BOT, (self.x, self.boty))
        window.blit(self.PIPE_TOP, (self.x, self.boty - (self.pipe_gap + self.PIPE_HEIGHT)))


#
def redrawGameWindow(birds, pipes, score, gen):
    # insert all draw functions of various classes
    # insert constant images which are to be blit on the screen, e.g. background, score, etc
    window.blit(bgPic, (0, 0))
    for bird in birds:
        for pipe in pipes:
            bird.draw()
            pipe.draw()
        if len(pipes) > 0 and len(birds) > 0:
            if len(pipes) == 1:
                pipe_in = 0
            if len(pipes) == 2:
                pipe_in = 1
            pygame.draw.lines(window, (255, 0, 0), True,
                              ((bird.x + Bird.BIRD_WIDTH, bird.y + Bird.BIRD_HEIGHT), (pipes[pipe_in].x, pipes[pipe_in].boty), (pipes[pipe_in].x, pipes[pipe_in].boty - Pipe.pipe_gap)), 1)
    # score
    score_label = STAT_FONT.render("Score: " + str(score), 1, (255, 255, 255))
    window.blit(score_label, (600 - score_label.get_width() - 15, 10))

    # generations
    if gen == 0:
        gen = 1
    score_label = STAT_FONT.render("Gens: " + str(gen - 1), 1, (255, 255, 255))
    window.blit(score_label, (10, 10))

    # alive
    score_label = STAT_FONT.render("Alive: " + str(len(birds)), 1, (255, 255, 255))
    window.blit(score_label, (10, 50))

    pygame.display.update()

#
def eval_genomes(genomes, config):  # required for NEAT
    global gen
    gen += 1
    score = 0
    active = False
    play = True
    pipe_count = -40
    pipe_index = 0
    birds = []
    pipes = [(Pipe(random.randint(150, 450), 400))]
    nets = []
    ge = []

    for genome_id, g in genomes:
        g.fitness = 0
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        birds.append(Bird(100, 300))
        ge.append(g)
    #
    while play and len(birds) > 0:
        clock.tick(30)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                play = False
                pygame.quit()
                quit()
                break

        if pipe_count < 50 and len(pipes) <= 2:
            pipe_count += 1
            if pipe_count == 50:
                pipes.append(Pipe(random.randint(150, 450), 400))
                pipe_count = -40
        if pipes[0].x + Pipe.PIPE_WIDTH <= 0:
            pipes.pop(0)

        if pipes[0].x + Pipe.PIPE_WIDTH < 300 and len(pipes) == 2:
            pipe_index = 1
        if pipe_index == 1 and len(pipes) < 2:
            pipe_index = 0

        for i, bird in enumerate(birds):
            output = nets[birds.index(bird)].activate(
                (
                    bird.y, abs(bird.y - pipes[pipe_index].boty),
                    abs(bird.y - abs(pipes[pipe_index].boty - Pipe.pipe_gap)),
                    abs(pipes[pipe_index].x + Pipe.PIPE_WIDTH - bird.x)))
            if output[0] > 0:
                bird.fly()
            elif bird.tick_count < 3.3:
                bird.tick_count += 0.4
                if bird.tick_count > 2.4:
                    bird.tilt_up = False
            bird.move()

        for pipe in pipes:
            pipe.x = pipe.x - Pipe.PIPE_VELOCITY
            for i, bird in enumerate(birds):
                ge[i].fitness += 0.1
                if bird.y + bird.BIRD_HEIGHT >= 600 or bird.hit(pipe.x, pipe.boty) or bird.y <= 0:
                    ge[i].fitness -= 1
                    birds.pop(i)
                    nets.pop(i)
                    ge.pop(i)
                elif bird.x == pipe.x + Pipe.PIPE_WIDTH:
                    ge[i].fitness += 5

        redrawGameWindow(birds, pipes, score, gen)  # constantly loop through redrawGameWindow function to update the blit images


# AI

def run(config_file):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                                # siehe config_feedforward... datei
                                neat.DefaultStagnation, config_file)  # alle abschnitte kommen in die klammern
    # außer der erste (bereits vorausgesetzt)
    population = neat.Population(config)  # einstellen der population wie in der config datei
    population.add_reporter(neat.StdOutReporter(True))  # gibt zusätzliche informationen in der console aus
    stats = neat.StatisticsReporter()  # weitere statistische infos werden ausgegeben
    population.add_reporter(stats)

    winner = population.run(eval_genomes, 50)
    print('\nBest genome:\n{!s}'.format(winner))


if __name__ == "__main__":  # googlen... immer machen, die configure Datei muss geladen
    local_dir = os.path.dirname(__file__)  # werden, indem man dessen speicherort findet
    config_path = os.path.join(local_dir,
                               "config_feedforward_FlappyBirdAI.txt")  # diese wird der run funktion eingespeist
    run(config_path)
