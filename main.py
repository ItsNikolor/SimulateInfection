import math
import random
import sys

import numpy as np
from numba import njit
from numpy.random import choice
import pygame
import pygame.freetype
from pygame.locals import *


def distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


class InputBox:
    COLOR_INACTIVE = pygame.Color('lightskyblue3')
    COLOR_ACTIVE = pygame.Color('dodgerblue2')

    def __init__(self, x, y, w, h, font, m, text=''):
        self.rect = pygame.Rect(x, y, w, h)
        self.color = InputBox.COLOR_INACTIVE
        self.text = text
        self.font = font
        self.txt_surface, _ = font.render(text, (255, 255, 255))
        self.active = False

        self.city = None
        self.m = m

    def assign_city(self, city):
        self.text = str(city.vaccination) if city.vaccination else ''
        self.txt_surface, _ = self.font.render(self.text, (255, 255, 255))
        self.city = city
        self.active = True
        self.color = InputBox.COLOR_ACTIVE if self.m.tick_spent == -1 else InputBox.COLOR_INACTIVE

    def handle_event(self, event):
        if self.m.tick_spent != -1 or self.city is None:
            return
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.active = not self.active
            else:
                self.active = False
            self.color = InputBox.COLOR_ACTIVE if self.active else InputBox.COLOR_INACTIVE
            self.draw(self.m.win)
        if event.type == pygame.KEYDOWN:
            if self.active:
                if event.key == pygame.K_BACKSPACE:
                    before = int(self.text) if self.text else 0
                    self.text = self.text[:-1]
                    after = before // 10

                    self.m.money = self.m.money + (before - after) * self.m.cost
                    self.city.vaccination = after

                    self.city.draw_info(self.m.win)
                    self.txt_surface, _ = self.font.render(self.text, (255, 255, 255))
                    self.draw((self.m.win))

                else:
                    if event.unicode and event.unicode in '0123456789':
                        if (self.text == '' and event.unicode != '0') or self.text != '':
                            before = int(self.text) if self.text else 0
                            self.text += event.unicode
                            after = before * 10 + int(event.unicode)
                            if (after * self.m.cost > (self.m.money + before * self.m.cost) or
                                    after > self.city.count[Person.Status.DEFAULT]):
                                self.text = self.text[:-1]
                            else:
                                self.m.money = self.m.money + (before - after) * self.m.cost
                                self.city.vaccination = after
                                self.city.draw_info(self.m.win)
                                self.txt_surface, _ = self.font.render(self.text, (255, 255, 255))
                                self.draw((self.m.win))

    def draw(self, screen):
        screen.blit(self.txt_surface, (self.rect.x + 5, self.rect.y + 5))
        pygame.draw.rect(screen, self.color, self.rect, 2)
        pygame.display.update((0, 0, self.m.RESERVED_W, Map.TEXTSIZE * 8 + 7))


class Map:
    SPEED = 2
    P_W = 0.3
    TEXTSIZE = 20
    P_H = TEXTSIZE * 9 + 5

    p_dead = None
    overload = None
    p_overload = None
    p_infect = None
    self_isolation = None
    social_distance = None

    BAD_ROUND = False

    def __init__(self, filename, font):
        self.font = pygame.freetype.Font(font, Map.TEXTSIZE)
        self.input_box = InputBox(0, Map.TEXTSIZE * 6, 200, Map.TEXTSIZE + 5, self.font, self, "")
        self.time_spent = 0
        self.tick_spent = -1
        self.current_round = 0
        self.need_draw = True
        self.active = None

        info_object = pygame.display.Info()
        self.W = info_object.current_w
        self.H = info_object.current_h

        self.RESERVED_W = int(self.W * Map.P_W)
        self.RESERVED_H = self.H - Map.P_H

        self.win = pygame.display.set_mode((self.W, self.H), pygame.FULLSCREEN | pygame.HWSURFACE)
        pygame.display.set_caption('Map')

        self.cities = []
        with open(filename, 'r', encoding='utf-8') as f:

            self.vvp = float(f.readline().split('=')[1])
            self.money = self.vvp

            self.cost = float(f.readline().split('=')[1])
            self.minimum_wage = float(f.readline().split('=')[1])

            Map.p_infect = float(f.readline().split('=')[1])
            Map.p_dead = float(f.readline().split('=')[1])
            Map.overload = float(f.readline().split('=')[1])
            Map.p_overload = float(f.readline().split('=')[1])
            p_begin = float(f.readline().split('=')[1])
            Map.self_isolation = float(f.readline().split('=')[1])
            Map.social_distance = float(f.readline().split('=')[1])
            self.round_time = int(f.readline().split('=')[1])

            Person.HEAL_LENGTH = int(f.readline().split('=')[1])
            Person.PLACE_LENGTH = int(f.readline().split('=')[1])

            for line in f:
                name, n, density, place_count = line[:-1].split(' ')
                n = int(n)
                density = float(density)
                place_count = float(place_count)
                r = min(n // 2, 100)
                count = int(n * n * density)

                failed = 0
                while True:
                    x = random.randint(r, self.RESERVED_W - r)
                    y = random.randint(r + Map.P_H, Map.P_H + self.RESERVED_H - r)

                    for c in self.cities:
                        if distance(c.center, (x, y)) < r + c.r:
                            break
                    else:
                        break

                    failed += 1
                    if failed == 100:
                        break
                self.cities.append(City((x, y), r, name, n, count,
                                        self.W - self.RESERVED_W, self.H,
                                        self.RESERVED_W, p_begin, place_count,
                                        self.font, self))

    def draw(self):
        self.win.fill((0, 0, 0))
        for c in self.cities:
            c.draw(self.win)
        if self.active:
            self.active.draw_grid(self.win)
            self.active.draw_info(self.win)
            self.active.draw_people(self.win)
            self.input_box.draw(self.win)
        self.need_draw = False
        pygame.display.update()

    def check_click(self, p):
        for c in self.cities:
            if c.check_click(p):
                self.active = c
                self.input_box.assign_city(self.active)
                self.need_draw = True
                return True
        return False

    def step(self):
        if self.tick_spent % Map.SPEED == 0:
            for c in self.cities:
                c.step()
            self.need_draw = True
            self.time_spent += 1
            if self.time_spent % self.round_time == 0:
                self.tick_spent = -1
                self.current_round += 1

                sick_count = 0
                healthy_count = 0
                person_count = 0
                for c in self.cities:
                    sick_count += c.count[Person.Status.SICK]
                    healthy_count += c.count[Person.Status.DEFAULT] + c.count[Person.Status.HEALTHY]
                    person_count += len(c.persons.action)
                self.money += self.vvp * healthy_count / person_count
                self.money -= sick_count * self.minimum_wage
                if self.money < 0:
                    Map.BAD_ROUND = True
                else:
                    Map.BAD_ROUND = False
                self.money = max(0, self.money)

                self.input_box.color = InputBox.COLOR_ACTIVE

        self.tick_spent += self.tick_spent != -1

        if self.need_draw:
            self.draw()

    def start_simulation(self):
        self.tick_spent = 0
        self.input_box.text = ''
        self.input_box.color = InputBox.COLOR_INACTIVE
        self.input_box.txt_surface, _ = self.font.render(self.input_box.text, (255, 255, 255))
        for c in self.cities:
            c.vaccinate()

    def handle_event(self, event):
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                pygame.quit()
                sys.exit()
            if event.key == pygame.K_RETURN:
                if self.tick_spent == -1:
                    self.start_simulation()

        self.input_box.handle_event(event)
        if event.type == pygame.MOUSEBUTTONDOWN:
            pos = pygame.mouse.get_pos()
            self.check_click(pos)


class Person:
    HOME_LENGTH = None
    PLACE_LENGTH = None
    HEAL_LENGTH = None

    class Color:
        DEFAULT = (0, 255, 255)
        SICK = (255, 0, 0)
        HEALTHY = (0, 255, 0)
        HOME_DEFAULT = (165, 10, 255)
        HOME_SICK = (255, 255, 0)
        HOME_HEALTHY = (125, 125, 125)
        DEAD = (255, 255, 255)

    class Status:
        DEFAULT = 0
        SICK = 1
        HEALTHY = 2
        DEAD = 3

    class Action:
        SitHome = 0
        WalkingPlace = 1
        SitPlace = 2
        WalkingHome = 3
        Choose = 4
        DEAD = 5

    def __init__(self, x_home, y_home, status):
        self.status = status
        self.action = Person.Action.Choose

        self.sit = 0
        self.sick = 0

        self.x = x_home
        self.y = y_home

        self.x_home = x_home
        self.y_home = y_home

        self.x_to = -1
        self.y_to = -1

        self.places = None

    def add_places(self, places, p):
        draw = choice(np.arange(len(places)), 200, p=p)
        self.places = [places[i] for i in draw]


class Persons:
    def __init__(self, persons, city):
        self.city = city

        status = []
        action = []
        sit = []
        sick = []
        x = []
        y = []
        places = []

        for p in persons:
            status.append(p.status)
            action.append(p.action)
            sit.append(p.sit)
            sick.append(p.sick)
            x.append(p.x)
            y.append(p.y)
            places.append(p.places)

        self.status = np.array(status, dtype=int)
        self.action = np.array(action, dtype=np.int32)
        self.sit = np.array(sit, dtype=int)
        self.sick = np.array(sick, dtype=int)

        self.X = np.array([*zip(x, y)], dtype=int)
        self.X_home = np.array(self.X, dtype=int)
        self.X_to = np.array(self.X, dtype=int)

        self.places = np.array(places, dtype=int)

        self.place_ind = 0

    def step(self):
        step(self.action, self.sit, self.status, self.sick,
             self.X_to, self.X_home, self.X, self.places, self.place_ind,
             Person.HEAL_LENGTH, Person.PLACE_LENGTH, Person.Action.Choose,
             Person.Action.WalkingPlace, Person.Action.SitPlace, Person.Action.WalkingHome,
             Person.Status.SICK, Person.Status.HEALTHY, self.city.count,
             Map.self_isolation, Person.Action.DEAD, Person.Status.DEAD,
             Map.p_dead + Map.BAD_ROUND * 0.1 +
             min(1, self.city.count[Person.Status.SICK] / (len(self.action) * Map.overload)) * Map.p_overload,
             Person.Status.DEFAULT, self.city.grid.cells, self.city.grid.grid_size, Map.p_infect, Map.social_distance)
        self.place_ind = (self.place_ind + 1) % self.places.shape[1]


@njit
def coord_to_ind(coord, n):
    return coord[1] * n + coord[0]


@njit
def step(action, sit,
         status, sick,
         X_to, X_home, X,
         places, place_i,
         HEAL_LENGTH, PLACE_LENGTH, Choose,
         WalkingPlace, SitPlace, WalkingHome,
         SICK, HEALTHY, count, SELF_ISOLATION,
         DEAD_ACTION, DEAD_STATUS, P_DEAD,
         DEFAULT_STATUS, cells, grid_n, P_INFECT, SOCIAL_DISTANCE):
    i = 0
    while i != len(action):
        if action[i] == DEAD_ACTION:
            i += 1
            continue
        elif action[i] == Choose:
            place = places[i, place_i]
            dist = 1 + 2 * np.abs(X_home[i] - place).sum() + PLACE_LENGTH
            p = SELF_ISOLATION * dist / (1 + SELF_ISOLATION * dist - SELF_ISOLATION)
            if random.random() > p:
                cells[coord_to_ind(X[i], grid_n), 3 + status[i]] -= 1
                cells[coord_to_ind(X[i], grid_n), status[i]] += 1

                action[i] = WalkingPlace
                X_to[i] = place
            else:
                action[i] = Choose
        else:
            if status[i] == DEFAULT_STATUS and cells[coord_to_ind(X[i], grid_n), 6] > 0:
                status[i] = SICK
                cells[coord_to_ind(X[i], grid_n), 6] -= 1

            if action[i] == WalkingPlace:
                if X[i, 0] == X_to[i, 0] and X[i, 1] == X_to[i, 1]:
                    action[i] = SitPlace if PLACE_LENGTH else WalkingHome
                    continue
                else:
                    cells[coord_to_ind(X[i], grid_n), status[i]] -= 1
                    if X[i, 0] == X_to[i, 0]:
                        X[i, 1] += 1 if X_to[i, 1] > X[i, 1] else -1
                    elif X[i, 1] == X_to[i, 1]:
                        X[i, 0] += 1 if X_to[i, 0] > X[i, 0] else -1
                    else:
                        r = random.randint(0, 1)
                        X[i, r] += 1 if X_to[i, r] > X[i, r] else -1
                    cells[coord_to_ind(X[i], grid_n), status[i]] += 1
            elif action[i] == SitPlace:
                sit[i] += 1
                if sit[i] == PLACE_LENGTH:
                    sit[i] = 0
                    action[i] = WalkingHome
            elif action[i] == WalkingHome:
                if X[i, 0] == X_home[i, 0] and X[i, 1] == X_home[i, 1]:
                    action[i] = Choose
                    cells[coord_to_ind(X[i], grid_n), status[i]] -= 1
                    cells[coord_to_ind(X[i], grid_n), 3 + status[i]] += 1
                else:
                    cells[coord_to_ind(X[i], grid_n), status[i]] -= 1
                    if X[i, 0] == X_home[i, 0]:
                        X[i, 1] += 1 if X_home[i, 1] > X[i, 1] else -1
                    elif X[i, 1] == X_home[i, 1]:
                        X[i, 0] += 1 if X_home[i, 0] > X[i, 0] else -1
                    else:
                        r = random.randint(0, 1)
                        X[i, r] += 1 if X_home[i, r] > X[i, r] else -1
                    cells[coord_to_ind(X[i], grid_n), status[i]] += 1

        if status[i] == SICK:
            sick[i] += 1
            if sick[i] == HEAL_LENGTH:
                sick[i] = 0
                status[i] = HEALTHY if random.random() > P_DEAD else DEAD_STATUS
                count[SICK] -= 1
                count[status[i]] += 1

                is_home = 3 if action[i] == Choose else 0
                cells[coord_to_ind(X[i], grid_n), is_home + SICK] -= 1
                if status[i] == HEALTHY:
                    cells[coord_to_ind(X[i], grid_n), is_home + HEALTHY] += 1
                else:
                    action[i] = DEAD_ACTION

        if action[i] != Choose:
            if status[i] == DEFAULT_STATUS:
                if random.random() > (1 - P_INFECT) ** ((cells[coord_to_ind(X[i], grid_n), SICK]) * SOCIAL_DISTANCE):
                    status[i] = SICK

                    cells[coord_to_ind(X[i], grid_n), SICK] += 1
                    cells[coord_to_ind(X[i], grid_n), DEFAULT_STATUS] -= 1

                    count[SICK] += 1
                    count[DEFAULT_STATUS] -= 1
            elif status[i] == SICK:
                count_infected = np.random.binomial(cells[coord_to_ind(X[i], grid_n), DEFAULT_STATUS],
                                                    P_INFECT * SOCIAL_DISTANCE)

                cells[coord_to_ind(X[i], grid_n), SICK] += count_infected
                cells[coord_to_ind(X[i], grid_n), DEFAULT_STATUS] -= count_infected

                count[SICK] += count_infected
                count[DEFAULT_STATUS] -= count_infected

                cells[coord_to_ind(X[i], grid_n), 6] += count_infected
        i += 1


class Cell:
    image = False
    R = None
    IMAGE_HOME = None
    IMAGE_OUTSIDE = None

    @staticmethod
    def init_image(filename, r):
        Cell.R = r
        image = pygame.image.load(filename)
        image = image.convert_alpha()
        image = pygame.transform.scale(image, (2 * r, 2 * r))

        def color_surface(surface, red, green, blue):
            arr = pygame.surfarray.pixels3d(surface)
            arr[:, :, 0] = red
            arr[:, :, 1] = green
            arr[:, :, 2] = blue

        Cell.IMAGE_OUTSIDE = [None, None, None]

        tmp = image.copy()
        color_surface(tmp, *Person.Color.SICK)
        Cell.IMAGE_OUTSIDE[Person.Status.SICK] = tmp

        tmp = image.copy()
        color_surface(tmp, *Person.Color.DEFAULT)
        Cell.IMAGE_OUTSIDE[Person.Status.DEFAULT] = tmp

        tmp = image.copy()
        color_surface(tmp, *Person.Color.HEALTHY)
        Cell.IMAGE_OUTSIDE[Person.Status.HEALTHY] = tmp

        Cell.IMAGE_HOME = [None, None, None]

        tmp = image.copy()
        color_surface(tmp, *Person.Color.HOME_SICK)
        Cell.IMAGE_HOME[Person.Status.SICK] = tmp

        tmp = image.copy()
        color_surface(tmp, *Person.Color.HOME_DEFAULT)
        Cell.IMAGE_HOME[Person.Status.DEFAULT] = tmp

        tmp = image.copy()
        color_surface(tmp, *Person.Color.HOME_HEALTHY)
        Cell.IMAGE_HOME[Person.Status.HEALTHY] = tmp

    @staticmethod
    def draw(win, cell, x, y, w, h):
        r = Cell.R
        draw_home = True
        draw_maximum = max(1, int(w * h / (np.pi * r * r)))  # ~сколько кругов помещается в клетке
        for person_status, person_color in [
            (Person.Status.SICK, Person.Color.SICK),
            (Person.Status.DEFAULT, Person.Color.DEFAULT),
            (Person.Status.HEALTHY, Person.Color.HEALTHY)
        ]:
            for _ in range(cell[person_status]):
                if draw_maximum == 0:
                    return
                if Cell.image:  # использовать спрайт в качестве модельки человека
                    win.blit(Cell.IMAGE_OUTSIDE[person_status], (random.random() * (w - 2 * r) + x,
                                                                 random.random() * (h - 2 * r) + y))
                else:  # рисовать круг в качестве модельки человека
                    pygame.draw.circle(win, person_color,
                                       (random.random() * (w - 2 * r) + x + r,
                                        random.random() * (h - 2 * r) + y + r),
                                       r)
                draw_maximum -= 1
        if draw_home:
            home_x, home_y = 1, 1
            for person_status, person_color in [
                (3 + Person.Status.SICK, Person.Color.HOME_SICK),
                (3 + Person.Status.DEFAULT, Person.Color.HOME_DEFAULT),
                (3 + Person.Status.HEALTHY, Person.Color.HOME_HEALTHY)
            ]:
                for _ in range(cell[person_status]):
                    if draw_maximum == 0:
                        return
                    if Cell.image:
                        win.blit(Cell.IMAGE_HOME[person_status], (x + (home_x - 1) * r, (home_y - 1) * r + y))
                    else:
                        pygame.draw.circle(win, person_color,
                                           (home_x * r + x, home_y * r + y),
                                           r)
                    draw_maximum -= 1
                    home_x += 2
                    if home_x * r + r > w:
                        home_x = 1
                        home_y += 2
                        if home_y * r + r > h:
                            return


class Grid:
    def __init__(self, n, W, H, x0, y0):
        self.x0 = x0
        self.y0 = y0
        self.w = W / n
        self.h = H / n

        self.grid_size = n

        self.coord = [[x, y] for x in range(n) for y in range(n)]
        self.cells = np.zeros((n * n, 7), dtype=int)

    def draw(self, win):
        for i, (x, y) in enumerate(self.coord):
            Cell.draw(win, self.cells[i], x * self.w + self.x0, y * self.h + self.y0, self.w, self.h)

    def add(self, person):
        self.cells[coord_to_ind((person.x, person.y), self.grid_size), 3 + person.status] += 1

    def draw_grid(self, win):
        grid = False
        if grid:
            n = self.cells.shape[0]
            for i in range(n):
                pygame.draw.line(win, (126, 126, 126),
                                 (self.x0 + 0, self.y0 + i * self.h),
                                 (self.x0 + n * self.w, self.y0 + i * self.h))

                pygame.draw.line(win, (126, 126, 126),
                                 (self.x0 + i * self.w, self.y0),
                                 (self.x0 + i * self.w, self.y0 + self.h * n))


class City:
    def __init__(self, center, r, name, grid_n, person_n, W, H, x0, p_begin, place_count, font, m):
        self.map = m
        self.vaccination = 0
        self.font = font
        self.center = center
        self.r = r
        self.name = name

        place_count = int(grid_n * grid_n * place_count)
        places = []
        for _ in range(place_count):
            i = random.randint(0, grid_n - 1)
            j = random.randint(0, grid_n - 1)
            places.append((i, j))

        self.grid = Grid(grid_n, W, H, x0, 0)
        self.count = np.array([0, 0, 0, 0], dtype=int)

        persons = []
        for _ in range(person_n):
            row = random.randint(0, grid_n - 1)
            column = random.randint(0, grid_n - 1)

            person = Person(row, column, Person.Status.DEFAULT if random.random() > p_begin else Person.Status.SICK)
            persons.append(person)
            self.count[person.status] += 1
            self.grid.add(person)

        persons_homes = np.array([(p.x_home, p.y_home) for p in persons])

        dist = np.abs(np.array(places)[..., None] - persons_homes.T[None]).sum(axis=1)
        eps = 1
        dist = 1 / (dist + eps)
        p = dist / dist.sum(axis=0)[None]
        for i in range(len(persons)):
            persons[i].add_places(places, p[:, i])

        random.shuffle(persons)
        self.persons = Persons(persons, self)

    def draw(self, win):
        tmp = 0

        for status, color in zip([Person.Status.DEFAULT, Person.Status.SICK, Person.Status.HEALTHY, Person.Status.DEAD],
                                 [Person.Color.DEFAULT, Person.Color.SICK, Person.Color.HEALTHY, Person.Color.DEAD]):
            self.draw_arc(win, color,
                          int(tmp / len(self.persons.action) * 360),
                          int((tmp + self.count[status]) / len(self.persons.action) * 360))
            tmp += self.count[status]

        if self.count[Person.Status.SICK] > len(self.persons.action) * 0.45:
            pygame.draw.circle(win, (255, 255, 255), self.center, self.r + 3, width=3)

    def draw_arc(self, win, color, start, end):
        p = [self.center]

        for n in range(start, end):
            x = self.center[0] + int(self.r * math.cos(n * math.pi / 180))
            y = self.center[1] + int(self.r * math.sin(n * math.pi / 180))
            p.append((x, y))
        p.append(self.center)
        if len(p) > 2:
            pygame.draw.polygon(win, color, p)

    def draw_grid(self, win):
        self.grid.draw_grid(win)

    def draw_people(self, win):
        self.grid.draw(win)

    def draw_info(self, win):
        text_color = (165, 10, 255)
        text_size = Map.TEXTSIZE
        win.fill((0, 0, 0), (0, 0, self.map.RESERVED_W, text_size * 8 + 7))

        self.font.render_to(win, (0, 0), self.name, text_color)
        # self.font.render_to(win, (0, 0), '' + str(self.grid.cells.sum(axis=0)), text_color)
        self.font.render_to(win, (0, text_size), 'Численность: ' + str(len(self.persons.action)), text_color)
        self.font.render_to(win, (0, text_size * 2),
                            'Инфицированныe: ' + str(self.count[Person.Status.SICK]), Person.Color.SICK)
        self.font.render_to(win, (0, text_size * 3),
                            'Выздоровевшие: ' + str(self.count[Person.Status.HEALTHY]), Person.Color.HEALTHY)
        self.font.render_to(win, (0, text_size * 4), 'Здоровыe: ' + str(self.count[Person.Status.DEFAULT]),
                            Person.Color.DEFAULT)
        self.font.render_to(win, (0, text_size * 5), 'Мёртвые: ' + str(self.count[Person.Status.DEAD]),
                            Person.Color.DEAD)
        self.font.render_to(win, (0, text_size * 7 + 7), f'Деньги: {self.map.money:.2f}', text_color)
        self.font.render_to(win, (0, text_size * 8 + 7), f'Цена прививки: {self.map.cost:.2f}', text_color)

        # pygame.display.update((0, 0, self.m.RESERVED_W, text_size * 8 + 7))

    def check_click(self, p):
        return distance(self.center, p) < self.r

    def step(self):
        self.persons.step()

    def vaccinate(self):
        self.count[Person.Status.DEFAULT] -= self.vaccination
        self.count[Person.Status.HEALTHY] += self.vaccination
        i = 0
        while self.vaccination != 0:
            if self.persons.status[i] == Person.Status.DEFAULT:
                self.persons.status[i] = Person.Status.HEALTHY
                self.vaccination -= 1

                is_home = 3 if self.persons.action[i] == Person.Action.Choose else 0
                self.grid.cells[coord_to_ind(self.persons.X[i], self.grid.grid_size), is_home + Person.Status.DEFAULT] -= 1
                self.grid.cells[coord_to_ind(self.persons.X[i], self.grid.grid_size), is_home + Person.Status.HEALTHY] += 1

            i += 1


# Press the green button in the gutter to run the script.
def mainLoop():
    pygame.init()

    m = Map('Map.txt', 'OpenSans-LightItalic.ttf')
    Cell.init_image('Lol_circle.png', 2)

    clock = pygame.time.Clock()
    n_fps = 20
    sum_fps = np.zeros(n_fps)
    i = 0
    while True:
        sum_fps[i] = int(clock.get_fps())
        i = (i + 1) % n_fps
        # print(sum_fps.mean())
        clock.tick(60)
        for event in pygame.event.get():
            m.handle_event(event)
        m.step()


if __name__ == '__main__':
    mainLoop()
