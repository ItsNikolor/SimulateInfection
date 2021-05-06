import math
import random
import sys

import numpy as np
from numba import njit
from numpy.random import choice
import pygame
import pygame.freetype
from pygame.locals import *

#Окно для ввода числа прививок
class InputBox:
    COLOR_INACTIVE = pygame.Color('lightskyblue3')
    COLOR_ACTIVE = pygame.Color('dodgerblue2')

    #Конструктор
    def __init__(self, x, y, w, h, font, map, text=''):
        pass

    #Ассоциировать окно с городом
    def assign_city(self, city):
        pass

    #Обработать событие
    def handle_event(self, event):
        pass

    #Нарисовать окно
    def draw(self, screen):
        pass


#Карта на которой всё отображается
class Map:
    SPEED = 2
    P_W = 0.3
    TEXTSIZE = 20
    P_H = TEXTSIZE * 9 + 5

    p_dead = None
    p_infect = None
    self_isolation = None
    social_distance = None

    BAD_ROUND = False

    #Конструктор
    def __init__(self, filename, font):
        pass

    #Нарисовать карту
    def draw(self):
        pass

    #Проверить клик
    def check_click(self, pos):
        pass

    #Шаг симуляции
    def step(self):
        pass

    #Начать симуляцию
    def start_simulation(self):
        pass

    #Обработать событие
    def handle_event(self, event):
        pass


#Личность
class Person:
    HOME_LENGTH = None
    PLACE_LENGTH = None
    HEAL_LENGTH = None

    #Enum цвет
    class Color:
        DEFAULT = (0, 255, 255)
        SICK = (255, 0, 0)
        HEALTHY = (0, 255, 0)
        HOME_DEFAULT = (165, 10, 255)
        HOME_SICK = (255, 255, 0)
        HOME_HEALTHY = (125, 125, 125)
        DEAD = (255, 255, 255)

    #Enum статус
    class Status:
        DEFAULT = 0
        SICK = 1
        HEALTHY = 2
        DEAD = 3

    #Enum действие
    class Action:
        SitHome = 0
        WalkingPlace = 1
        SitPlace = 2
        WalkingHome = 3
        Choose = 4
        DEAD = 5

    #Конструктор
    def __init__(self, x_home, y_home, status):
        pass

    #Добавить места
    def add_places(self, places, prob):
        pass


#Много личностей
class Persons:
    #Конструктор
    def __init__(self, persons, city):
        pass

    #Сделать шаг
    def step(self):
        pass


#Клетка
class Cell:
    image = False
    R = None
    IMAGE_HOME = None
    IMAGE_OUTSIDE = None

    #Инициализировать картинку для личности
    @staticmethod
    def init_image(filename, r):
        pass

    #Нарисовать людей в клетке
    @staticmethod
    def draw(win, cell, x, y, w, h):
        pass


#Клетчатая сетка
class Grid:
    #Конструктор
    def __init__(self, n, W, H, x0, y0):
        pass

    #Нарисовать сетку
    def draw(self, win):
        pass

    #Добавить личность на сетку
    def add(self, person):
        pass


#Город
class City:
    #Конструктор
    def __init__(self, center, r, name, grid_n, person_n, W, H, x0, p_begin, place_count, font, map):
        pass

    #Нарисовать город (в виде круга)
    def draw(self, win):
        pass

    #Нарисовать сегмент круга
    def draw_arc(self, win, color, start, end):
        pass

    #Нарисовать людей в городе
    def draw_people(self, win):
        pass

    #Нарисовать информацию о городе
    def draw_info(self, win):
        pass

    #Проверить был ли клик на город
    def check_click(self, pos):
        pass

    #Шаг симуляции
    def step(self):
        pass

    #Провести вакцинацию
    def vaccinate(self):
        pass


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
