import os

import cv2
import pygame
from pygame.locals import KMOD_CTRL
from pygame.locals import K_ESCAPE
from pygame.locals import K_q

from .CarlaWorld import CarlaWorld

RED = (255,0,0)

class HUD(object):
    def __init__(self, width, height, options=pygame.HWSURFACE | pygame.DOUBLEBUF) -> None:
        super().__init__()
        pygame.init()
        pygame.font.init()
        self.id = -1
        self.dim = (width, height)
        self.width = width
        self.height = height

        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        font_name = 'courier' if os.name == 'nt' else 'mono'
        fonts = [x for x in pygame.font.get_fonts() if font_name in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 12 if os.name == 'nt' else 14)
        self._notifications = FadingText(font, (width, 40), (0, height - 40))

        self.server_fps = 0
        self.frame = 0
        self.simulation_time = 0
        self._server_clock = pygame.time.Clock()

        self.clock = pygame.time.Clock()
        self.controller = KeyboardControl(self)
        self.display = pygame.display.set_mode(
            (width, height),
            options)


    def on_world_tick(self, timestamp):
        """Gets informations from the world at every tick"""
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame = timestamp.frame_count
        self.simulation_time = timestamp.elapsed_seconds

    def _tick(self, world) -> None:
        self.clock.tick()
        self.controller.parse_events()
        pass

    def _step(self, world) -> None:
        self._tick(world)

    def _compute_content(self, carlaWorld):
        pass

    def render(self, carlaWorld: CarlaWorld):
        self._tick(carlaWorld.world)
        self._compute_content(carlaWorld)
        self._render_info(carlaWorld)
        self._render_detection(carlaWorld)
        self._render_lidar(carlaWorld)
        pygame.display.flip()
        pass

    def _render_info(self, carlaWorld):
        pass

    def _render_detection(self, carlaWorld):
        sensor = carlaWorld.get_sensor("Camera")
        if sensor is None:
            image = self.__noDataImage("No camera found")
            image_rect = image.get_rect(center=(250, self.height / 2))
            self.display.blit(image, image_rect)
            return  #no sensor found
        image = sensor.getState()
        if image is None:
            image = self.__noDataImage("No camera found")
            image_rect = image.get_rect(center=(250, self.height / 2))
            self.display.blit(image, image_rect)
            return  # no image found
        #scale image:
        image = image_resize(image,width=500, height=500)
        surface = pygame.surfarray.make_surface(image.swapaxes(0, 1))
        surface_rect = surface.get_rect(center=(250, self.height / 2))
        self.display.blit(surface, surface_rect)
        pass

    def _render_lidar(self, carlaWorld):
        #image = carla.World.get
        sensor = carlaWorld.get_sensor("Camera")
        if sensor is None:
            image = self.__noDataImage("No lidar data found")
            image_rect = image.get_rect(center=(750, self.height / 2))
            self.display.blit(image, image_rect)
            return  # no sensor found
        image = sensor.getState()
        if image is None:
            image = self.__noDataImage("No lidar data found")
            image_rect = image.get_rect(center=(750, self.height / 2))
            self.display.blit(image, image_rect)
            return  # no image found
        image_rect = image.get_rect(center=(750, self.height / 2))
        self.display.blit(image, image_rect)
        pass

    def __noDataImage(self,text="No data available"):
        font = pygame.font.SysFont(None, 48)
        image = font.render(text, True, RED)
        return image


# ==============================================================================
# -- FadingText ----------------------------------------------------------------
# ==============================================================================


class FadingText(object):
    """ Class for fading text """

    def __init__(self, font, dim, pos):
        """Constructor method"""
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        """Set fading text"""
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, _, clock):
        """Fading text method for every tick"""
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)

    def render(self, display):
        """Render fading text method"""
        display.blit(self.surface, self.pos)

# ==============================================================================
# -- KeyboardControl -----------------------------------------------------------
# ==============================================================================

class KeyboardControl(object):
    def __init__(self, hud):
        #hud.notification("Press 'H' or '?' for help.", seconds=4.0)
        pass

    def parse_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise KeyboardInterrupt()
                return True
            if event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True

    @staticmethod
    def _is_quit_shortcut(key):
        """Shortcut for quitting"""
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)


# ==============================================================================
# -- Utility functions ---------------------------------------------------------
# ==============================================================================

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized