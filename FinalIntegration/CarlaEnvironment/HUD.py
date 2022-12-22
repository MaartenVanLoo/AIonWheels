import logging
import math
import os
import datetime
import time

import cv2
import numpy as np
import pygame
from pygame.locals import KMOD_CTRL
from pygame.locals import K_ESCAPE
from pygame.locals import K_q, K_f, K_F12, K_r

from .CarlaWorld import CarlaWorld
import carla

RED = (255,0,0)

class HUD(object):
    def __init__(self, width, height, options=pygame.HWSURFACE | pygame.DOUBLEBUF | pygame.RESIZABLE) -> None:
        super().__init__()
        #dynamic scaling parameters
        self.dim = (width, height)
        self.margin = (10, 10)  # horizontal, vertical
        self.spacing = 20
        self.info_width = 250
        self.frame_dim = (550, 500)
        self.frame_content_dim = (500, 380)
        self.width = width
        self.height = height

        pygame.init()
        pygame.font.init()
        pygame.display.set_caption("AIonWheels HUD")

        self.icon_path = os.path.realpath(os.path.dirname(__file__)) + '/HUD.ico'
        self.background_path = os.path.realpath(os.path.dirname(__file__)) + "/background1.jpg"
        self.frame_path = os.path.realpath(os.path.dirname(__file__))+ "/HUD_frame2.png"

        if os.path.exists(self.icon_path) and os.path.isfile(self.icon_path):
            pygame.display.set_icon(pygame.image.load(self.icon_path))
        if os.path.exists(self.background_path) and os.path.isfile(self.background_path):
            self.background = pygame.transform.scale(pygame.image.load(self.background_path), (width, height))
        else:
            self.background = None

        if os.path.exists(self.frame_path) and os.path.isfile(self.frame_path):
            self.image_frame = pygame.transform.scale(pygame.image.load(self.frame_path),
                                                      (self.frame_dim[0],self.frame_dim[1]))
        else:
            self.image_frame = None

        self.id = -1



        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        font_name = 'courier' if os.name == 'nt' else 'mono'
        fonts = [x for x in pygame.font.get_fonts() if font_name in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 12 if os.name == 'nt' else 14)
        self._notifications = FadingText(font, (width, 40), (0, height - 40), color = (0,0,0,0))
        self._warnings = FadingText(font, (width, 40), (0, height - 40), color = (201, 34, 34, 0))

        self.server_fps = 0
        self.frame = 0
        self.simulation_time = 0
        self._server_clock = pygame.time.Clock()

        self.clock = pygame.time.Clock()
        self.controller = KeyboardControl(self)
        self.display = pygame.display.set_mode(
            (width, height),
            options)
        self.options = options

        self.logger = logging.getLogger("AIonWheels")
        self.carlaWorld = None

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

    def _setBackground(self):
        if self.background:
            self.display.blit(self.background, (0, 0))

    def render(self, carlaWorld: CarlaWorld):
        start = time.time()
        self._setBackground()
        self._tick(carlaWorld.world)
        self._compute_content(carlaWorld)
        self._render_info(carlaWorld)       #compute and render info

        self._render_detection(carlaWorld)  # Camera view
        self._render_lidar(carlaWorld)      # lidar view

        self._collisionMessage(carlaWorld)  #popup messages on collision

        #udpate fading
        self._warnings.tick()
        self._warnings.render(self.display)         # won't render without warning
        self._notifications.tick()
        self._notifications.render(self.display)    # won't render without notification
        pygame.display.flip()
        stop = time.time()
        print(f"HUD render time:\t\t\t{(stop - start) * 1000:3.0f} ms")
        pass

    def _get_info(self, carlaWorld):
        vehicle = carlaWorld.getPlayer().getVehicle()
        transform = vehicle.get_transform()
        vel = vehicle.get_velocity()
        control = vehicle.get_control()
        heading = 'N' if abs(transform.rotation.yaw) < 89.5 else ''
        heading += 'S' if abs(transform.rotation.yaw) > 90.5 else ''
        heading += 'E' if 179.5 > transform.rotation.yaw > 0.5 else ''
        heading += 'W' if -0.5 > transform.rotation.yaw > -179.5 else ''

        self._info_text = [
            'Server:     % 16.0f FPS' % self.server_fps,
            'Client:     % 16.0f FPS' % self.clock.get_fps(),
            '',
            'Vehicle:    % 20s' % HUD._get_actor_display_name(vehicle, truncate=20),
            'Map:        % 20s' % carlaWorld.map.name.split('/')[-1],
            'Simulation time: % 15s' % datetime.timedelta(seconds=int(self.simulation_time)),
            '',
            'Loaded models:',
            'RL model    % 20s' % carlaWorld.rl_module.getModelName(),
            'DL lidar    % 20s' % carlaWorld.dl_lidar.getModelName(),
            'DL classify % 20s' % carlaWorld.dl_recognition.getModelName(),
            '',
            'Speed:       % 14.0f km/h' % (3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)),
            'Speed limit: % 14.0f km/h' % (carlaWorld.getPlayer().getSpeedLimit()*3.6),
            'Target speed:% 14.0f km/h' % (carlaWorld.getPlayer().getTargetSpeed()*3.6),
            'Distance:    % 17.2f m'  % (carlaWorld.rl_module.frames[-1][0]),
            '',
            u'Heading:   % 16.0f\N{DEGREE SIGN} % 2s' % (transform.rotation.yaw, heading),
            'Location:   % 20s' % ('(% 5.1f, % 5.1f)' % (transform.location.x, transform.location.y)),
            'Height:     % 18.0f m' % transform.location.z,
            '']

        if isinstance(control, carla.VehicleControl):
            self._info_text += [
                ('Throttle:', control.throttle, 0.0, 1.0),
                ('Steer:', control.steer, -1.0, 1.0),
                ('Brake:', control.brake, 0.0, 1.0),
            ]

        self._info_performance =[

        ]

    def _render_info(self,carlaWorld):
        max_info_height = min(500, self.dim[1]- self.margin[1])
        self._get_info(carlaWorld)
        info_surface = pygame.Surface((self.info_width, max_info_height))
        info_surface.set_alpha(50)
        self.display.blit(info_surface,(0,0))

        v_offset = 4
        bar_h_offset = 100
        bar_width = 106

        for item in self._info_text:
            if v_offset  + 18 > max_info_height:
                break
            if isinstance(item, list):
                if len(item) > 1:
                    points = [(x + 8, v_offset + 8 + (1 - y) * 30) for x, y in enumerate(item)]
                    pygame.draw.lines(self.display, (255, 136, 0), False, points, 2)
                item = None
                v_offset += 18
            elif isinstance(item, tuple):
                if isinstance(item[1], bool):
                    rect = pygame.Rect((bar_h_offset, v_offset + 8), (6, 6))
                    pygame.draw.rect(self.display, (255, 255, 255), rect, 0 if item[1] else 1)
                else:
                    rect_border = pygame.Rect((bar_h_offset, v_offset + 8), (bar_width, 6))
                    pygame.draw.rect(self.display, (255, 255, 255), rect_border, 1)
                    fig = (item[1] - item[2]) / (item[3] - item[2])
                    if item[2] < 0.0:
                        rect = pygame.Rect(
                            (bar_h_offset + fig * (bar_width - 6), v_offset + 8), (6, 6))
                    else:
                        rect = pygame.Rect((bar_h_offset, v_offset + 8), (fig * bar_width, 6))
                    pygame.draw.rect(self.display, (255, 255, 255), rect)
                item = item[0]
            if item:  # At this point has to be a str.
                surface = self._font_mono.render(item, True, (255, 255, 255))
                self.display.blit(surface, (8, v_offset))
            v_offset += 18

    def _render_detection(self, carlaWorld):
        sensor = carlaWorld.get_sensor("Camera")
        pos = (
            self.info_width + self.spacing + self.image_frame.get_width()/2,
            self.margin[1] + self.image_frame.get_height()/2
        )
        image_rect = self.image_frame.get_rect(center=pos)
        self.display.blit(self.image_frame, image_rect)
        if sensor is None:
            self.__noDataImage("No camera found",center=pos)
            return  #no sensor found
        image = sensor.getState()
        if image is None:
            self.__noDataImage("No camera data found",center=pos)
            return  # no image found
        #scale image:
        image = image_resize(image,width=self.frame_content_dim[0], height=self.frame_content_dim[1])
        surface = pygame.surfarray.make_surface(image.swapaxes(0, 1))
        surface_rect = surface.get_rect(center=pos)
        self.display.blit(surface, surface_rect)
        pass

    def _render_lidar(self, carlaWorld):
        pos = (
            self.info_width + self.spacing *2 + self.image_frame.get_width() * 3 / 2,
            self.margin[1] + self.image_frame.get_height() / 2
        )

        image_rect = self.image_frame.get_rect(center=pos)
        self.display.blit(self.image_frame, image_rect)
        sensor = carlaWorld.get_sensor("Lidar")
        if sensor is None:
            self.__noDataImage("No lidar found",center=pos)
            return  # no sensor found
        bev_image = carlaWorld.dl_lidar.bev_image
        if bev_image is None:
            self.__noDataImage("No lidar data found",center=pos)
            return  # no image found
        bev_image = image_resize(bev_image, width=self.frame_content_dim[0], height=self.frame_content_dim[0])
        clipping = int((self.frame_content_dim[0] - self.frame_content_dim[1])/2)
        if clipping > 0:
            bev_image = bev_image[:,clipping:-clipping,:] #equal to size of "camera"
        surface = pygame.surfarray.make_surface(bev_image[:,:,::-1])    # note R & B channels must be swapped ,
                                                                        # achieved by reversing BGR to RGB
        image_rect = surface.get_rect(center=pos)
        self.display.blit(surface, image_rect)
        pass

    def __noDataImage(self,text="No data available", center = None):
        if center is None:
            center = (self.width/2, self.height/2)

        font = pygame.font.SysFont(None, 36)
        image = font.render(text, True, RED)
        image_rect = image.get_rect(center=center)
        self.display.blit(image, image_rect)

    def _collisionMessage(self, carlaWorld):
        sensor = carlaWorld.get_sensor("CollisionSensor")
        if sensor is None:
            return # no sensor found
        state = sensor.getState()
        if state is None:
            return
        last_collision, message = state

        if last_collision == self.frame:
            self.warning(message, frames = 40)




    def notification(self, text, frames=40.0):
        """Notification text"""
        self._notifications.set_text(text, frames=frames)

    def warning(self, text, frames = 40.0):
        self._warnings.set_text(text, frames=frames)
    @staticmethod
    def _get_actor_display_name(actor, truncate=250):
        """Method to get actor display name"""
        name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
        return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name



# ==============================================================================
# -- FadingText ----------------------------------------------------------------
# ==============================================================================


class FadingText(object):
    """ Class for fading text """

    def __init__(self, font, dim, pos, color):
        """Constructor method"""
        self.font = font
        self.dim = dim
        self.pos = pos
        self.frames_left = 0
        self.surface = pygame.Surface(self.dim)
        self.color = color

    def set_text(self, text, color=(255, 255, 255), frames=40.0):
        """Set fading text"""
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.frames_left = frames
        self.surface.fill(self.color)
        self.surface.blit(text_texture, (10, 11))

    def tick(self):
        """Fading text method for every tick"""
        self.frames_left = max(0.0, self.frames_left - 1)
        if self.frames_left > 0:
            self.surface.set_alpha(int(max(300.0 * ((self.frames_left - 1) / 40), 0.0)))

    def render(self, display):
        """Render fading text method"""
        if self.frames_left == 0:
            return
        display.blit(self.surface, self.pos)

# ==============================================================================
# -- KeyboardControl -----------------------------------------------------------
# ==============================================================================

class KeyboardControl(object):
    def __init__(self, hud: HUD):
        #hud.notification("Press 'H' or '?' for help.", seconds=4.0)
        self.hud = hud
        pass

    def parse_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise KeyboardInterrupt()

            if event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    raise KeyboardInterrupt()
                if self._is_enter_full_Screen(event.key):
                    self._enter_fullscreen()
                    continue
                if self._is_exit_full_screen(event.key):
                    self._exit_fullscreen()
                    continue
                if event.key == K_r:
                    if self.hud.carlaWorld is None:
                        return
                    self.hud.carlaWorld.reset()

            if event.type == pygame.VIDEORESIZE:
                w = event.w
                h = event.h
                self.hud.display = pygame.display.set_mode((w,h), self.hud.options)
                #rescale background
                if os.path.exists(self.hud.background_path) and os.path.isfile(self.hud.background_path):
                    self.hud.background = pygame.transform.scale(pygame.image.load(self.hud.background_path), (w, h))
                else:
                    self.hud.background = None

    def _enter_fullscreen(self):
        if self.hud.options & pygame.FULLSCREEN:
            print("Already in fullscreen")
            return
        info = pygame.display.Info()  # You have to call this before pygame.display.set_mode()
        screen_width, screen_height = info.current_w, info.current_h
        self.hud.options ^= pygame.FULLSCREEN
        self.hud.options ^= pygame.RESIZABLE
        self.hud.display = self._toggle_fullscreen(screen_width,screen_height)

    def _exit_fullscreen(self):
        if not (self.hud.options & pygame.FULLSCREEN):
            print("Not in fullscreen")
            return
        self.hud.options ^= pygame.FULLSCREEN
        self.hud.options ^= pygame.RESIZABLE
        self.hud.display = self._toggle_fullscreen(self.hud.width, self.hud.height)
        pass

    def _toggle_fullscreen(self, width, height):
        #https://www.pygame.org/wiki/toggle_fullscreen?parent=CookBook
        screen = self.hud.display
        tmp = screen.convert()
        caption = pygame.display.get_caption()
        cursor = pygame.mouse.get_cursor()  # Duoas 16-04-2007

        w, h = width, height
        bits = screen.get_bitsize()

        pygame.display.quit()
        pygame.display.init()

        screen = pygame.display.set_mode((w, h), self.hud.options, bits)
        screen.blit(tmp, (0, 0))
        pygame.display.set_caption(*caption)

        pygame.key.set_mods(0)  # HACK: work-a-round for a SDL bug??

        pygame.mouse.set_cursor(*cursor)  # Duoas 16-04-2007

        #rescale background
        if os.path.exists(self.hud.background_path) and os.path.isfile(self.hud.background_path):
            self.hud.background = pygame.transform.scale(pygame.image.load(self.hud.background_path), (w, h))
        else:
            self.hud.background = None


        return screen

    @staticmethod
    def _is_quit_shortcut(key):
        """Shortcut for quitting"""
        return (key == K_q and pygame.key.get_mods() & KMOD_CTRL)

    @staticmethod
    def _is_exit_full_screen(key):
        return (key == K_ESCAPE)

    @staticmethod
    def _is_enter_full_Screen(key):
        return (key == K_F12 or key == K_f)

# ==============================================================================
# -- Utility functions ---------------------------------------------------------
# ==============================================================================

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA, addBorder = False):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    top = 0
    down = 0
    left = 0
    right = 0
    h1 = width * (h / w)
    w2 = height * (h / w)

    if (h1 < height):
        #only vertical borders
        top = int((height - h1) / 2);
        down = top
        image = cv2.resize(image, (width, int(h1)), interpolation=inter)
    else:
        #only horizontal borders
        left = int((width - w2) / 2);
        right = left
        image = cv2.resize(image, (int(w2), height), interpolation=inter)

    if addBorder:
        image = cv2.copyMakeBorder(image, top, down, left, right, cv2.BORDER_CONSTANT, None, value = 0)
    # return the resized image
    return image