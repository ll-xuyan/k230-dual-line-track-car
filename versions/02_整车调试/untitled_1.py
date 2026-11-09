import time
from media.sensor import *
from media.display import *
from media.media import *

image_shape = [480, 640]
sensor = Sensor(id=2, width=1280, height=960, fps=30)
sensor.reset()
sensor.set_framesize(width=image_shape[1], height=image_shape[0])
sensor.set_pixformat(Sensor.RGB888)
Display.init(Display.ST7701, width=image_shape[1], height=image_shape[0], to_ide=True, quality=50)
MediaManager.init()
sensor.run()
clock = time.clock()

while True:
    clock.tick()
    img = sensor.snapshot()
    Display.show_image(img)
    print("FPS:", clock.fps())
