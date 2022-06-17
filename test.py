import numpy as np
from utils.dobot import *

db = Dobot(port='/dev/ttyUSB0')
db.setOrigin()

x = 0; y = 0; z= 0;
xmove = 4; ymove = 4; zmove = 10;

for j in range(5):
    for i in range(5):
        print(x,y,z)
        db.move([x,y,z])
        db.move([x,y,z-zmove])
        db.move([x,y,z])
        x += xmove
    xmove *= -1
    x += xmove
    y += ymove