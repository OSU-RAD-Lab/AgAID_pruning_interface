from Branch import Branch
from Vec3 import Vec3
from ObjWriter import ObjWriter
import os

if __name__ == '__main__':
    try: os.mkdir(f"output")
    except: pass
    for z in range(1):
        try: os.mkdir(f"output/{z}")
        except: pass
        for x in range(20):
            Branch.fromValue(
                value =     x/19,
                direction = Vec3.random(z),
                seed =      z + 1
            ).writeEverything(f"output/{z}/{x}")