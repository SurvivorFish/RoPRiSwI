import tkinter as tk
from tkinter import ALL
import math


class Point:
    def __init__(self, *arguments):
        self.x = arguments[0]
        self.y = arguments[1]
        self.z = arguments[2]

    def __eq__(self, other):
        return (self.x == other.x and self.y == other.y)

    def __ne__(self, other):
        return (self.x != other.x and self.y != other.y)

    def polar(self):
        ans = math.atan2(self.y, self.x)
        if ans < 0:
            return ans + 2 * math.pi
        return ans

    def deg(self):
        return self.polar() * 180 / math.pi

    def rotatez(self, AN):
        x1 = self.x
        y1 = self.y
        self.x, self.y = math.cos(AN) * x1 - math.sin(AN) * y1, math.sin(AN) * x1 + math.cos(AN) * y1

    def rotatey(self, AN):
        x1 = self.x
        z1 = self.z
        self.x, self.z = math.cos(AN) * x1 - math.sin(AN) * z1, math.sin(AN) * x1 + math.cos(AN) * z1

    def rotatex(self, AN):
        y1 = self.y
        z1 = self.z
        self.y, self.z = math.cos(AN) * y1 - math.sin(AN) * z1, math.sin(AN) * y1 + math.cos(AN) * z1


class Vector:
    def __init__(self, *arguments):
        self.x = arguments[0]
        self.y = arguments[1]
        self.z = arguments[2]

    def __add__(self, Vector):
        return Vector(self.x + Vector.x, self.y + Vector.y, self.z + Vector.z)

    def __sub__(self, other):
        return Vector(self.x - other.x, self.y - other.y, self.z - other.z)

    def __neg__(self):
        return Vector(-self.x, -self.y, -self.z)

    def __mul__(self, digit):
        return Vector(self.x * digit, self.y * digit, self.z * digit)

    def length(self):
        return ((self.x) ** 2 + (self.y) ** 2 + (self.z) ** 2) ** 0.5

    def scalar(self, Point):
        return (self.x * Point.x + self.y * Point.y + self.z * Point.z)

    def vector_mult(self, Point):
        return Vector(self.y * Point.z - self.z * Point.y, self.z * Point.x - self.x * Point.z, self.x * Point.y - self.y * Point.x)


class Face:
    def __init__(self, order):
        self.order = order[:]

    def rotate(self):
        for point in self.order:
            point.rotatex(alpha_x)
            point.rotatey(alpha_y)
            point.rotatez(alpha_z)

    def normal(self):
        A, B, C = self.order[0], self.order[1], self.order[2]
        a = Vector(B.x - A.x, B.y - A.y, B.z - A.z)
        b = Vector(C.x - B.x, C.y - B.y, C.z - B.z)
        return (-a).vector_mult(b)

    def visible(self, normal):
        return Vector(0, 0, 1).scalar(self.normal()) >= 0

    def light(self):
        return Vector(0, 0, 1).scalar(self.normal()) / (self.normal().length())


class Figure:
    def __init__(self, faces):
        self.faces = faces[:]

    def rotate(self):
        for face in self.faces:
            face.rotate()


def rgb_to_string(rgb):
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


alpha_x = 0.01
alpha_y = 0.02
alpha_z = 0.03

def create_tetraedr():
    a = math.sqrt(2) / 3
    b = math.sqrt(2 / 3)
    P0 = Point(0, 0, 1)
    P1 = Point(0, 2 * a, -1/3)
    P2 = Point(-b, -a, -1/3)
    P3 = Point(b, -a, -1/3)

    F1 = Face([P0, P2, P3])
    F2 = Face([P0, P3, P1])
    F3 = Face([P0, P1, P2])
    F4 = Face([P3, P2, P1])

    return Figure([F1, F2, F3, F4])


def create_octaedron():
    P0 = Point(0, 1, 0)
    P1 = Point(-1, 0, 0)
    P2 = Point(0, -1, 0)
    P3 = Point(1, 0, 0)
    P4 = Point(0, 0, -1)
    P5 = Point(0, 0, 1)

    F1 = Face([P0, P1, P5])
    F2 = Face([P1, P2, P5])
    F3 = Face([P2, P3, P5])
    F4 = Face([P3, P0, P5])
    F5 = Face([P1, P0, P4])
    F6 = Face([P2, P1, P4])
    F7 = Face([P3, P2, P4])
    F8 = Face([P0, P3, P4])

    return Figure([F1, F2, F3, F4, F5, F6, F7, F8])


def create_cube():
    phi = (1 + math.sqrt(5)) / 2
    aa = 1/(3)**0.5

    P0 = Point(-aa, -aa, -aa)
    P1 = Point(aa, -aa, -aa)
    P2 = Point(aa, aa, -aa)
    P3 = Point(-aa, aa, -aa)
    P4 = Point(-aa, -aa, aa)
    P5 = Point(aa, -aa, aa)
    P6 = Point(aa, aa, aa)
    P7 = Point(-aa, aa, aa)

    F1 = Face([P1, P5, P4, P0])
    F2 = Face([P6, P7, P4, P5])
    F3 = Face([P2, P6, P5, P1])
    F4 = Face([P7, P6, P2, P3])
    F5 = Face([P4, P7, P3, P0])
    F6 = Face([P0, P3, P2, P1])

    polyhedron = Figure([F1, F2, F3, F4, F5, F6])
    return polyhedron


def create_dodecahedron():
    aa = ((5)**0.5 +1)*(5+(5)**0.5)**0.5/(2* 30**0.5)
    bb = (2*(5-(5)**0.5)/15)**0.5
    cc = ((5)**0.5-2)*aa
    dd = 1/(3)**0.5
    ee = (2*(5+(5)**0.5)/15)**0.5
    gg = bb/((5)**0.5-1)
    hh = ((5)**0.5-1)/(2*(3)**0.5)
    jj = ee/((5)**0.5+1)
    ii = ((5)**0.5+1)/(2*(3)**0.5)

    P0 = Point(0, -bb, -aa)
    P1 = Point(dd, -cc, -aa)
    P2 = Point(hh, gg, -aa)
    P3 = Point(-hh, gg, -aa)
    P4 = Point(-dd, -cc, -aa)
    P5 = Point(0, -ee, -cc)
    P6 = Point(ii, -jj, -cc)
    P7 = Point(dd, aa, -cc)
    P8 = Point(-dd, aa, -cc)
    P9 = Point(-ii, -jj, -cc)
    P10 = Point(-dd, -aa, cc)
    P11 = Point(dd, -aa, cc)
    P12 = Point(ii, jj, cc)
    P13 = Point(0, ee, cc)
    P14 = Point(-ii, jj, cc)
    P15 = Point(-hh, -gg, aa)
    P16 = Point(hh, -gg, aa)
    P17 = Point(dd, cc, aa)
    P18 = Point(0, bb, aa)
    P19 = Point(-dd, cc, aa)

    F1 = Face([P0, P4, P3, P2, P1])
    F2 = Face([P11, P5, P0, P1, P6])
    F3 = Face([P12, P6, P1, P2, P7])
    F4 = Face([P13, P7, P2, P3, P8])
    F5 = Face([P14, P8, P3, P4, P9])
    F6 = Face([P10, P9, P4, P0, P5])
    F7 = Face([P16, P11, P6, P12, P17])
    F8 = Face([P17, P12, P7, P13, P18])
    F9 = Face([P18, P13, P8, P14, P19])
    F10 = Face([P19, P14, P9, P10, P15])
    F11 = Face([P15, P10, P5, P11, P16])
    F12 = Face([P15, P16, P17, P18, P19])
    
    return Figure([F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, F11, F12])


def create_icosahedron():
    a = 1 / math.sqrt(5)
    b = (5 + math.sqrt(5)) / 10
    h = (5- math.sqrt(5)) / 10
    d = math.sqrt(h)
    e = math.sqrt(b)
    
    P0 = Point(0, 0, -1)
    P1 = Point(0, 2*a, -a)
    P2 = Point(-e, h, -a)
    P3 = Point(-d, -b, -a)
    P4 = Point(d, -b, -a)
    P5 = Point(e, h, -a)
    P6 = Point(d, b, a)
    P7 = Point(-d, b, a)
    P8 = Point(-e, -h, a)
    P9 = Point(0, -2 *a, a)
    P10 = Point(e, -h, a)
    P11 = Point(0, 0, 1)

    F1 = Face([P0, P1, P5])
    F2 = Face([P0, P2, P1])
    F3 = Face([P0, P3, P2])
    F4 = Face([P0, P4, P3])
    F5 = Face([P0, P5, P4])
    F6 = Face([P5, P1, P6])
    F7 = Face([P1, P2, P7])
    F8 = Face([P2, P3, P8])
    F9 = Face([P3, P4, P9])
    F10 = Face([P4, P5, P10])
    F11 = Face([P1, P7, P6])
    F12 = Face([P2, P8, P7])
    F13 = Face([P3, P9, P8])
    F14 = Face([P4, P10, P9])
    F15 = Face([P5, P6, P10])
    F16 = Face([P11, P8, P9])
    F17 = Face([P11, P9, P10])
    F18 = Face([P11, P10, P6])
    F19 = Face([P11, P6, P7])
    F20 = Face([P11, P7, P8])
    
    return Figure([F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, F11, F12, F13, F14, F15, F16, F17, F18, F19, F20])
	

polyhedron = create_tetraedr()
move_id = None
is_rotating = True


def move():
    global move_id
    w.delete(ALL)
    polyhedron.rotate()
    for face in polyhedron.faces:
        if face.visible(face):
            coords = []
            for p in face.order:
                coords += [p.x, p.y]
            w.create_polygon(
                [90 * item + 250 for item in coords],
                fill=rgb_to_string([
                    int(face.light() * 150),
                    int(face.light() * 150),
                    int(face.light() * 150),
                ])
            )
    if is_rotating:
        move_id = root.after(70, move)


def switch_figure():
    global polyhedron
    if switch_button.cget('text') == 'Switch to Cube':
        polyhedron = create_cube()
        switch_button.config(text="Switch to Octaedron")
    elif switch_button.cget('text') == 'Switch to Octaedron':
        polyhedron = create_octaedron()
        switch_button.config(text="Switch to Dodecaedron")
    elif switch_button.cget('text') == 'Switch to Dodecaedron':
        polyhedron = create_dodecahedron()
        switch_button.config(text="Switch to Ikosahedron")
    elif switch_button.cget('text') == "Switch to Ikosahedron":
        polyhedron = create_icosahedron()
        switch_button.config(text="Switch to Tetrahedron")
    else:
        polyhedron = create_tetraedr()
        switch_button.config(text="Switch to Cube")


def toggle_rotation():
    global is_rotating
    is_rotating = not is_rotating
    if is_rotating:
        rotation_button.config(text="Stop Rotation")
        move()
    else:
        rotation_button.config(text="Start Rotation")


root = tk.Tk()
root.title("3D Polyhedron Rotation")


button_frame = tk.Frame(root)
button_frame.pack(side=tk.TOP, pady=10)


switch_button = tk.Button(button_frame, text="Switch to Cube", command=switch_figure)
switch_button.pack(side=tk.LEFT, padx=5)


rotation_button = tk.Button(button_frame, text="Stop Rotation", command=toggle_rotation)
rotation_button.pack(side=tk.LEFT, padx=5)


w = tk.Canvas(root, width=500, height=500)
w.pack()
move()
root.mainloop()