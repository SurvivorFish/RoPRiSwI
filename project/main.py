from numpy.f2py.crackfortran import previous_context

import drawing
import time  # Для паузы в анимации, если нужно
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors

# =============================================
# Переобозначения для красоты
# =============================================
Vector = drawing.Vector
create_octahedron = drawing.create_octahedron
Point = drawing.Point
visualize_polyhedron = drawing.visualize_polyhedron
create_shararam = drawing.create_shararam

# =============================================
# Основной блок (Демонстрация с анимацией)
# =============================================
if __name__ == "__main__":
    # --- Параметры ---
    BODY_SIZE = 0.25  # Размер тела (расстояние от центра до вершины для октаэдра)
    ROTATION_AXIS = 'z'  # Вокруг какой оси вращать ('x', 'y', 'z')
    ROTATION_SPEED_DEG_PER_SEC = 1  # Скорость вращения в градусах в секунду
    ANIMATION_INTERVAL_MS = 30  # Интервал обновления кадра (в миллисекундах)
    AMBIENT_LIGHT = 0.2  # Уровень фонового освещения
    # Направление на источник света (например, солнце сверху-справа-спереди)
    light_direction = Vector(1, 1, 1).normalize()

    # Создаем ОКТАЭДР
    initial_body = create_shararam(Point(0, 0, 0), m=100000000000, r=BODY_SIZE, N=10, M=10, V=Vector(0, 0, 0))
    secondary_body = create_octahedron(Point(0, 5, 0), m=100000, size=0.5 * BODY_SIZE, V=Vector(5, 0, 0))
    print(secondary_body.V)

    # --- Настройка Matplotlib ---
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection='3d')

    # --- Функция обновления для анимации ---
    start_time = time.time()
    previous_time = time.time()
    fps = 30
    dt = 1/fps


    def update(frame):
        global initial_body, secondary_body
        # Вычисляем текущий угол поворота
        '''current_angle_deg = (elapsed_time * ROTATION_SPEED_DEG_PER_SEC) % 360
        current_angle_rad = math.radians(current_angle_deg)'''

        # Вращаем ИСХОДНОЕ тело (чтобы избежать накопления ошибок)
        '''initial_body = initial_body.move(initial_body.V*dt)
        secondary_body = secondary_body.move(secondary_body.V*dt)'''


        # Визуализируем повернутое тело
        frame_title = "Симуляция"
        visualize_polyhedron(ax, (initial_body, secondary_body), light_direction,
                             title=frame_title, ambient_light=AMBIENT_LIGHT)

        # Возвращаем измененные элементы (для blitting, хотя здесь он может не работать)
        # return ax.patches + ax.lines + ax.texts + [ax.collections] # Примерно так
        return ax,  # Проще вернуть оси целиком, если blit=False


    # --- Создание и запуск анимации ---
    # frames=None означает, что анимация будет идти "бесконечно" (пока не закроете окно)
    # blit=False обычно надежнее для 3D графики matplotlib
    ani = FuncAnimation(fig, update, frames=None,
                        interval=ANIMATION_INTERVAL_MS, blit=False,
                        cache_frame_data=False)  # cache_frame_data=False, чтобы время было актуальным

    plt.tight_layout()
    plt.show()

    """from threading import Thread

    def writer(n):
        for i in range(n):
            print(i)
            time.sleep(1)

    def plotdraw():
        plt.tight_layout()
        plt.show()

    phys = Thread(target=writer, args=(500000,))
    draw = Thread(target=plotdraw)
    draw.start()
    phys.start()
    draw.join()
    phys.join()"""

    print("Запуск анимации...")
    print(f"Вращение Октаэдра вокруг оси '{ROTATION_AXIS.upper()}' со скоростью {ROTATION_SPEED_DEG_PER_SEC} град/сек.")
    print(f"Свет: {light_direction}")

    print("Окно закрыто.")
