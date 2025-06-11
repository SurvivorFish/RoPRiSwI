import math
from typing import List, Tuple, Optional

# --- импорты для визуализации ---
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors

# Константа для сравнения чисел с плавающей запятой
EPSILON = 1e-9


# =============================================
# Класс Точка (Point)
# =============================================
class Point:
    """Представляет точку в 3D пространстве."""

    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self) -> str:
        """Возвращает строковое представление точки."""
        return f"Point({self.x:.3f}, {self.y:.3f}, {self.z:.3f})"

    def __eq__(self, other: object) -> bool:
        """Проверяет равенство двух точек с учетом погрешности."""
        if not isinstance(other, Point):
            return NotImplemented
        return (abs(self.x - other.x) < EPSILON and
                abs(self.y - other.y) < EPSILON and
                abs(self.z - other.z) < EPSILON)

    def __sub__(self, other: 'Point') -> 'Vector':
        """Вычитание двух точек дает вектор."""
        if not isinstance(other, Point):
            return NotImplemented
        return Vector(self.x - other.x, self.y - other.y, self.z - other.z)

    def __add__(self, vector: 'Vector') -> 'Point':
        """Сложение точки и вектора дает новую точку (смещение)."""
        if not isinstance(vector, Vector):
            return NotImplemented
        return Point(self.x + vector.x, self.y + vector.y, self.z + vector.z)

    def distance_to(self, other: 'Point') -> float:
        """Расстояние до другой точки."""
        return (self - other).magnitude()

    def rotate_x(self, angle_rad: float, center: Optional['Point'] = None) -> 'Point':
        """Вращение точки вокруг оси X через центр center."""
        if center is None:
            center = Point(0, 0, 0)

        p_relative = self - center  # Переносим в начало координат относительно центра
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)

        y_new = p_relative.y * cos_a - p_relative.z * sin_a
        z_new = p_relative.y * sin_a + p_relative.z * cos_a
        # Создаем новый повернутый относительный вектор
        rotated_relative_vec = Vector(p_relative.x, y_new, z_new)
        # Переносим обратно
        return center + rotated_relative_vec

    def rotate_y(self, angle_rad: float, center: Optional['Point'] = None) -> 'Point':
        """Вращение точки вокруг оси Y через центр center."""
        if center is None:
            center = Point(0, 0, 0)

        p_relative = self - center
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)

        x_new = p_relative.x * cos_a + p_relative.z * sin_a
        z_new = -p_relative.x * sin_a + p_relative.z * cos_a
        rotated_relative_vec = Vector(x_new, p_relative.y, z_new)
        return center + rotated_relative_vec

    def rotate_z(self, angle_rad: float, center: Optional['Point'] = None) -> 'Point':
        """Вращение точки вокруг оси Z через центр center."""
        if center is None:
            center = Point(0, 0, 0)

        p_relative = self - center
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)

        x_new = p_relative.x * cos_a - p_relative.y * sin_a
        y_new = p_relative.x * sin_a + p_relative.y * cos_a
        rotated_relative_vec = Vector(x_new, y_new, p_relative.z)
        return center + rotated_relative_vec


# =============================================
# Класс Вектор (Vector)
# =============================================
class Vector:
    """Представляет вектор в 3D пространстве."""

    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self) -> str:
        """Возвращает строковое представление вектора."""
        return f"Vector({self.x:.3f}, {self.y:.3f}, {self.z:.3f})"

    def __eq__(self, other: object) -> bool:
        """Проверяет равенство двух векторов с учетом погрешности."""
        if not isinstance(other, Vector):
            return NotImplemented
        return (abs(self.x - other.x) < EPSILON and
                abs(self.y - other.y) < EPSILON and
                abs(self.z - other.z) < EPSILON)

    def __add__(self, other: 'Vector') -> 'Vector':
        """Сложение двух векторов."""
        if not isinstance(other, Vector):
            return NotImplemented
        return Vector(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: 'Vector') -> 'Vector':
        """Вычитание двух векторов."""
        if not isinstance(other, Vector):
            return NotImplemented
        return Vector(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> 'Vector':
        """Умножение вектора на скаляр."""
        if not isinstance(scalar, (int, float)):
            return NotImplemented
        return Vector(self.x * scalar, self.y * scalar, self.z * scalar)

    def __rmul__(self, scalar: float) -> 'Vector':
        """Умножение скаляра на вектор (для поддержки scalar * vector)."""
        return self.__mul__(scalar)

    def __truediv__(self, scalar: float) -> 'Vector':
        """Деление вектора на скаляр."""
        if not isinstance(scalar, (int, float)):
            return NotImplemented
        if abs(scalar) < EPSILON:
            raise ZeroDivisionError("Нельзя делить вектор на ноль.")
        return Vector(self.x / scalar, self.y / scalar, self.z / scalar)

    def __neg__(self) -> 'Vector':
        """Возвращает противоположный вектор."""
        return Vector(-self.x, -self.y, -self.z)

    def magnitude_squared(self) -> float:
        """Возвращает квадрат длины (магнитуды) вектора."""
        return self.x ** 2 + self.y ** 2 + self.z ** 2

    def magnitude(self) -> float:
        """Возвращает длину (магнитуду) вектора."""
        mag_sq = self.magnitude_squared()
        # Избегаем ошибки с корнем из отрицательного числа из-за погрешностей
        return math.sqrt(max(0.0, mag_sq))

    def normalize(self) -> 'Vector':
        """Возвращает нормализованный вектор (единичной длины)."""
        mag = self.magnitude()
        if mag < EPSILON:
            # Можно вернуть нулевой вектор или вызвать исключение
            # print("Warning: Normalizing zero vector")
            return Vector(0, 0, 0)
        return self / mag

    def dot(self, other: 'Vector') -> float:
        """Скалярное произведение двух векторов."""
        if not isinstance(other, Vector):
            return NotImplemented
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: 'Vector') -> 'Vector':
        """Векторное произведение двух векторов."""
        if not isinstance(other, Vector):
            return NotImplemented
        return Vector(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )

    def angle_between(self, other: 'Vector') -> float:
        """Угол (в радианах) между двумя векторами."""
        mag_product = self.magnitude() * other.magnitude()
        if mag_product < EPSILON:
            return 0.0  # Угол между нулевым вектором и любым другим не определен строго, примем 0

        dot_product = self.dot(other)
        # Избегаем выхода за пределы [-1, 1] из-за ошибок округления
        cos_theta = max(-1.0, min(1.0, dot_product / mag_product))
        # Дополнительная проверка на случай, если все же вышли за пределы
        if not (-1.0 <= cos_theta <= 1.0):
            print(f"Warning: acos input out of range: {cos_theta}. Clamping.")
            cos_theta = max(-1.0, min(1.0, cos_theta))
        return math.acos(cos_theta)

    def rotate_x(self, angle_rad: float) -> 'Vector':
        """Вращение вектора вокруг оси X."""
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        y_new = self.y * cos_a - self.z * sin_a
        z_new = self.y * sin_a + self.z * cos_a
        return Vector(self.x, y_new, z_new)

    def rotate_y(self, angle_rad: float) -> 'Vector':
        """Вращение вектора вокруг оси Y."""
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        x_new = self.x * cos_a + self.z * sin_a
        z_new = -self.x * sin_a + self.z * cos_a
        return Vector(x_new, self.y, z_new)

    def rotate_z(self, angle_rad: float) -> 'Vector':
        """Вращение вектора вокруг оси Z."""
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        x_new = self.x * cos_a - self.y * sin_a
        y_new = self.x * sin_a + self.y * cos_a
        return Vector(x_new, y_new, self.z)


# =============================================
# Класс Плоскость (Plane)
# =============================================
class Plane:
    """Представляет плоскость в 3D пространстве."""

    def __init__(self, point_on_plane: Point, normal_vector: Vector):
        """
        Создает плоскость, проходящую через point_on_plane
        с нормалью normal_vector.
        """
        self.point = point_on_plane
        # Нормаль всегда хранится нормализованной
        self.normal = normal_vector.normalize()
        # Коэффициент D из уравнения плоскости Ax + By + Cz + D = 0
        origin_to_point_vec = Vector(self.point.x, self.point.y, self.point.z)
        self.d = -self.normal.dot(origin_to_point_vec)

    def __repr__(self) -> str:
        """Возвращает строковое представление плоскости (уравнение)."""
        return (f"Plane({self.normal.x:.2f}x + {self.normal.y:.2f}y + "
                f"{self.normal.z:.2f}z + {self.d:.2f} = 0)")

    @classmethod
    def from_points(cls, p1: Point, p2: Point, p3: Point) -> Optional['Plane']:
        """Создает плоскость по трем точкам. Возвращает None при коллинеарности."""
        v1 = p2 - p1
        v2 = p3 - p1
        normal = v1.cross(v2)
        if normal.magnitude() < EPSILON:
            print(f"Warning: Точки {p1}, {p2}, {p3} коллинеарны, нельзя создать плоскость.")
            return None
        return cls(p1, normal)

    def signed_distance_to_point(self, point: Point) -> float:
        """
        Возвращает знаковое расстояние от точки до плоскости.
        Положительно, если точка находится "над" плоскостью (в направлении нормали),
        отрицательно, если "под", и ноль, если на плоскости.
        """
        point_vec = Vector(point.x, point.y, point.z)
        return self.normal.dot(point_vec) + self.d
        # Альтернативный способ: return self.normal.dot(point - self.point)

    def is_on_plane(self, point: Point, tolerance: float = EPSILON) -> bool:
        """Проверяет, лежит ли точка на плоскости с заданной точностью."""
        return abs(self.signed_distance_to_point(point)) < tolerance


# =============================================
# Класс Грань (Facet)
# =============================================
class Facet:
    """Представляет грань многогранника (плоский многоугольник)."""

    def __init__(self, vertices: List[Point]):
        if len(vertices) < 3:
            raise ValueError("Грань должна иметь минимум 3 вершины.")
        self.vertices = vertices
        self._normal: Optional[Vector] = None
        self._plane: Optional[Plane] = None  # Кэшируем плоскость и нормаль

    def __repr__(self) -> str:
        """Возвращает строковое представление грани."""
        verts_str = ", ".join(map(str, self.vertices))
        return f"Facet([{verts_str}])"

    @property
    def center(self) -> Point:
        """Приблизительный центр грани (среднее арифметическое вершин)."""
        sum_x = sum(v.x for v in self.vertices)
        sum_y = sum(v.y for v in self.vertices)
        sum_z = sum(v.z for v in self.vertices)
        n = len(self.vertices)
        if n == 0: return Point(0, 0, 0)  # Защита от деления на ноль
        return Point(sum_x / n, sum_y / n, sum_z / n)

    @property
    def normal(self) -> Vector:
        """
        Вычисляет и кэширует нормаль грани. Нормаль направлена "наружу" при
        правильном порядке обхода вершин (против часовой стрелки,
        если смотреть снаружи).
        """
        if self._normal is None:
            if len(self.vertices) < 3:
                self._normal = Vector(0, 0, 0)  # Невозможно вычислить
            else:
                v1 = self.vertices[1] - self.vertices[0]
                # Ищем вторую неколлинеарную точку
                cross_product = Vector(0, 0, 0)
                for i in range(2, len(self.vertices)):
                    v2 = self.vertices[i] - self.vertices[0]
                    cross_product = v1.cross(v2)
                    if cross_product.magnitude() >= EPSILON:
                        break  # Нашли подходящую пару векторов

                if cross_product.magnitude() < EPSILON:
                    print(
                        f"Warning: Не удалось вычислить нормаль для грани с вершинами {self.vertices}. Вершины могут быть коллинеарны.")
                    self._normal = Vector(0, 0, 0)  # Признак проблемы
                else:
                    self._normal = cross_product.normalize()
        return self._normal

    @property
    def plane(self) -> Optional[Plane]:
        """Возвращает плоскость, содержащую эту грань. Может вернуть None."""
        if self._plane is None:
            if len(self.vertices) >= 3:
                # Пытаемся создать плоскость из первых трех точек
                self._plane = Plane.from_points(self.vertices[0], self.vertices[1], self.vertices[2])
                # Если не получилось (коллинеарны), попробуем другую комбинацию, если есть
                if self._plane is None and len(self.vertices) > 3:
                    for i in range(3, len(self.vertices)):
                        self._plane = Plane.from_points(self.vertices[0], self.vertices[1], self.vertices[i])
                        if self._plane is not None:
                            break  # Нашли рабочую комбинацию
                # Проверка: если все равно не создали, значит проблема с гранью
                if self._plane is None:
                    print(f"Warning: Невозможно создать плоскость для грани {self.vertices}")

            else:
                # Ситуация с менее чем 3 вершинами - неясно, какая плоскость
                print(f"Warning: Невозможно определить плоскость для грани с < 3 вершинами: {self.vertices}")
                self._plane = None  # Явно указываем невозможность

        return self._plane

    def rotate(self, axis: str, angle_rad: float, center: Point) -> 'Facet':
        """
        Вращает грань вокруг заданной оси ('x', 'y', 'z') на угол angle_rad
        вокруг точки center. Возвращает новую грань.
        """
        rotated_vertices = []
        for vertex in self.vertices:
            if axis.lower() == 'x':
                rotated_vertices.append(vertex.rotate_x(angle_rad, center))
            elif axis.lower() == 'y':
                rotated_vertices.append(vertex.rotate_y(angle_rad, center))
            elif axis.lower() == 'z':
                rotated_vertices.append(vertex.rotate_z(angle_rad, center))
            else:
                raise ValueError("Ось должна быть 'x', 'y' или 'z'")
        # Создаем новую грань, нормаль и плоскость пересчитаются при доступе
        return Facet(rotated_vertices)

    def move(self, vector: Vector) -> 'Facet':
        """
        Вращает грань вокруг заданной оси ('x', 'y', 'z') на угол angle_rad
        вокруг точки center. Возвращает новую грань.
        """
        vs = []
        for vertex in self.vertices:
            x = vertex.x + vector.x
            y = vertex.y + vector.y
            z = vertex.z + vector.z
            vs.append(Point(x, y, z))

        # Создаем новую грань
        return Facet(vs)


# =============================================
# Класс Многогранник (Polyhedron)
# =============================================
class Polyhedron:
    """Представляет многогранник как набор граней."""

    def __init__(self, facets: List[Facet], V: Vector = Vector(0, 0, 0), edge_col: str = 'black', face_col : str = 'red'):
        self.facets = facets
        self._vertices_cache: Optional[List[Point]] = None
        self._center_cache: Optional[Point] = None
        self.V = V  # скорость
        self.edge_col = edge_col
        self.face_col = face_col

    def __repr__(self) -> str:
        return f"Polyhedron(facets={len(self.facets)})"

    @property
    def vertices(self) -> List[Point]:
        """Возвращает список всех уникальных вершин многогранника (кэшированный)."""
        if self._vertices_cache is None:
            unique_vertices = []
            seen_coords = set()
            for facet in self.facets:
                for vertex in facet.vertices:
                    # Используем округление для борьбы с ошибками float при поиске уникальных
                    rounded_coords = (round(vertex.x, 6), round(vertex.y, 6), round(vertex.z, 6))
                    if rounded_coords not in seen_coords:
                        unique_vertices.append(vertex)
                        seen_coords.add(rounded_coords)
            self._vertices_cache = unique_vertices
        return self._vertices_cache

    @property
    def center(self) -> Point:
        """Приблизительный центр многогранника (среднее арифм. уник. вершин, кэшир.)."""
        if self._center_cache is None:
            all_vertices = self.vertices  # Используем кэшированные вершины
            if not all_vertices:
                self._center_cache = Point(0, 0, 0)
            else:
                sum_x = sum(v.x for v in all_vertices)
                sum_y = sum(v.y for v in all_vertices)
                sum_z = sum(v.z for v in all_vertices)
                n = len(all_vertices)
                self._center_cache = Point(sum_x / n, sum_y / n, sum_z / n)
        return self._center_cache

    def rotate(self, axis: str, angle_rad: float, center: Optional[Point] = None) -> 'Polyhedron':
        """
        Вращает многогранник вокруг заданной оси ('x', 'y', 'z') на угол angle_rad.
        Если center не задан, вращение происходит вокруг геометрического центра многогранника.
        Возвращает новый многогранник (с очищенным кэшем).
        """
        rotation_center = center if center is not None else self.center

        rotated_facets = [facet.rotate(axis, angle_rad, rotation_center) for facet in self.facets]

        # Создаем новый объект, кэш будет пересчитан при первом доступе
        return Polyhedron(rotated_facets)

    def move(self, vector: Vector) -> 'Polyhedron':
        """
        Вращает многогранник вокруг заданной оси ('x', 'y', 'z') на угол angle_rad.
        Если center не задан, вращение происходит вокруг геометрического центра многогранника.
        Возвращает новый многогранник (с очищенным кэшем).
        """
        rotated_facets = [facet.move(vector) for facet in self.facets]
        # Создаем новый объект, кэш будет пересчитан при первом доступе
        return Polyhedron(rotated_facets)

    # --- Методы, связанные с видимостью и освещением ---

    def is_facet_visible(self, facet: Facet, view_vector: Vector) -> bool:
        """
        Определяет, видна ли грань с точки зрения наблюдателя (back-face culling).
        view_vector - вектор ОТ грани К наблюдателю.
        """
        # Если нормаль нулевая (проблема с гранью), считаем невидимой
        normal = facet.normal
        if normal.magnitude() < EPSILON:
            return False
        dot_product = normal.dot(view_vector)
        # Грань видна, если угол > 90 градусов (скалярное произведение < 0)
        # или если вектор взгляда почти параллелен плоскости (для ребер)
        return dot_product < EPSILON  # Слегка смягчаем условие

    def calculate_lighting(self, facet: Facet, light_direction: Vector, ambient_light: float = 0.1) -> float:
        """
        Рассчитывает интенсивность освещения грани (простая модель Ламберта + Ambient).
        Возвращает значение от ambient_light (минимум) до 1.0 (максимум).
        light_direction - нормализованный вектор В НАПРАВЛЕНИИ источника света.
        ambient_light - минимальный уровень освещенности (0 до 1).
        """
        normal = facet.normal
        if normal.magnitude() < EPSILON:
            return ambient_light  # Неосвещенная, если нет нормали

        # Косинус угла между нормалью и направлением света
        # Интенсивность диффузного света (от 0 до 1)
        diffuse_intensity = max(0, normal.dot(light_direction.normalize()))

        # Полная интенсивность = фоновый свет + диффузный свет * (1 - фоновый)
        # чтобы максимальное значение оставалось 1.0
        total_intensity = ambient_light + diffuse_intensity * (1.0 - ambient_light)
        return max(ambient_light, min(1.0, total_intensity))  # Ограничиваем [ambient, 1.0]

    @center.setter
    def center(self, value):
        self._center = value


# =============================================
# Функция создания Октаэдра (Восьмигранника)
# =============================================
def create_octahedron(center: Point or tuple, size: float, V: Vector = Vector(0, 0, 0)) -> Polyhedron:
    if type(center) == tuple:
        x = center[0]
        y = center[1]
        z = center[2]
        center = Point(x, y, z)
    """Создает многогранник-октаэдр с центром в center.
       size - расстояние от центра до вершины вдоль оси."""
    s = size  # Используем size напрямую как координату вершины
    cx, cy, cz = center.x, center.y, center.z

    # 6 вершин октаэдра
    v = [
        Point(cx + s, cy, cz),  # 0: +X
        Point(cx - s, cy, cz),  # 1: -X
        Point(cx, cy + s, cz),  # 2: +Y
        Point(cx, cy - s, cz),  # 3: -Y
        Point(cx, cy, cz + s),  # 4: +Z (Верх)
        Point(cx, cy, cz - s)  # 5: -Z (Низ)
    ]

    # 8 треугольных граней
    # Порядок вершин важен для правильной ориентации нормали (против часовой стрелки снаружи)
    facets = [
        # Верхние 4 грани (смотрят вверх)
        Facet([v[4], v[0], v[2]]),  # Верх, +X, +Y
        Facet([v[4], v[2], v[1]]),  # Верх, +Y, -X
        Facet([v[4], v[1], v[3]]),  # Верх, -X, -Y
        Facet([v[4], v[3], v[0]]),  # Верх, -Y, +X
        # Нижние 4 грани (смотрят вниз)
        Facet([v[5], v[2], v[0]]),  # Низ, +Y, +X
        Facet([v[5], v[1], v[2]]),  # Низ, -X, +Y
        Facet([v[5], v[3], v[1]]),  # Низ, -Y, -X
        Facet([v[5], v[0], v[3]])  # Низ, +X, -Y
    ]
    return Polyhedron(facets, V=V)


# =============================================
# Функция создания Шара
# =============================================
def create_shararam(center: Point or tuple, r: float, n: int, m: int, V: Vector = Vector(0, 0, 0)) -> Polyhedron:
    if type(center) == tuple:
        cx = center[0]
        cy = center[1]
        cz = center[2]
    else:
        cx, cy, cz = center.x, center.y, center.z

    """Создает сферу из треугольничков с центром в center.
       r - радиус.
       n, m - количество точек, см ниже"""
    ps = [Point(cx, cy, cz + r)]  # = points - list точек
    fs = []  # = faces - list граней
    for i in range(1, n + 1):  # vertical angle (first and last points already added)
        phi = i * math.pi / (n + 1)
        for e in range(m):  # horizontal part of angle
            theta = e * 2 * math.pi * 1 / m
            ps.append(Point(cx + r * math.sin(phi) * math.cos(theta), cy + r * math.sin(phi) * math.sin(theta),
                            cz + r * math.cos(phi)))
    ps.append(Point(cx, cy, cz - r))

    # сначала добавим верхний и нижний уровни (которые касаются тех точек)
    # всего точек 1 + n*m + 1 = n*m+2
    for i in range(m - 1):
        fs.append(Facet([ps[0], ps[1 + i], ps[2 + i]]))
    fs.append(Facet([ps[m], ps[1], ps[0]]))
    for i in range(m - 1):
        fs.append(Facet([ps[n * m + 1], ps[n * m - i], ps[n * m - i - 1]]))
    fs.append(Facet([ps[n * m + 1], ps[n * m - m + 1], ps[n * m]]))

    if n != 1:
        for i in range(n - 1):
            for j in range(m):
                if j != m - 1:
                    fs.append(Facet([ps[i * m + j + 1], ps[(i + 1) * m + j + 1], ps[(i + 1) * m + j + 2]]))
                else:
                    fs.append(Facet([ps[i * m + m], ps[(i + 1) * m + m], ps[(i + 1) * m + 1]]))
                if j != 0:
                    fs.append(Facet([ps[(i + 1) * m + j + 1], ps[i * m + j + 1], ps[i * m + j]]))
                else:
                    fs.append(Facet([ps[(i + 1) * m + 1], ps[i * m + 1], ps[i * m + m]]))
        return Polyhedron(fs, edge_col='red', V=V)


# =============================================
# Функции для Визуализации
# =============================================
def _get_polyhedron_bounds(polyhedron: Polyhedron, padding: float = 0.1) -> Tuple[
    Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    """Вычисляет границы многогранника для установки пределов осей."""
    vertices = polyhedron.vertices
    if not vertices:
        return ((-1, 1), (-1, 1), (-1, 1))

    min_x = min(v.x for v in vertices)
    max_x = max(v.x for v in vertices)
    min_y = min(v.y for v in vertices)
    max_y = max(v.y for v in vertices)
    min_z = min(v.z for v in vertices)
    max_z = max(v.z for v in vertices)

    # Используем максимальный размах для определения отступа
    max_range = max(max_x - min_x, max_y - min_y, max_z - min_z, 1.0)  # Минимум 1.0
    pad = max_range * (0.5 + padding)  # Отступ чуть больше

    # Центрируем пределы вокруг центра объекта для лучшего вида
    center = polyhedron.center
    return (
        (center.x - pad, center.x + pad),
        (center.y - pad, center.y + pad),
        (center.z - pad, center.z + pad)
    )


def get_shaded_color(base_color_name: str, intensity: float) -> Tuple[float, float, float, float]:
    """Получает RGBA цвет на основе базового и интенсивности освещения."""
    base_rgba = mcolors.to_rgba(base_color_name)
    # Умножаем RGB компоненты на интенсивность, альфа остается той же
    # (можно сделать альфу зависимой от интенсивности, если нужно)
    shaded_rgb = tuple(max(0, min(1, c * intensity)) for c in base_rgba[:3])
    return (*shaded_rgb, base_rgba[3])  # Возвращаем как RGBA


def visualize_polyhedron(ax: Axes3D, polyhedrons: Polyhedron or tuple or list,
                         light_dir: Vector, title: str = "",
                         ambient_light: float = 0.15):
    if type(polyhedrons) is Polyhedron:
        polyhedrons = list(polyhedrons)

    ax.clear()  # Очищаем предыдущий кадр

    all_vertices = []
    face_colors = []
    edge_colors = []
    for polyhedron in polyhedrons:
        for facet in polyhedron.facets:
            view_vector = facet.center  # Вектор от центра грани к камере

            # Получаем координаты вершин для Poly3DCollection
            verts = [(p.x, p.y, p.z) for p in facet.vertices]
            all_vertices.append(verts)

            # Расчет освещенности видимой грани
            intensity = polyhedron.calculate_lighting(facet, light_dir, ambient_light)
            f_color = get_shaded_color(polyhedron.face_col, intensity)
            e_color = polyhedron.edge_col

            face_colors.append(f_color)
            edge_colors.append(e_color)

    if not all_vertices:  # Если нет видимых граней
        print("Нет видимых граней для отрисовки.")
        # Нарисуем хотя бы оси
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
    else:
        # Создаем коллекцию полигонов
        poly_collection = Poly3DCollection(all_vertices, facecolors=face_colors,
                                           edgecolors=edge_colors, linewidths=0.5, zsort='average')
        ax.add_collection3d(poly_collection)

        # Настройка осей
        x_lim, y_lim, z_lim = (0, 0), (0, 0), (0, 0)
        for polyhedron in polyhedrons:
            x_lim0, y_lim0, z_lim0 = _get_polyhedron_bounds(polyhedron, padding=0.2)
            x_lim, y_lim, z_lim = (min(x_lim[0], x_lim0[0]), max(x_lim[1], x_lim0[1])), (
                min(y_lim[0], y_lim0[0]), max(y_lim[1], y_lim0[1])), (
                                      min(z_lim[0], z_lim0[0]), max(z_lim[1], z_lim0[1]))
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_zlim(z_lim)
        # Важно для правильного отображения пропорций
        ax.set_box_aspect([ub - lb for lb, ub in [x_lim, y_lim, z_lim]])  # Равные пропорции

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.6)
