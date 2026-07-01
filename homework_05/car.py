"""
Создайте класс `Car`, наследник `Vehicle`
"""

from homework_05.base import Vehicle
from homework_05.engine import Engine


class Car(Vehicle):
    # У машины по умолчанию двигатель не установлен.
    engine: Engine | None = None

    def set_engine(self, engine: Engine):
        """Установить экземпляр Engine на текущую машину."""
        self.engine = engine
