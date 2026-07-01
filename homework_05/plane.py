"""
Создайте класс `Plane`, наследник `Vehicle`
"""

from homework_05.base import Vehicle
from homework_05 import exceptions


class Plane(Vehicle):
    cargo: float = 0
    max_cargo: float = 0

    def __init__(
        self,
        weight: float = 0,
        fuel: float = 0,
        fuel_consumption: float = 0,
        max_cargo: float = 0,
    ):
        # Переопределяем инициализатор осмысленно: добавляем новый
        # аргумент max_cargo, а общую часть делегируем родителю.
        super().__init__(weight, fuel, fuel_consumption)
        self.max_cargo = max_cargo

    def load_cargo(self, amount: float):
        """Догрузить `amount`, если это не приведёт к перегрузу."""
        if self.cargo + amount <= self.max_cargo:
            self.cargo += amount
        else:
            raise exceptions.CargoOverload

    def remove_all_cargo(self) -> float:
        """Разгрузить всё; вернуть, сколько груза было до разгрузки."""
        current_cargo = self.cargo
        self.cargo = 0
        return current_cargo
