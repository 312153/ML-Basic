"""
Доработайте класс `Vehicle`
"""

from abc import ABC

from homework_05 import exceptions


class Vehicle(ABC):
    # Атрибуты со значениями по умолчанию (уровень класса).
    # `started` намеренно НЕ попадает в инициализатор: машина/самолёт
    # всегда создаётся заглушённым, завести его — отдельное действие.
    weight: float = 0
    started: bool = False
    fuel: float = 0
    fuel_consumption: float = 0

    def __init__(self, weight: float = 0, fuel: float = 0, fuel_consumption: float = 0):
        self.weight = weight
        self.fuel = fuel
        self.fuel_consumption = fuel_consumption

    def start(self):
        """Завести двигатель: если ещё не заведён — проверить топливо."""
        if not self.started:
            if self.fuel > 0:
                self.started = True
            else:
                raise exceptions.LowFuelError

    def move(self, distance: float):
        """Проехать `distance`, списав топливо. Хватает вплоть до нуля."""
        needed_fuel = distance * self.fuel_consumption
        if self.fuel >= needed_fuel:
            self.fuel -= needed_fuel
        else:
            raise exceptions.NotEnoughFuel
