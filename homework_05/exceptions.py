"""
Объявите следующие исключения:
- LowFuelError
- NotEnoughFuel
- CargoOverload
"""


class LowFuelError(Exception):
    """Попытка завестись, когда топлива нет (fuel <= 0)."""


class NotEnoughFuel(Exception):
    """Топлива не хватает, чтобы преодолеть запрошенную дистанцию."""


class CargoOverload(Exception):
    """Загружаемый груз в сумме с текущим превышает max_cargo."""
