"""Toy module used for demos and end-to-end tests."""


def ticket_price(age: int) -> int:
    """Return the ticket price in dollars for a visitor of the given age."""
    if age < 0:
        raise ValueError("age must be non-negative")
    if age < 12:
        return 5
    elif age >= 65:
        return 7
    return 10


def group_total(ages: list[int], discount: float = 0.0) -> float:
    """Total price for a group, with an optional fractional discount for 5+ people."""
    total = sum(ticket_price(a) for a in ages)
    if len(ages) >= 5 and discount > 0:
        total = total * (1 - discount)
    return round(total, 2)
