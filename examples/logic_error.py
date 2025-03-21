"""
Example of a program with a logic error in discount calculation.
"""


def calculate_discount(price, quantity):
    """
    Calculate the final price after applying a quantity-based discount.

    Logic error: discount is calculated on unit price, not total price.
    Should use (price * quantity * discount_rate) instead of (price * discount_rate).
    """
    # Determine discount rate based on quantity
    if quantity >= 10:
        discount_rate = 0.1  # 10% discount for 10+ items
    elif quantity >= 5:
        discount_rate = 0.05  # 5% discount for 5-9 items
    else:
        discount_rate = 0  # No discount for fewer than 5 items

    # Logic error: discount applied to unit price only, not total
    discount = price * discount_rate

    # Calculate final price
    final_price = price * quantity - discount

    return final_price


def display_receipt(item_name, price, quantity):
    """Generate a receipt with price breakdown."""
    total = calculate_discount(price, quantity)

    # Calculate what the correct total should be
    if quantity >= 10:
        correct_discount_rate = 0.1
    elif quantity >= 5:
        correct_discount_rate = 0.05
    else:
        correct_discount_rate = 0

    correct_total = price * quantity * (1 - correct_discount_rate)

    print(f"Receipt for {quantity}x {item_name}")
    print(f"Unit price: ${price:.2f}")
    print(f"Quantity: {quantity}")
    print(f"Subtotal: ${price * quantity:.2f}")

    if quantity >= 5:
        print(f"Discount applied: {correct_discount_rate * 100:.0f}%")

    print(f"Final price: ${total:.2f}")

    # Check if there's a mismatch due to the logic error
    if abs(total - correct_total) > 0.01:
        print("\nWARNING: Potential pricing error detected!")
        print(f"Expected total: ${correct_total:.2f}")
        print(f"Difference: ${abs(total - correct_total):.2f}")


if __name__ == "__main__":
    # Test with different quantities
    print("=== Small Order ===")
    display_receipt("Widget", 9.99, 3)

    print("\n=== Medium Order ===")
    display_receipt("Gadget", 19.99, 7)

    print("\n=== Large Order ===")
    display_receipt("Thingamajig", 24.99, 12)