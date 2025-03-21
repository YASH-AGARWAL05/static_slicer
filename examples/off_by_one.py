"""
Example of a classic off-by-one error in Python.
"""


def process_list(items):
    """
    Process a list by combining adjacent elements.
    Contains a classic off-by-one error in the loop boundary.
    """
    if not items:
        return []

    result = []

    # Bug: The loop should stop at len(items)-1 to avoid index out of range
    # This is a common off-by-one error
    for index in range(len(items)):
        try:
            # This will fail at the last index with an IndexError
            # Because items[index+1] will be out of bounds
            current = items[index]

            # Attempt to access the next element (will fail for the last element)
            next_value = items[index + 1]

            # Process the elements together
            combined = f"{current}+{next_value}"
            result.append(combined)
        except IndexError:
            # This handles the error, but it's better to fix the loop boundary
            # Proper fix would be: for index in range(len(items)-1)
            result.append(str(items[index]))

    return result


def fix_process_list(items):
    """
    Fixed version of the function that properly handles list boundaries.
    """
    if not items:
        return []

    result = []

    # Correctly iterate to avoid index error
    for index in range(len(items) - 1):
        current = items[index]
        next_value = items[index + 1]
        combined = f"{current}+{next_value}"
        result.append(combined)

    # Add the last element
    if items:
        result.append(str(items[-1]))

    return result


def validate_results(original, fixed):
    """Compare results from both implementations."""
    if original == fixed:
        print("✅ Both implementations produce the same result.")
    else:
        print("❌ Results differ!")
        print(f"Original implementation: {original}")
        print(f"Fixed implementation: {fixed}")

        # Analyze the difference
        if len(original) != len(fixed):
            print(f"Different lengths: original={len(original)}, fixed={len(fixed)}")

        for i, (o, f) in enumerate(zip(original, fixed)):
            if o != f:
                print(f"First difference at index {i}: {o} vs {f}")


if __name__ == "__main__":
    # Test with a numeric list
    test_data = [10, 20, 30, 40, 50]

    print("Original list:", test_data)

    # Run both implementations
    result1 = process_list(test_data)
    result2 = fix_process_list(test_data)

    print("\nBuggy implementation result:", result1)
    print("Fixed implementation result:", result2)

    print("\nValidation:")
    validate_results(result1, result2)

    # Test with an edge case (single item)
    print("\n--- Testing with a single item ---")
    single_item = [42]

    print("Original list:", single_item)
    result1 = process_list(single_item)
    result2 = fix_process_list(single_item)

    print("Buggy implementation result:", result1)
    print("Fixed implementation result:", result2)

    print("\nValidation:")
    validate_results(result1, result2)