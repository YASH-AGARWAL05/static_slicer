"""
Example of a buggy sorting implementation with off-by-one error and logic issues.
"""


def bubble_sort(arr):
    """
    Implementation of bubble sort with a bug.
    The bug is that the inner loop doesn't properly account for
    already sorted elements, causing an off-by-one error.
    """
    n = len(arr)

    # Create a copy to avoid modifying the input
    sorted_list = arr.copy()

    # Bug: should be range(n-1) for the outer loop
    for i in range(n):
        # Bug: should adjust inner loop to n-i-1
        for j in range(n - 1):
            if sorted_list[j] > sorted_list[j + 1]:
                # Swap elements
                sorted_list[j], sorted_list[j + 1] = sorted_list[j + 1], sorted_list[j]

    return sorted_list


def is_sorted(arr):
    """Check if an array is sorted."""
    for i in range(len(arr) - 1):
        if arr[i] > arr[i + 1]:
            return False
    return True


def find_first_unsorted(arr):
    """Find the index of the first unsorted element."""
    for i in range(len(arr) - 1):
        if arr[i] > arr[i + 1]:
            return i
    return -1  # List is sorted


if __name__ == "__main__":
    # Test with a simple array
    test_array = [64, 34, 25, 12, 22, 11, 90]

    print("Original array:", test_array)

    # Sort using the buggy implementation
    sorted_array = bubble_sort(test_array)

    print("Sorted array:", sorted_array)

    # Check if correctly sorted
    if is_sorted(sorted_array):
        print("✅ Array is correctly sorted")
    else:
        print("❌ Array is NOT correctly sorted")
        unsorted_idx = find_first_unsorted(sorted_array)
        print(f"First unsorted element at index {unsorted_idx}")