
import random
import numpy as np
import time
import math


def swap(arr, j, i):
    tmp = arr[j]
    arr[j] = arr[i]
    arr[i] = tmp


def bubble_sort(arr):
    """compares every element n times"""

    for i in range(len(arr)):
        for j in range(1, len(arr)):

            if (arr[j] < arr[j-1]):
                swap(arr, j, j-1)


def insertion_sort(arr):
    """compares every pair of element until its not succesful"""

    for n in range(1, len(arr)):
        j = n
        while (arr[j] < arr[j-1]) and (j > 0):
            swap(arr, j, j-1)
            j -= 1


def merge_sort(arr):
    """splits list into lots of portions, then merges all them sorting in the process"""

    def sort(arr, lo, hi, aux):
        if (hi - lo) <= 1:
            return

        mid = (hi + lo) // 2

        sort(arr, lo, mid, aux)
        sort(arr, mid, hi, aux)
        merge(arr, lo, mid, hi, aux)

    def merge(arr, lo, mid, hi, aux):
        n = hi - lo

        i = lo
        j = mid

        for k in range(n):
            if i == mid:
                aux[k] = arr[j]
                j += 1
            elif j == hi:
                aux[k] = arr[i]
                i += 1

            elif (arr[i] < arr[j]):
                aux[k] = arr[i]
                i += 1

            else:
                aux[k] = arr[j]
                j += 1

        arr[lo:hi] = aux[0:n]

    aux = arr.copy()

    sort(arr, 0, len(arr), aux)


def selection_sort(arr):
    """finds min/max in the unsorted portion of the list and brings it to the sorted part"""

    n = len(arr)
    for i in range(n):
        current_min_idx = i
        j = i

        for j in range(i+1, n):
            if arr[current_min_idx] > arr[j]:
                current_min_idx = j

        if i != current_min_idx:
            swap(arr, i, current_min_idx)


def quick_sort(a):

    n = len(a)
    if n <= 1:
        return a  # Base case: already sorted

    # Choose a pivot element (middle element here)
    pivot = a[n // 2]

    # Elements smaller than the pivot
    left = [x for x in a if x < pivot]

    # Elements equal to the pivot
    middle = [x for x in a if x == pivot]

    # Elements greater than the pivot
    right = [x for x in a if x > pivot]

    return quick_sort(left) + middle + quick_sort(right)


def count_sort(arr):

    # implentation 0
    counter = [0 for i in range(max(arr) + 1)]
    for elem in arr:
        counter[elem] += 1

    arr_idx = 0
    for i in range(len(counter)):
        for j in range(counter[i]):
            arr[arr_idx] = i
            arr_idx += 1

    # implentation 1
    # np_counter = np.zeros(max(arr)+1, dtype=int)
    # for elem in arr:
    #     np_counter[elem] += 1

    # arr_idx = 0
    # for i in range(np_counter.size):
    #     for j in range(np_counter[i]):
    #         arr[arr_idx] = i
    #         arr_idx += 1


def check_sort(arr):

    for i in range(1, len(arr)):
        if arr[i] < arr[i-1]:
            return False
    return True


def heap_sort(arr):
    def build_maxheap():
        for i in range(n // 2, -1, -1):
            heapify(i)

    def heapify(parent):

        bigger = parent
        left = 2 * parent
        right = 2 * parent + 1

        if (left < n) and (arr[parent] < arr[left]):
            bigger = left

        if (right < n) and (arr[bigger] < arr[right]):
            bigger = right

        if parent != bigger:
            swap(arr, parent, bigger)
            heapify(bigger)

    n = len(arr)
    build_maxheap()
    for i in range(n-1, -1, -1):
        swap(arr, 0, i)
        n -= 1
        heapify(0)


def cocktail_shaker_sort(arr):
    n = len(arr)

    while 1:
        swapped = False
        for i in range(1, n):
            if arr[i] < arr[i - 1]:
                swap(arr, i, i - 1)
                swapped = True

        if not swapped:
            break

        swapped = False
        for i in range(n - 1, 0, -1):
            if arr[i] < arr[i - 1]:
                swap(arr, i, i - 1)
                swapped = True

        if not swapped:
            break

        n -= 1


def gnome_sort(arr):

    for i in range(1, len(arr)):
        swapped = True
        j = i
        while j > 0 and swapped:
            swapped = False
            if arr[j] < arr[j-1]:
                swap(arr, j, j-1)
                swapped = True
            j -= 1


def comb_sort(arr):
    """Bubble sort but doesnt compare adjacent elements compares specific gap size"""

    n = len(arr)
    gap = n
    swapped = True
    while swapped:
        if gap != 1:
            gap = int(gap // 1.3)
        j = 0
        swapped = False
        while j+gap < n:
            if (arr[j] > arr[j+gap]):
                swap(arr, j, j+gap)
                swapped = True
            j += 1


def gravity_sort(arr):

    biggest = max(arr)
    beads = [[1 if arr[j] > i else 0 for i in range(biggest)]
             for j in range(len(arr))]

    # implementation 0
    # cycle through every element
    # for i in range(len(beads)):
    #     for col in range(len(beads)-1):
    #         for row in range(len(beads[col])):
    #             if beads[col][row] and not beads[col+1][row]:
    #                 beads[col][row] = 0
    #                 beads[col+1][row] = 1

    # implementation 1
    # when not swapping then move onto next
    for i in range(len(beads)):

        for col in range(len(beads)-1):
            run = True
            row = biggest - 1
            while row >= 0 and run:
                if not beads[col][row]:
                    pass
                elif not beads[col+1][row]:
                    beads[col][row] = 0
                    beads[col+1][row] = 1
                else:
                    run = False
                row -= 1

    for i in range(len(arr)):
        arr[i] = sum(beads[i])


def pancake_sort(arr):

    def find_min(start):
        c_min = start
        for i in range(start+1, len(arr)):
            if arr[c_min] > arr[i]:
                c_min = i
        return c_min

    for i in range(len(arr)-1):

        min_idx = find_min(i)

        stack = arr[min_idx:]

        arr[min_idx:] = stack[::-1]
        unsorted_part = arr[i:]

        arr[i:] = unsorted_part[::-1]


def dbl_selection_sort(arr):

    lo = 0
    hi = len(arr)-1

    while hi > lo:

        current_min_idx = lo
        current_max_idx = hi

        # find min and max at same time
        for i in range(lo, hi+1):
            if arr[i] < arr[current_min_idx]:
                current_min_idx = i
            if arr[i] > arr[current_max_idx]:
                current_max_idx = i

        # if min was at lo already then dont swap
        if lo != current_min_idx:
            swap(arr, lo, current_min_idx)

            # if max was now at lo and lo has now just changed
            if current_max_idx == lo:
                current_max_idx = current_min_idx

        if hi != current_max_idx:
            swap(arr, hi, current_max_idx)

        lo += 1
        hi -= 1


def bitonic_sort_ntwrk(arr):
    """A network or series of comparisons that if followed will always sort the array, 
    (this network was thought by someone very clever),
    This only works on arrays of size 2^n
    (Can be adapdted but adis i think),
    The performance of this is bad, but using networks allows for parrallel comparisons and swapping
    This makes this the fastest (i think) if we use parrallel computing cause we are doing way less
    comparisons per cycle
    This was thought up by some dude named Ken Batcher in 1968"""

    def box1(n):

        start_idx = 0
        step = 2*n

        idx = start_idx

        while start_idx + step - 1 < len(arr):
            idx += step - 1

            gap = step - 1

            for i in range(n):
                if arr[idx - gap] > arr[idx]:
                    swap(arr, idx, idx-gap)

                idx -= 1
                gap -= 2

            start_idx += step
            idx = start_idx

    def box2(n):

        start_idx = 0
        step = 2*n

        gap = n
        idx = start_idx

        while start_idx + step - 1 < len(arr):
            idx = idx + step - 1

            for i in range(n):
                if arr[idx - gap] > arr[idx]:
                    swap(arr, idx, idx-gap)

                idx -= 1

            start_idx += step
            idx = start_idx

    boxes = [box1, box2]

    iters = int(math.log2(len(arr)))

    for i in range(iters):

        for j in range(i+1):
            box1_idx = pow(2, i)
            boxes[0](box1_idx)

            box2_idx = box1_idx // 2

            while box2_idx >= 1:
                boxes[1](box2_idx)

                box2_idx //= 2


def exchange_sort(arr):

    for root in range(len(arr)):
        current_min_idx = root
        for j in range(root, len(arr)):
            if arr[current_min_idx] > arr[j]:
                swap(arr, j, root)


def odd_even_sort(arr):

    swapped = True
    odd_even_mode = 0

    while swapped:

        swapped = False
        idx = odd_even_mode

        while idx+1 < len(arr):
            if arr[idx] > arr[idx+1]:
                swapped = True
                swap(arr, idx, idx+1)

            idx += 2

        odd_even_mode = not odd_even_mode


def test(unsorted_arr):

    sort_funcs = {
        0: bubble_sort,
        1: insertion_sort,
        2: merge_sort,
        3: selection_sort,
        4: quick_sort,
        5: count_sort,
        6: heap_sort,
        7: cocktail_shaker_sort,
        8: gnome_sort,
        9: comb_sort,
        # 10: gravity_sort,  # this ones rlly bad
        11: pancake_sort,
        12: dbl_selection_sort,
        13: bitonic_sort_ntwrk,
        14: exchange_sort,
        15: odd_even_sort
    }

    stats_dict = {}

    for func in sort_funcs.values():
        test_arr = unsorted_arr.copy()

        if str(func).split()[1].title() == "Quick_Sort":
            start = time.time()
            test_arr = func(test_arr)
            end = time.time()
        else:
            start = time.time()
            func(test_arr)
            end = time.time()

        print(f"{str(func).split()[1].title():<20} -> Completed")
        success = check_sort(test_arr)

        stats_dict[str(func).split()[1].title()] = {"Time": end-start,
                                                    "Success": str(bool(success))}

    stats_dict = dict(
        sorted(stats_dict.items(), key=lambda item: item[1]["Time"]))

    placement = 1
    print(f"{'Placement':<15}{'Function':<25}{'Success':<15}{'Time'}\n")
    for stat_tuple in stats_dict.items():

        print(
            f"{placement:<15}{stat_tuple[0]:<25}{stat_tuple[1]['Success']:<15}{stat_tuple[1]['Time'] * 1000} ms")
        placement += 1


def main():

    arr1 = [random.randint(0, 10_000) for i in range(10_000)]
    test(arr1)


if __name__ == "__main__":
    main()
