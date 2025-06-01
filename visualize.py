import pygame as py
import random
import sys
import time
import math

WIDTH = 1400
HEIGHT = 600

WIN = py.display.set_mode((WIDTH, HEIGHT))

FPS = 60

WHITE = "#FFFFFF"
BLACK = "#000000"
RED = "#FF0000"

BG = BLACK
BLOCK_COL = WHITE

PADDING = 30


class Blocks:
    def __init__(self, w, h):
        self.surf = py.Surface((w, h))


def draw_scene(arr):
    WIN.fill(BG)
    num_elem = len(arr)
    block_width = (WIDTH-PADDING) / num_elem

    max_height = HEIGHT - 50
    elem_height = max_height / max(arr)

    running_x = PADDING / 2

    for i in range(num_elem):
        block_height = elem_height*arr[i]
        block = py.Surface((block_width, block_height))
        block.fill(BLOCK_COL)
        WIN.blit(block, (running_x, HEIGHT - 20 - block_height))
        running_x += block_width

    py.display.update()


def swap(arr, j, i):
    tmp = arr[j]
    arr[j] = arr[i]
    arr[i] = tmp


def garbage_sort(arr, asc, delay):
    """randomize till sorted"""
    def check_sorted():
        if asc:
            for i in range(1, len(arr)):
                if arr[i] < arr[i-1]:
                    return False

        else:
            for i in range(1, len(arr)):
                if arr[i] > arr[i-1]:
                    return False
        return True

    while 1:
        draw_scene(arr)
        py.time.wait(delay)
        for event in py.event.get():  # Handle Pygame events
            if event.type == py.QUIT:
                py.quit()
                sys.exit()

        if check_sorted():
            return

        random.shuffle(arr)


def bubble_sort(arr, asc, delay):
    """compares every element n times"""

    for i in range(len(arr)):
        for j in range(1, len(arr)):

            if (asc and arr[j] < arr[j-1]) or (not asc and arr[j] > arr[j-1]):
                swap(arr, j, j-1)
                draw_scene(arr)
                py.time.wait(delay)


def insertion_sort(arr, asc, delay):
    """compares every pair of element until its not succesful"""

    for n in range(1, len(arr)):
        j = n
        if asc:

            while (arr[j] < arr[j-1]) and (j > 0):
                swap(arr, j, j-1)
                draw_scene(arr)
                py.time.wait(delay)
                j -= 1

        else:
            while (arr[j] > arr[j-1]) and (j > 0):
                swap(arr, j, j-1)
                draw_scene(arr)
                py.time.wait(delay)
                j -= 1


def merge_sort(arr, asc, delay):
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

            elif (arr[i] > arr[j] and not asc) or (asc and arr[i] < arr[j]):
                aux[k] = arr[i]
                i += 1

            else:
                aux[k] = arr[j]
                j += 1

            draw_scene(aux)
            py.time.delay(delay)

        arr[lo:hi] = aux[0:n]

    aux = arr.copy()

    sort(arr, 0, len(arr), aux)


def selection_sort(arr, asc, delay):
    """finds min/max in the unsorted portion of the list and brings it to the sorted part"""

    imlpenation = 0

    # implentation 0
    if imlpenation == 0:
        if asc:
            for i in range(len(arr)):
                current_min_idx = i
                j = i
                for j in range(i, len(arr)):
                    if arr[current_min_idx] > arr[j]:
                        current_min_idx = j

                if i != current_min_idx:
                    swap(arr, i, current_min_idx)
                    draw_scene(arr)
                    py.time.wait(delay)

        else:
            for i in range(len(arr)):
                current_max_idx = i
                j = i
                for j in range(i, len(arr)):
                    if arr[current_max_idx] < arr[j]:
                        current_max_idx = j

                if i != current_max_idx:
                    swap(arr, i, current_max_idx)
                    draw_scene(arr)
                    py.time.wait(delay)

    # implentation 1
    else:
        if asc:
            for i in range(len(arr)):
                min_idx = arr.index(min(arr[i:len(arr)]))
                if i != min_idx:
                    swap(arr, i, min_idx)
                    draw_scene(arr)
                    py.time.wait(delay)

        else:
            for i in range(len(arr)):
                max_idx = arr.index(max(arr[i:len(arr)]))
                if i != max_idx:
                    swap(arr, i, max_idx)
                    draw_scene(arr)
                    py.time.wait(delay)


def quick_sort(arr, asc, delay):

    def quicksort(a):

        if len(a) <= 1:
            return a  # Base case: already sorted

        # Choose a pivot element (middle element here)
        mid = len(a) // 2
        pivot = a[mid]

        # Elements smaller than the pivot
        if asc:
            left = [x for x in a if x < pivot]
        else:
            left = [x for x in a if x > pivot]

        # Elements equal to the pivot
        middle = [x for x in a if x == pivot]

        # Elements greater than the pivot
        if asc:
            right = [x for x in a if x > pivot]
        else:
            right = [x for x in a if x < pivot]

        a = quicksort(left) + middle + quicksort(right)

        aux[len(aux) - len(a):] = a
        draw_scene(aux)
        py.time.wait(delay)
        return a

    aux = arr.copy()
    arr = quicksort(arr)
    draw_scene(arr)


def count_sort(arr, asc, delay):

    draw_scene(arr)

    n = len(arr)
    counter = [0 for i in range(n)]

    for i in range(n):
        counter[arr[i] - 1] += 1
        draw_scene(counter)
        pass

    arr_idx = 0
    for i in range(len(counter)):
        for j in range(counter[i]):
            if asc:
                arr[arr_idx] = i+1
            else:
                arr[-arr_idx] = i+1
            arr_idx += 1
            draw_scene(arr)

    draw_scene(arr)


def heap_sort(arr, asc, delay):

    def build_maxheap():
        for i in range(n // 2, -1, -1):
            heapify(i)

    def heapify(parent):

        bigger = parent
        left = 2 * parent
        right = 2 * parent + 1

        # just handeling ascending business check other prgram for implenation
        if (left < n) and ((asc and arr[parent] < arr[left]) or (not asc and arr[parent] > arr[left])):
            bigger = left

        if (right < n) and ((asc and arr[bigger] < arr[right]) or (not asc and arr[bigger] > arr[right])):
            bigger = right

        if parent != bigger:
            swap(arr, parent, bigger)
            draw_scene(arr)
            py.time.wait(delay)
            heapify(bigger)

    n = len(arr)
    build_maxheap()
    for i in range(n-1, -1, -1):
        swap(arr, 0, i)
        draw_scene(arr)
        py.time.wait(delay)
        n -= 1
        heapify(0)


def cocktail_shaker_sort(arr, asc, delay):
    n = len(arr)
    swapped = True

    while 1:
        swapped = False
        for i in range(1, n):
            if (asc and arr[i] < arr[i - 1]) or (not asc and arr[i] > arr[i - 1]):
                swap(arr, i, i - 1)
                draw_scene(arr)
                py.time.wait(delay)
                swapped = True

        if not swapped:
            break

        swapped = False
        for i in range(n - 1, len(arr) - n, -1):
            if (asc and arr[i] < arr[i - 1]) or (not asc and arr[i] > arr[i - 1]):
                swap(arr, i, i - 1)
                draw_scene(arr)
                py.time.wait(delay)
                swapped = True

        if not swapped:
            break

        n -= 1


def gnome_sort(arr, asc, delay):

    for i in range(1, len(arr)):
        swapped = True
        j = i
        while j > 0 and swapped:
            swapped = False
            if (asc and arr[j] < arr[j-1]) or (not asc and arr[j] > arr[j-1]):
                swap(arr, j, j-1)
                draw_scene(arr)
                py.time.delay(delay)
                swapped = True
            j -= 1


def comb_sort(arr, asc, delay):
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
            if (asc and arr[j] > arr[j+gap]) or (not asc and arr[j] < arr[j+gap]):
                swap(arr, j, j+gap)
                draw_scene(arr)
                py.time.wait(delay)
                swapped = True

            j += 1


def gravity_sort(arr, asc, delay):
    """This creates a matrix of 1s and zeroes. 
    each column represents a number in out array, 
    each column has i 1s where arr[col] = i. 
    Then if there is a 0 to the left of a 1 that 1 shifts right. 
    kind of like gravity :), think of tilting an abacus"""

    def draw():
        for i in range(len(arr)):
            arr[i] = sum(beads[i])

        draw_scene(arr)
        py.time.wait(delay)

    biggest = max(arr)
    beads = [[1 if arr[j] > i else 0 for i in range(biggest)]
             for j in range(len(arr))]

    # implementation 1
    # when not swapping then move onto next
    for i in range(len(beads)):

        if asc:
            for col in range(len(beads)-1):
                run = True
                row = biggest - 1
                while row >= 0 and run:
                    if not beads[col][row]:
                        pass
                    elif not beads[col+1][row]:
                        beads[col][row] = 0
                        beads[col+1][row] = 1
                        # draw()
                    else:
                        run = False
                    row -= 1
        else:

            for col in range(len(beads)-1, 0, -1):
                run = True
                row = biggest - 1
                while row >= 0 and run:
                    if not beads[col][row]:
                        pass
                    elif not beads[col-1][row]:
                        beads[col][row] = 0
                        beads[col-1][row] = 1
                        # draw()
                    else:
                        run = False
                    row -= 1
        draw()


def pancake_sort(arr, asc, delay):

    def find_min(start):
        c_min = start
        for i in range(start+1, len(arr)):
            if arr[c_min] > arr[i]:
                c_min = i
        return c_min

    def find_max(start):

        c_max = start
        for i in range(start+1, len(arr)):
            if arr[c_max] < arr[i]:
                c_max = i
        return c_max

    for i in range(len(arr)-1):

        if asc:
            min_idx = find_min(i)

            stack = arr[min_idx:]
            arr[min_idx:] = stack[::-1]
        else:
            max_idx = find_max(i)

            stack = arr[max_idx:]
            arr[max_idx:] = stack[::-1]

        draw_scene(arr)
        py.time.wait(delay)

        unsorted_part = arr[i:]
        arr[i:] = unsorted_part[::-1]

        draw_scene(arr)
        py.time.wait(delay)


def dbl_selection_sort(arr, asc, delay):

    def draw():
        draw_scene(arr)
        py.time.wait(delay)

    lo = 0
    hi = len(arr)-1

    while hi > lo:

        current_min_idx = lo
        current_max_idx = hi

        for i in range(lo, hi+1):
            if arr[i] < arr[current_min_idx]:
                current_min_idx = i

            if arr[i] > arr[current_max_idx]:
                current_max_idx = i

        if not asc:
            lo, hi = hi, lo

        if lo != current_min_idx:
            swap(arr, lo, current_min_idx)

            if current_max_idx == lo:
                current_max_idx = current_min_idx

        if hi != current_max_idx:
            swap(arr, hi, current_max_idx)

        if not asc:
            lo, hi = hi, lo

        lo += 1
        hi -= 1

        draw()


def bitonic_sort_ntwrk(arr, asc, delay):
    """A network or series of comparisons that if followed will always sort the array, 
    (this network was thought by someone very clever),
    This only works on arrays of size 2^n
    (Can be adapdted but to something more general but too complex, i think),
    The performance of this is bad, but using networks allows for parrallel comparisons and swapping
    This makes this the fastest (i think) if we use parrallel computing cause we are doing way less
    comparisons per cycle
    This was thought up by some dude named Ken Batcher in 1968"""

    def draw():
        draw_scene(arr)
        py.time.wait(delay)

    def box1(n):

        start_idx = 0
        step = 2*n

        idx = start_idx

        while start_idx + step - 1 < len(arr):
            idx += step - 1

            gap = step - 1

            for i in range(n):
                if (asc and arr[idx - gap] > arr[idx]) or (not asc and arr[idx - gap] < arr[idx]):
                    swap(arr, idx, idx-gap)
                    draw()
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
                if (asc and arr[idx - gap] > arr[idx]) or (not asc and arr[idx - gap] < arr[idx]):
                    swap(arr, idx, idx-gap)
                    draw()
                idx -= 1

            start_idx += step
            idx = start_idx

    draw()

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


def exchange_sort(arr, asc, delay):

    if asc:
        for root in range(len(arr)):
            current_min_idx = root
            for j in range(root, len(arr)):
                if arr[current_min_idx] > arr[j]:

                    swap(arr, j, root)
                    draw_scene(arr)
                    py.time.wait(delay)
    else:
        for root in range(len(arr)):
            current_max_idx = root
            for j in range(root, len(arr)):
                if arr[current_max_idx] < arr[j]:

                    swap(arr, j, root)
                    draw_scene(arr)
                    py.time.wait(delay)


def odd_even_sort(arr, asc, delay):

    draw_scene(arr)

    swapped = True
    odd_even_mode = 0

    while swapped:

        swapped = False
        idx = odd_even_mode

        while idx+1 < len(arr):
            if (asc and arr[idx] > arr[idx+1]) or (not asc and arr[idx] < arr[idx+1]):
                swapped = True
                swap(arr, idx, idx+1)
                draw_scene(arr)
                py.time.wait(delay)
            idx += 2

        odd_even_mode = not odd_even_mode


def visualize(sort_func, asc, delay, num_elems, random_array):

    clock = py.time.Clock()
    if random_array:
        my_arr = [random.randint(0, num_elems) for i in range(num_elems)]
    else:
        my_arr = [i+1 for i in range(num_elems)]

    random.shuffle(my_arr)

    done = False
    while 1:

        for event in py.event.get():
            if event.type == py.QUIT:
                py.quit()
                sys.exit()

        # py.display.update()
        clock.tick(FPS)
        if not done:
            start = time.time()
            sort_func(my_arr, asc=asc, delay=delay)
            end = time.time()

            print("Done")
            print(my_arr)
            print(f"Ok, {str(sort_func).split()[1].title()} -> {end-start}s")
            done = True


def main():

    sort_funcs = {0: garbage_sort,
                  1: bubble_sort,
                  2: insertion_sort,
                  3: merge_sort,
                  4: selection_sort,
                  5: quick_sort,
                  6: count_sort,
                  7: heap_sort,
                  8: cocktail_shaker_sort,
                  9: gnome_sort,
                  10: comb_sort,
                  11: gravity_sort,
                  12: pancake_sort,
                  13: dbl_selection_sort,
                  14: bitonic_sort_ntwrk,
                  15: exchange_sort,
                  16: odd_even_sort }

    visualize(sort_func=sort_funcs[3],
              asc=True,
              delay=0,
              num_elems=400,  # 1370 is max
              random_array=False)


if __name__ == "__main__":
    main()
