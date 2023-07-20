import os
from multiprocessing import Pool

pi = 3.1415926535


def f(x):
    print('module name:', __name__)
    print('parent process:', os.getppid())
    print('process id:', os.getpid())
    a = [(x + y + pi) ** .5 for y in range(100000000)]
    return a


if __name__ == '__main__':
    with Pool(5) as p:
        res = p.map(f, [1, 2, 3])
        for item in res:
            print(len(item))
