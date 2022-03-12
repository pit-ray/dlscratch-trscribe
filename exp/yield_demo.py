
def t1():
    print('pre')
    try:
        return 1
    finally:
        print('post')


def t2():
    print('pre')
    try:
        yield 2
    finally:
        print('post')


print(t1())

print(next(t2()))
