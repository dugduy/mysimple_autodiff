def mydeco(func):
    def wrapper(c,d):
        print('ASD!!!')
        return func(c,d)
    return wrapper

@mydeco
def asd(a,b):
    return a+b

print(asd(4,5))