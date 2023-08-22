from .CGs import Variable,np
# array creations
def ones(shape,dtype=None,name=''):
    return Variable(np.ones(shape,dtype=dtype),dtype,name)
def zeros(shape,dtype=None,name=''):
    return Variable(np.zeros(shape,dtype),dtype,name)
def full(shape,fill_value,dtype=None,name=''):
    return Variable(np.full(shape,fill_value,dtype),dtype,name)
def ones_like(arr_like,dtype=None,name=''):
    return Variable(np.ones_like(arr_like.value,dtype),dtype,name)
def zeros_like(arr_like,dtype=None,name=''):
    return Variable(np.zeros_like(arr_like.value,dtype),dtype,name)
def full_like(arr_like,fill_value,dtype=None,name=''):
    return Variable(np.full_like(arr_like.value,fill_value,dtype),dtype,name)
# random function
def randn(shape,dtype=None,name=''):
    return Variable(np.random.randn(*shape),dtype,name)
def randint(low,high=None,size=None,dtype=None,name=''):
    return Variable(np.random.randint(low,high,size,dtype),dtype,name)
def rand(shape,dtype=None,name=''):
    return Variable(np.random.rand(*shape),dtype,name)
def random(size=None,dtype=None,name=''):
    return Variable(np.random.random(size),dtype,name)
def random_sample(size=None,dtype=None,name=''):
    return Variable(np.random.random_sample(size),dtype,name)
# ...