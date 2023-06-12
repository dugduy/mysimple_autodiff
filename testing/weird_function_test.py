from CGs_grad import *

# def sigmoid(x):
#     return 1/(1+np.e**-x)

# def softmax(x):
#     norm_x=np.e**x
#     return norm_x/reduce_sum(norm_x)

def f(x):
    return reduce_sum(log(x))

Graph().as_default()

x=Variable([1.,2.,3.,4.],'x')
y=f(x)

sess=Session()
print(sess.run(y))
graded=compute_gradients(y,sess.nodes_postorder)

print(graded.get(x))