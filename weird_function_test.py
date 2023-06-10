from CGs_grad import *

def sigmoid(x):
    return 1/(1+np.e**-x)

Graph().as_default()

x=Variable([0,1,2,3,4],'x')
y=sigmoid(x)

sess=Session()
print(sess.run(y))
graded=compute_gradients(y,sess.nodes_postorder)

print(graded.get(x))