from CGs_grad import *

Graph().as_default()

x=Variable(5,'x')
y=x**2-1/x+1
z=2*y-1

sess=Session()
print(sess.run(z))

graded=compute_gradients(z,sess.nodes_postorder)
print(graded.get(x))