from CGs_grad import *

Graph().as_default()

x=Variable(3,'x')
y=(4*x**2 - 3*x + 2)/(x - 2)

sess=Session()
print(sess.run(y))

graded=compute_gradients(y,sess.nodes_postorder)
print(graded.get(x))