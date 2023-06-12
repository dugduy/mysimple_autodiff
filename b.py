from CGs_grad import *

Graph().as_default()

# x=Variable([[1,2],[3,4],[5,6]],'x')
# y=reduce_sum(x,1,name='y')
x=Variable(
    [
        [1,2,3,4],
        [5,6,7,8],
        [9,10,11,12]
    ]
)
z=Variable(
    [1,2,3,4]
)

y=x*z

sess=Session()
print(sess.run(y))

print(compute_gradients(y,sess.nodes_postorder)[z])