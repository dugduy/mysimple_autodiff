from CGs_grad import *

Graph().as_default()

A=Variable(
    [
        [1,2,3,4],
        [5,6,7,8],
        [9,10,11,12]
    ],
    'A'
)

B=Variable(
    [
        [1,2],
        [3,4],
        [5,6],
        [7,8]
    ]
)

C=A@B

sess=Session()
output=sess.run(C)
print(output)
graded=compute_gradients(C,sess.nodes_postorder)
print(graded.get(A))
print(graded.get(B))