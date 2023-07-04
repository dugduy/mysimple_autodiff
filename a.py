from CGs import *
from CGs_grad import gradients

Graph().as_default()

a=Variable([5,6],'a')
b=tile(a,(2,3),name='b')

print(b)
print(gradients(b).get(a))