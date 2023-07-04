from CGs import *
from CGs_grad import gradients

Graph().as_default()

a=Variable(5.)
b=a*4-1/a
print(gradients(b).get(a))