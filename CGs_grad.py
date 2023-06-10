from queue import Queue
from CGs import *
_gradient_registry={}
class RegGrad:
    def __init__(self,op_type) -> None:
        self._optype=eval(op_type)
    def __call__(self,f):
        _gradient_registry[self._optype]=f

@RegGrad('neg')
def _neg_gradient(op,grad):
    return -grad

@RegGrad('add')
def _add_gradient(op,grad):
    return grad,grad

@RegGrad('sub')
def _sub_gradient(op,grad):
    return grad,-grad

@RegGrad('mul')
def _mul_gradient(op,grad):
    a,b=op.inputs
    return grad*b,grad*a

@RegGrad('matmul')
def _matmul_gradient(op,grad):
    A,B=op.inputs
    return grad@B.T,A.T@grad

@RegGrad('div')
def _div_gradient(op,grad):
    a,b=op.inputs
    return grad/b,-grad*a/b**2

@RegGrad('pow')
def _pow_gradient(op,grad):
    a,b=op.inputs
    # return grad*b*a**(b-1),1#,grad*op.output*np.log(a)
    return grad*b*a**(b-1),grad*op.output*np.log(a)


def compute_gradients(op,steps=[]):
    grad_table={op:np.ones_like(op.output)}
    q=Queue()
    q.put(op)
    while not q.empty():
        node = q.get()
        if node!=op:
            grad_table[node]=0
            for consumer in node.consumers:
                if consumer in steps:
                    lossgrad_wrt_input=_gradient_registry[consumer.__class__](consumer,grad_table.get(consumer,0))
                    if len(consumer.input_nodes)>1:
                        lossgrad_wrt_input=lossgrad_wrt_input[consumer.input_nodes.index(node)]
                    grad_table[node]+=lossgrad_wrt_input
        
        if hasattr(node,'input_nodes'):
            for i in node.input_nodes:
                q.put(i)
    return grad_table