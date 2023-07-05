from CGs import *
_gradient_registry={}
class RegGrad:
    def __init__(self,op_type) -> None:
        self._optype=eval(op_type)
    def __call__(self,f):
        def wrapper(op,grad):
            raw_grad=f(op,grad)
            final_grad=[]
            for input_node,crgrad in zip(op.input_nodes,raw_grad):
                if crgrad.shape!=input_node.shape:                        
                        while crgrad.ndim > input_node.ndim:
                            crgrad=reduce_sum(crgrad,axis=0)
                        while crgrad.ndim < input_node.ndim:
                            crgrad=expand_dims(crgrad,axis=0)
                        for axis, size in enumerate(crgrad.shape):
                            if size==1:
                                crgrad=reduce_sum(crgrad,axis=axis,keep_dims=1)
                        crgrad=tile(crgrad,reps=np.array(input_node.shape)//crgrad.shape)
                final_grad.append(crgrad)
            return final_grad
        _gradient_registry[self._optype]=wrapper
        return wrapper

@RegGrad('Neg')
def _neg_gradient(op,grad):
    return [-grad]

@RegGrad('Add')
def _add_gradient(op,grad):
    return grad,grad

@RegGrad('Sub')
def _sub_gradient(op,grad):
    return grad,-grad

@RegGrad('Mul')
def _mul_gradient(op,grad):
    a,b=op.input_nodes[0],op.input_nodes[1]
    return grad*b,grad*a

@RegGrad('Matmul')
def _matmul_gradient(op,grad):
    A,B=op.input_nodes[0],op.input_nodes[1]
    return grad@B.T,A.T@grad

@RegGrad('Div')
def _div_gradient(op,grad):
    a,b=op.input_nodes[0],op.input_nodes[1]
    return grad/b,-grad*a/b**2

@RegGrad('Pow')
def _pow_gradient(op,grad):
    a,b=op.input_nodes[0],op.input_nodes[1]
    return grad*b*a**(b-1),grad*log(a)*a**b

@RegGrad('Log')
def _log_gradient(op,grad):
    return [grad/op.input_nodes[0]]

@RegGrad('Reduce_sum')
def _reduce_sum_gradient(op,grad):
    A=op.input_nodes[0]
    output_shape=np.array(A.shape)
    output_shape[op.axis]=1
    grad=np.reshape(grad,output_shape)
    return [grad]

@RegGrad('Expandim')
def _expandim_gradient(op,grad):
    A=op.input_nodes[0]
    return [grad.reshape(A.shape)]

@RegGrad('Tile')
def _tile_gradient(op,grad):
    A=op.input_nodes[0]
    return [grad.reshape(op.reps+A.shape)]
@RegGrad('Cast')
def _cast_gradient(op,grad):
    return [cast(grad,dtype=op.dtype)]
@RegGrad('Transpose')
def _transpose_gradient(op,grad):
    reT=np.zeros(len(op.new_dim_index),'int')
    for i, dim in enumerate(op.new_dim_index):
        reT[dim]=i
    return [transpose(grad,new_dim_index=reT)]
@RegGrad('Maximum')
def _maximum_gradient(op,grad):
    A,B=op.input_nodes
    where_a_gt_b=A>B
    return grad*where_a_gt_b.value,grad*~where_a_gt_b.value
@RegGrad('Minimum')
def _minimum_gradient(op,grad):
    A,B=op.input_nodes
    where_a_lt_b=A<B
    return grad*where_a_lt_b.value,grad*~where_a_lt_b.value

def gradients(target_var):
    grad_dict={target_var:Variable(np.ones_like(target_var.value))}
    steps=traverse_postorder(target_var)
    seen=set()
    steps=[i for i in steps if hasattr(i.ops,'input_nodes') and not (i in seen or seen.add(i))]
    steps.reverse()

    for node in steps:
        grad_fn=_gradient_registry[node.ops.__class__]
        grads=grad_fn(node.ops,grad_dict[node])
        for input_node,grad in zip(node.ops.input_nodes,grads):
            grad_dict[input_node]=grad_dict.get(input_node,0)+grad
    return grad_dict