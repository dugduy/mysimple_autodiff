from typing import Any
import numpy as np 
class Operation:
    def __init__(self,input_nodes=[], name='') -> None:
        self.input_nodes=input_nodes
        self.name=name
    def compute(self):
        pass

class InitOp(Operation):
    def __init__(self, name='') -> None:
        self.name=name
    def compute(self,val):
        return val

class Neg(Operation):
    def __init__(self, a, name='') -> None:
        super().__init__([a], name)
    def compute(self,a_val):
        return -a_val

class Add(Operation):
    def __init__(self, a,b, name='') -> None:
        super().__init__([a,b], name)
    def compute(self, x_val, y_val):
        return x_val+y_val

class Sub(Add):
    def compute(self, x_val, y_val):
        return x_val-y_val

class Mul(Add):
    def compute(self, x_val, y_val):
        return x_val*y_val
    
class Matmul(Add):
    def compute(self, x_val, y_val):
        return x_val@y_val

class Div(Add):
    def compute(self, x_val, y_val):
        return x_val/y_val

class Pow(Add):
    def compute(self, x_val, y_val):
        return x_val**y_val
    
class Reduce_sum(Operation):
    def __init__(self, A, axis=None, keep_dims=False, name='') -> None:
        super().__init__([A], name)
        self.axis=axis
        self.keep_dims=keep_dims
    def compute(self,A_val):
        return np.sum(A_val,self.axis,keepdims=self.keep_dims)
    
class Log(Neg):
    def compute(self, x_val):
        return np.log(x_val)

class Expandim(Operation):
    def __init__(self, A,axis, name='') -> None:
        super().__init__([A], name)
        self.axis=axis
    def compute(self,A_val):
        return np.expand_dims(A_val,self.axis)

class Tile(Operation):
    def __init__(self, A,reps, name='') -> None:
        super().__init__([A],name)
        self.reps=reps
    def compute(self,A_val):
        return np.tile(A_val,self.reps)

class Cast(Operation):
    def __init__(self, A,dtype='float32', name='') -> None:
        super().__init__([A], name)
        self.dtype=dtype
    def compute(self,A_val):
        return A_val.astype(self.dtype)

class Concate(Operation):
    def __init__(self, input_arrs,axis, name='') -> None:
        super().__init__(input_arrs, name)
        self.axis=axis
    def compute(self,input_values):
        return np.concatenate(input_values,self.axis)

class Transpose(Operation):
    def __init__(self, A,new_dim_index, name='') -> None:
        super().__init__([A], name)
        self.new_dim_index=new_dim_index
    def compute(self,A_val):
        return np.transpose(A_val,self.new_dim_index)

class Maximum(Operation):
    def __init__(self, A,B, name='') -> None:
        super().__init__([A,B], name)
    def compute(self,A_val,B_val):
        return np.maximum(A_val,B_val)

class Minimum(Operation):
    def __init__(self, A,B, name='') -> None:
        super().__init__([A,B], name)
    def compute(self,A_val,B_val):
        return np.minimum(A_val,B_val)
class Reshape(Operation):
    def __init__(self, A,new_shape, name='') -> None:
        super().__init__([A], name)
        self.newshape=new_shape
    def compute(self,A_val):
        return np.reshape(A_val,self.newshape)

def cgsfunc(func):
    def wrapper(*args,**kvagrs):
        input_nodes=[]
        for arg in args:
            if not type(arg)==Variable:
                input_nodes.append(Variable(arg))
            else:
                input_nodes.append(arg)
        return func(*input_nodes,**kvagrs)
    return wrapper

class Variable:
    def __init__(self,init_val, name='',ops=None) -> None:
        self.name=name
        self.value=np.array(init_val)
        self.ops=ops
        if ops is None:
            self.ops=InitOp('init_ops')
    def __getattr__(self, __name: str) -> Any:
        if __name=='shape':
            return self.value.shape
        elif __name=='ndim':
            return self.value.ndim
        elif __name=='T':
            dims=list(range(self.ndim))
            dims[-2],dims[-1]=dims[-1],dims[-2]
            return transpose(self,new_dim_index=dims)
        else:
            raise AttributeError(__name)
    def astype(self,dtype='float16'):
        return Variable(self.value.astype(dtype),self.name,ops=self.ops)
    def assign(self,value):
        if type(value)==Variable:
            value=value.value
        self.value=value
    def assign_add(self,add_val):
        if type(add_val)==Variable:
            add_val=add_val.value
        self.value+=add_val
    def assign_sub(self,sub_val):
        self.assign_add(-sub_val)
    def __add__(self,other):
        return add(self,other)
    def __radd__(self,other):
        return self+other
    def __sub__(self,other):
        return sub(self,other)
    def __rsub__(self,other):
        return -self+other
    def __mul__(self,other):
        return mul(self,other)
    def __rmul__(self,other):
        return self*other
    def __matmul__(self,other):
        return matmul(self,other)
    def __rmatmul__(self,other):
        return matmul(other,self)
    def __truediv__(self,other):
        return div(self,other)
    def __rtruediv__(self,other):
        return div(Variable(other),self)
    def __pow__(self,other):
        return pow(self,other)
    def __rpow__(self,other):
        return pow(other,self)
    def __neg__(self):
        return neg(self)
    # def __str__(self) -> str:
    #     return str(self.value)
    @cgsfunc
    def __lt__(self,other):
        return Variable(self.value<other.value)
    @cgsfunc
    def __gt__(self,other):
        return Variable(self.value>other.value)
    @cgsfunc
    def __eq__(self,other):
        return Variable(self.value==other.value)
    @cgsfunc
    def __le__(self,other):
        return Variable(self.value<=other.value)
    @cgsfunc
    def __ge__(self,other):
        return Variable(self.value>=other.value)
    @cgsfunc
    def __ne__(self,other):
        return Variable(self.value!=other.value)
    @cgsfunc
    def __mod__(self,other):
        return Variable(self.value%other.value)
    def __hash__(self):
        return hash((self.name,self.ops))
    def __getitem__(self,name):
        if type(name)==Variable:
            name=tuple(name.value.tolist())
        return Variable(self.value.__getitem__(name))
    def __repr__(self) -> str:
        return f'Variable({self.value},name="{self.name}")'


@cgsfunc
def reduce_sum(A,axis=None,keep_dims=False,name=''):
    sum_obj=Reduce_sum(A,axis,keep_dims,name)
    return Variable(sum_obj.compute(A.value),name,sum_obj)
@cgsfunc
def neg(a,name=''):
    neg_obj=Neg(a,name+'_ops')
    return Variable(neg_obj.compute(a.value),name,neg_obj)
@cgsfunc
def log(a,name=''):
    log_obj=Log(a,name+'_ops')
    return Variable(log_obj.compute(a.value),name,log_obj)
@cgsfunc
def add(a,b,name=''):
    add_obj=Add(a,b,name+'_ops')
    return Variable(add_obj.compute(a.value,b.value),name,add_obj)
@cgsfunc
def sub(a,b,name=''):
    sub_obj=Sub(a,b,name+'_ops')
    return Variable(sub_obj.compute(a.value,b.value),name,sub_obj)
@cgsfunc
def mul(a,b,name=''):
    mul_obj=Mul(a,b,name+'_ops')
    return Variable(mul_obj.compute(a.value,b.value),name,mul_obj)
@cgsfunc
def div(a,b,name=''):
    div_obj=Div(a,b,name+'_ops')
    return Variable(div_obj.compute(a.value,b.value),name,div_obj)
@cgsfunc
def pow(a,b,name=''):
    pow_obj=Pow(a,b,name+'_ops')
    return Variable(pow_obj.compute(a.value,b.value),name,pow_obj)
@cgsfunc
def matmul(A,B,name=''):
    matmul_obj=Matmul(A,B,name+'_ops')
    return Variable(matmul_obj.compute(A.value,B.value),name,matmul_obj)
@cgsfunc
def expand_dims(A,axis=0,name=''):
    exp_dim_obj=Expandim(A,axis,name+'_ops')
    return Variable(exp_dim_obj.compute(A.value),name,exp_dim_obj)
@cgsfunc
def tile(A,reps,name=''):
    tile_obj=Tile(A,reps,name+'_ops')
    return Variable(tile_obj.compute(A.value),name,tile_obj)
@cgsfunc
def cast(A,dtype='float',name=''):
    cast_obj=Cast(A,dtype,name+'_ops')
    return Variable(cast_obj.compute(A.value),name,cast_obj)
# @cgsfunc
# def concate(*arrs,axis=0,name=''):
#     cc_obj=Concate(arrs,axis,name)
#     return Variable(cc_obj.compute([arr.value for arr in arrs]),name,cc_obj)
@cgsfunc
def transpose(A,new_dim_index,name=''):
    T_obj=Transpose(A,new_dim_index,name+'_ops')
    return Variable(T_obj.compute(A.value),name,T_obj)
@cgsfunc
def constant(A,name=''):
    init_obj=InitOp(name+'_ops')
    return Variable(init_obj.compute(A),name,init_obj)
@cgsfunc
def maximum(A,B,name=''):
    maximum_obj=Maximum(A,B,name+'_ops')
    return Variable(maximum_obj.compute(A.value,B.value),name,maximum_obj)
@cgsfunc
def minimum(A,B,name=''):
    minimum_obj=Minimum(A,B,name+'_ops')
    return Variable(minimum_obj.compute(A.value,B.value),name,minimum_obj)
@cgsfunc
def reshape(A,newshape,name=''):
    reshaper=Reshape(A,newshape,name+'_ops')
    return Variable(reshaper.compute(A.value),name,reshaper)
@cgsfunc
def clip(A,min,max,name=''):
    return maximum(minimum(max,A),min)

def traverse_postorder(var_obj):
    nodes_postorder=[]
    def recurse(node):
        if hasattr(node.ops,'input_nodes'):
            for input_node in node.ops.input_nodes:
                recurse(input_node)
        nodes_postorder.append(node)
    recurse(var_obj)
    return nodes_postorder