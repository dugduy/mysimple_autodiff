from CGs_grad import *

xs=[-3,-2,-1,0,1,2,3,4,5,6]
ys=[-18,-13,-8,-3,2,7,12,17,22,27]

Graph().as_default()
w=Variable(0,'w')
b=Variable(0,'b')
sess=Session()

for i in range(500):
    tt_loss=0
    for x_batch,y_batch in zip(xs,ys):
        # x=PlaceHolder('x') this is suck!!!
        x=Variable(x_batch,'x')
        y_pred=x*w+b
        loss=(y_pred-y_batch)**2
        tt_loss+=loss
        
    sess.run(tt_loss)
    print('Epoch',i,'loss:',tt_loss)
    graded=compute_gradients(tt_loss,sess.nodes_postorder)
    w=Variable(w.value-graded.get(w)*0.001,'w')
    b=Variable(b.value-graded.get(b)*0.001,'b')
x=PlaceHolder('x')
y=x*w+b
print(sess.run(y,{x:12}))
# we can use pickle to save these paramts