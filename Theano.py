##Neural Network, Theano
##4 basic
import numpy as np
import theano.tensor as T
from theano import function

#basic
x = T.dscalar('x')
y = T.dscalar('y')
z = x + y
f = function([x,y],z) #input output

print(f(2,4))
#could use usual one, by above support GPU Parallel

#pretty printer, see the function
from theano import pp
print(pp(z))

#matrix
x = T.dmatrix('x') #float64 T.fmatrix() 32
y = T.dmatrix('y')
z = x+y #T.dot(x,y) times
f = function([x,y],z)
print(f(np.arange(12).reshape((3,4)),
	10*np.ones((3,4))))



##5 function
import theano
#activation function
x = T.dmatrix('x')
s = 1/(1+T.exp(-x)) #use theano inside not np.exp()
logistic = theano.function([x],s)
print(logistic([[0,1],[-2,-3]]))

#multiply output
a,b = T.dmatrices('a','b')
diff = a-b
abs_diff = abs(diff)
squ_diff = diff**2
f = theano.function([a,b],[diff,abs_diff,squ_diff])
print(
	f(np.ones((2,2)),
	np.arange(4).reshape((2,2))))

x1,x2,x3 = f(np.ones((2,2)),
	np.arange(4).reshape((2,2)))

#name of function
x,y,w = T.dscalars('x','y','w')
z = (x+y)*w
f = theano.function([x,
	theano.In(y,value=1),
	theano.In(w,value=2,name='weights')],
	z)
#y as default f(a,b=1,c)
print(f(23,)) #(23+1)*2 or f(23,2,weights=4) =100



##6 share value
#renewable variable

state = theano.shared(np.array(0,dtype=np.float64),	'state')
#make sure all dtype same after
inc = T.scalar('inc',dtype=state.dtype) #assign as above
accumulator = theano.function([inc],state,updates=[(state,state+inc)]) #state=state+inc
print(accumulator(10)) #not yet renew
print(accumulator(10)) #renew

#get variable value
print(state.get_value())
accumulator(1)
print(state.get_value())
accumulator(10)
print(state.get_value()) #才可以抓出數字

#set variable value
#set to new model
state.set_value(-1) #reset as -1
accumulator(3) #plus 3
print(state.get_value()) #as -1+3=2

#use share value temporately
tmp_func = state*2 + inc
a = T.scalar(dtype=state.dtype)
skip_shared = theano.function([inc,a],tmp_func,givens=[(state,a)])
#a as replacement for state, given state use a replace not change state itself
print(skip_shared(2,3)) #=8=3*2+2 where a=3 in skip_funct replace as state
print(state.get_value) #=still 2



##7 8 activation function
#cant use linear, y=Act(Wx)
#Act()=relu() or sigmoid(), tanh() diffable

#theano.tensor.nnet.nnet.sigmoid()
#hidden layer: softplus(), softmax(), relu(), tanh()
#classification: softmax for output
#regression: linear for output or relu if only positive

##9 define layer












