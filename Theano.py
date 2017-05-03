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

class Layer(object):
	def ___inti___(self,inputs,in_size,out_size,activation_function = None):
		self.W = theano.shared(np.random.normal(0,1,(in_size,out_size)))
		#def random initial better for gd
		self.b = theano.shared(np.zero((out_size,))+0.1) 
		self.Wx_plus_b = T.dot(input,self.W) +self.b
		self.activation_function = activation_function
		if activation_function is None:
			self.output = self.Wx_plus_b
		else:
			self.output = activation_function(self.Wx_plus_b)

'''
to define a layer like
l1 = Layer(input,in_size=1,out_size=10,activation_function)
l2 = Layer(l1.output,10,2,None)
'''



##10 11 Regression problem, prediction
from __future__ import print_function
import theano
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt


class Layer(object):
    def __init__(self, inputs, in_size, out_size, activation_function=None):
        self.W = theano.shared(np.random.normal(0, 1, (in_size, out_size)))
        self.b = theano.shared(np.zeros((out_size, )) + 0.1)
        self.Wx_plus_b = T.dot(inputs, self.W) + self.b
        self.activation_function = activation_function
        if activation_function is None:
            self.outputs = self.Wx_plus_b
        else:
            self.outputs = self.activation_function(self.Wx_plus_b)


# Make up some fake data
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise        # y = x^2 - 0.5

# show the fake data
plt.scatter(x_data, y_data)
plt.show()

# determine the inputs dtype
x = T.dmatrix("x")
y = T.dmatrix("y")

# add layers
l1 = Layer(x, 1, 10, T.nnet.relu)
l2 = Layer(l1.outputs, 10, 1, None)

# compute the cost
cost = T.mean(T.square(l2.outputs - y)) #300 pts avg

# compute the gradients
gW1, gb1, gW2, gb2 = T.grad(cost, [l1.W, l1.b, l2.W, l2.b])
#grad function according to cost from 4 parameter weight bias

# apply gradient descent
learning_rate = 0.05 
train = theano.function(
    inputs=[x, y],
    outputs=cost,
    updates=[(l1.W, l1.W - learning_rate * gW1),
             (l1.b, l1.b - learning_rate * gb1),
             (l2.W, l2.W - learning_rate * gW2),
             (l2.b, l2.b - learning_rate * gb2)])

# prediction
predict = theano.function(inputs=[x], outputs=l2.outputs)
#design prediction function need only x give pred_y

#plot fake data
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(x_data,y_data)
plt.ion() #cont renew
plt.show() #here block=True, will not block by ion

for i in range(1000):
    # training
    err = train(x_data, y_data)
    if i % 50 == 0:
    	#print(err)

    	#remove prev line
    	try:
    		ax.lines.remove(lines[0])
    	except Exception:
    		pass #avoid first step, there's no line to remove

    	#visualize result and improve
    	pred = prediction(x_data)
    	#plot into plot
    	lines = ax.plot(x_data,pred,'r-',lw=5)
    	plt.pause(1) #stop for 1 sec



##12 classification
def compute_accuracy(y_target, y_predict):
    correct_prediction = np.equal(y_predict, y_target)
    accuracy = np.sum(correct_prediction)/len(correct_prediction)
    return accuracy

rng = np.random

N = 400                          # training sample size
feats = 784                      # number of input variables

# generate a dataset: D = (input_values, target_class)
D = (rng.randn(N, feats), rng.randint(size=N, low=0, high=2))
#class 0,1

# Declare Theano symbolic variables
x = T.dmatrix("x")
y = T.dvector("y")

# initialize the weights and biases
W = theano.shared(rng.randn(feats), name="w")
b = theano.shared(0., name="b")

# Construct Theano expression graph
p_1 = T.nnet.sigmoid(T.dot(x, W) + b)   # Logistic Probability that target = 1 (activation function)
prediction = p_1 > 0.5                    # The prediction thresholded
xent = -y * T.log(p_1) - (1-y) * T.log(1-p_1) # Cross-entropy loss function
# or
# xent = T.nnet.binary_crossentropy(p_1, y) # this is provided by theano
cost = xent.mean() + 0.01 * (W ** 2).sum()# The cost to minimize (l2 regularization)
#avoid over fitting here, add sum(W^2)
gW, gb = T.grad(cost, [W, b])             # Compute the gradient of the cost

# Compile
learning_rate = 0.1
train = theano.function(
          inputs=[x, y],
          outputs=[prediction, xent.mean()],
          updates=((W, W - learning_rate * gW), 
          		   (b, b - learning_rate * gb)))
predict = theano.function(inputs=[x], outputs=prediction)

# Training
for i in range(500):
    pred, err = train(D[0], D[1]) #input=[x,y]
    if i % 50 == 0:
        print('cost:', err)
        print("accuracy:", compute_accuracy(D[1], predict(D[0])))

print("target values for D:")
print(D[1])
print("prediction on D:")
print(predict(D[0]))



##13 overfit
#solve: more example, regularization on W



##14 regularization
from __future__ import print_function
import theano
from sklearn.datasets import load_boston
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt


class Layer(object):
    def __init__(self, inputs, in_size, out_size, activation_function=None):
        self.W = theano.shared(np.random.normal(0, 1, (in_size, out_size)))
        self.b = theano.shared(np.zeros((out_size, )) + 0.1)
        self.Wx_plus_b = T.dot(inputs, self.W) + self.b
        self.activation_function = activation_function
        if activation_function is None:
            self.outputs = self.Wx_plus_b
        else:
            self.outputs = self.activation_function(self.Wx_plus_b)


def minmax_normalization(data):
    xs_max = np.max(data, axis=0)
    xs_min = np.min(data, axis=0)
    xs = (1 - 0) * (data - xs_min) / (xs_max - xs_min) + 0
    return xs
    #easier to learn into (0,1)

np.random.seed(100)
x_data = load_boston().data
# minmax normalization, rescale the inputs
x_data = minmax_normalization(x_data)
y_data = load_boston().target[:, np.newaxis]
#give newax as matrix in y

# cross validation, train test data split
x_train, y_train = x_data[:400], y_data[:400]
x_test, y_test = x_data[400:], y_data[400:]

x = T.dmatrix("x")
y = T.dmatrix("y")

l1 = Layer(x, 13, 50, T.tanh)       #13 >> 50 hidden lay
l2 = Layer(l1.outputs, 50, 1, None) #50 >> 1 houseprice

# the way to compute cost
cost = T.mean(T.square(l2.outputs - y))      
#cost without regularization

# cost = T.mean(T.square(l2.outputs - y)) + 0.1 * ((l1.W ** 2).sum() + (l2.W ** 2).sum())  
# with l2 regularization

# cost = T.mean(T.square(l2.outputs - y)) + 0.1 * (abs(l1.W).sum() + abs(l2.W).sum())  
# with l1 regularization

gW1, gb1, gW2, gb2 = T.grad(cost, [l1.W, l1.b, l2.W, l2.b])

learning_rate = 0.01
train = theano.function(
    inputs=[x, y],
    updates=[(l1.W, l1.W - learning_rate * gW1),
             (l1.b, l1.b - learning_rate * gb1),
             (l2.W, l2.W - learning_rate * gW2),
             (l2.b, l2.b - learning_rate * gb2)])

compute_cost = theano.function(inputs=[x, y], outputs=cost)

# record cost
train_err_list = []
test_err_list = []
learning_time = []
for i in range(1000):
    train(x_train, y_train)
    if i % 10 == 0:
        # record cost
        train_err_list.append(compute_cost(x_train, y_train))
        test_err_list.append(compute_cost(x_test, y_test))
        learning_time.append(i)

# plot cost history
plt.plot(learning_time, train_err_list, 'r-')
plt.plot(learning_time, test_err_list, 'b--')
plt.show()



##15 save model
#keep model by parameter
import pickle
def compute_accuracy(y_target, y_predict):
    correct_prediction = np.equal(y_predict, y_target)
    accuracy = np.sum(correct_prediction)/len(correct_prediction)
    return accuracy

rng = np.random

N = 400                          # training sample size
feats = 784                      # number of input variables

# generate a dataset: D = (input_values, target_class)
D = (rng.randn(N, feats), rng.randint(size=N, low=0, high=2))
#class 0,1

# Declare Theano symbolic variables
x = T.dmatrix("x")
y = T.dvector("y")

# initialize the weights and biases
W = theano.shared(rng.randn(feats), name="w")
b = theano.shared(0., name="b")

# Construct Theano expression graph
p_1 = T.nnet.sigmoid(T.dot(x, W) + b)   # Logistic Probability that target = 1 (activation function)
prediction = p_1 > 0.5                    # The prediction thresholded
xent = -y * T.log(p_1) - (1-y) * T.log(1-p_1) # Cross-entropy loss function
# or
# xent = T.nnet.binary_crossentropy(p_1, y) # this is provided by theano
cost = xent.mean() + 0.01 * (W ** 2).sum()# The cost to minimize (l2 regularization)
#avoid over fitting here, add sum(W^2)
gW, gb = T.grad(cost, [W, b])             # Compute the gradient of the cost

# Compile
learning_rate = 0.1
train = theano.function(
          inputs=[x, y],
          outputs=[prediction, xent.mean()],
          updates=((W, W - learning_rate * gW), 
          		   (b, b - learning_rate * gb)))
predict = theano.function(inputs=[x], outputs=prediction)

#training
for i in range(500):
	train(D[0],D[1])

#save
#create save folder at wd
with open('save/model.pickle','wb') as file:
	model = [w.get_value(),b.get_value()] #after train get parameter w,b
	pickle.dump(model, file)
	print(model[0][:10]) #look first 10 same?
	print('accuracy:',compute_accuracy(D[1],predict(D[0])))

#load
with open('save/model.pickle','rb') as file:
	model = pickle.load(file)
	w.set_value(model[0])
	b.set_value(model[1])
	print(w.get_value()[:10])
	print('accuracy:',compute_accuracy(D[1],predict(D[0])))	



##16 summary
#basic usage
#regression nn
#classification nn
#overfit solution in Theano
#save Theano nn for future

##############
#GPU
#NVIDIA CUDA backend
#ConvolutionalNN, deep learning
#RNN


