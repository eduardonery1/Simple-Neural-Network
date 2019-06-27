BASIC CONCEPTS LIKE BACKPROP ARE USED IN THE SCRIPT

0. The Blocks
	First we use numpy to define the sigmoid function and it's derivative, then we define the input and output matrixes. We are asking a question, what number or coeffients we need to use to transform X in Y, intelligence is induction after all. Later we can use the machine induction to build mpre complex actions. We are discovering how to give meaning to data, the machine will use two synapses or two groups of coeficients in our inputs to fransform them in to output. 

def nonlin(x,deriv=False):
	if(deriv==True):
	    return x*(1-x)

	return 1/(1+np.exp(-x))
    
X = np.array([[0,0,1],
            [0,1,1],
            [1,0,1],
            [1,1,1]])
                
y = np.array([[0],
			[1],
			[1],
			[0]])

np.random.seed(1)

syn0 = 2*np.random.random((3,4)) - 1
syn1 = 2*np.random.random((4,1)) - 1


1. The Layers
for j in xrange(60000):

    l0 = X
    l1 = nonlin(np.dot(l0,syn0))
    l2 = nonlin(np.dot(l1,syn1))

2. Measuring Error
    l2_error = y - l2
    
    if (j% 10000) == 0:
        print "Error:" + str(np.mean(np.abs(l2_error)))
        
3. Gradient Descent
How wrong we were? What should we change in the second layer? Higher the derivative of the sigmoid higher the uncertanty of the result, it means that if the l2_error is negative we are getting an aproximation of the lower point of the curve. 
    l2_delta = l2_error*nonlin(l2,deriv=True)
the dot product of l2_delta and the transpose of syn1 because: l2 is the dot prod of l1 and syn1, so the dot of syn1 and how wrong was l2 should be how wrong was l1. 
    l1_error = l2_delta.dot(syn1.T)
    l1_delta = l1_error * nonlin(l1,deriv=True)

4. Backpropagation or Learning

    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)

