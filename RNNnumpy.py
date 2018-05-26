import numpy as np

class RNNnumpy:
    #  Initialization
    def __init__(self,word_dim,hidden_dim=100,bptt_truncate=4):
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate

        self.U = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, word_dim))
        self.V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (word_dim, hidden_dim))
        self.W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))
    # 定义softmax函数
    def softmax(self,x):
        xt = np.exp(x - np.max(x))
        return xt / np.sum(xt)

    # Forward Propagation
    def forward_propagation(self, x):
        T = len(x)
        # 定义s 和 o向量
        s = np.zeros((T+1,self.hidden_dim))
        s[-1] = np.zeros(self.hidden_dim)
        o = np.zeros((T, self.word_dim))
        # 对每个时间step

        for t in np.arange(T):
            s[t] = np.tanh(self.U[:, x[t]] + self.W.dot(s[t-1]))

            o[t] = self.softmax(self.V.dot(s[t]))
        return [o, s]
    def predict(self, x):
        o,s = self.forward_propagation(x)
        return np.argmax(o, axis=1)

    def calculate_total_loss(self, x, y):
        L = 0
        # For each sentence...
        for i in np.arange(len(y)):
            o, s = self.forward_propagation(x[i])
            # We only care about our prediction of the "correct" words
            correct_word_predictions = o[np.arange(len(y[i])), y[i]]
            # Add to the loss based on how off we were
            L += -1 * np.sum(np.log(correct_word_predictions))
        return L

    def calculate_loss(self, x, y):
        # Divide the total loss by the number of training examples
        N = np.sum((len(y_i) for y_i in y))
        return self.calculate_total_loss(x, y) / N

    def bptt(self,x,y):
        T = len(y)
        # 执行前向传播
        o, s = self.forward_propagation(x)
        # We accumulate the gradients in these variables
        dLdU = np.zeros(self.U.shape)
        dLdV = np.zeros(self.V.shape)
        dLdW = np.zeros(self.W.shape)
        delta_o = o
        delta_o[np.arange(len(y)),y] -= 1
        # For each output backwards...
        for t in np.arange(T)[::-1]:
            dLdV += np.outer(delta_o[t], s[t].T)
            delta_t = self.V.T.dot(delta_o[t]) * (1-(s[t]**2))
            for bptt_step in np.arange(max(0, t-self.bptt_truncate), t+1)[::-1]:
                dLdW += np.outer(delta_t, s[bptt_step-1])
                dLdU[:,x[bptt_step]] += delta_t
                delta_t = self.W.T.dot(delta_t) * (1 - s[bptt_step - 1]**2)

        return [dLdU, dLdV,dLdW]

    def numpy_sgd_step(self,x,y,learning_rate):
        dLdU, dLdV, dLdW = self.bptt(x, y)

        self.U -= learning_rate* dLdU
        self.V -= learning_rate * dLdV
        self.W -= learning_rate * dLdW


    # def gradient_check(self, x, y, h=0.001, error_threshold=0.01):
    #     # Calculate the gradients using backpropagation. We want to checker if these are correct.
    #     bptt_gradients = model.bptt(x, y)
    #     # List of all parameters we want to check.
    #     model_parameters = ['U', 'V', 'W']
    #     # Gradient check for each parameter
    #     for pidx, pname in enumerate(model_parameters):
    #         # Get the actual parameter value from the mode, e.g. model.W
    #         parameter = operator.attrgetter(pname)(self)
    #         print
    #         "Performing gradient check for parameter %s with size %d." % (pname, np.prod(parameter.shape))
    #         # Iterate over each element of the parameter matrix, e.g. (0,0), (0,1), ...
    #         it = np.nditer(parameter, flags=['multi_index'], op_flags=['readwrite'])
    #         while not it.finished:
    #             ix = it.multi_index
    #             # Save the original value so we can reset it later
    #             original_value = parameter[ix]
    #             # Estimate the gradient using (f(x+h) - f(x-h))/(2*h)
    #             parameter[ix] = original_value + h
    #             gradplus = model.calculate_total_loss([x], [y])
    #             parameter[ix] = original_value - h
    #             gradminus = model.calculate_total_loss([x], [y])
    #             estimated_gradient = (gradplus - gradminus) / (2 * h)
    #             # Reset parameter to original value
    #             parameter[ix] = original_value
    #             # The gradient for this parameter calculated using backpropagation
    #             backprop_gradient = bptt_gradients[pidx][ix]
    #             # calculate The relative error: (|x - y|/(|x| + |y|))
    #             relative_error = np.abs(backprop_gradient - estimated_gradient) / (
    #                     np.abs(backprop_gradient) + np.abs(estimated_gradient))
    #             # If the error is to large fail the gradient check
    #             if relative_error > error_threshold:
    #                 print("Gradient Check ERROR: parameter=%s ix=%s" % (pname, ix))
    #                 print("+h Loss: %f" % gradplus)
    #                 print("-h Loss: %f" % gradminus)
    #                 print("Estimated_gradient: %f" % estimated_gradient)
    #                 print("Backpropagation gradient: %f" % backprop_gradient)
    #                 print("Relative Error: %f" % relative_error)
    #                 return
    #             it.iternext()
    #         print("Gradient check for parameter %s passed." % (pname))

