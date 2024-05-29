import numpy as np
from lab3b.layers.Layer import Layer


class RNN_Layer(Layer):
    def __init__(self, input_size, hidden_size, output_size, return_sequence=False, wScale=1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.wScale = wScale
        self.return_sequence = return_sequence
        self.Wi = np.random.uniform(-self.wScale, self.wScale, size=(self.input_size, self.hidden_size)) / np.sqrt(input_size)
        self.Wh = np.random.uniform(-self.wScale, self.wScale, size=(self.hidden_size, self.hidden_size)) / np.sqrt(hidden_size)
        self.Wo = np.random.uniform(-self.wScale, self.wScale, size=(self.hidden_size, self.output_size)) / np.sqrt(output_size)
        self.h_bias = np.zeros(self.hidden_size)
        self.o_bias = np.zeros(self.output_size)
        self.x = None
        self.h = None

    def calculate_output(self, x):
        self.x = x
        Wx = np.dot(x, self.Wi)
        batch_size, seq_len, _ = x.shape
        ht = np.zeros((batch_size, self.hidden_size))
        self.h = np.zeros((batch_size, seq_len, self.hidden_size))

        for i in range(seq_len):
            ht = self.activation_function(Wx[:, i, :] + np.dot(ht, self.Wh) + self.h_bias)
            self.h[:, i, :] = ht

        if self.return_sequence:
            output = np.dot(self.h, self.Wo) + self.o_bias
        else:
            final_hidden_state = self.h[:, -1, :]
            output = np.dot(final_hidden_state, self.Wo) + self.o_bias

        return output

    def back_propagation(self, do, learning_rate):
        batch_size, seq_len, _ = self.x.shape
        Wi_gradient = np.zeros_like(self.Wi)
        Wh_gradient = np.zeros_like(self.Wh)
        Wo_gradient = np.zeros_like(self.Wo)
        h_bias_gradient = np.zeros_like(self.h_bias)
        o_bias_gradient = np.zeros_like(self.o_bias)
        dx = np.zeros_like(self.x)
        error_gradient = np.zeros((batch_size, self.hidden_size))

        for i in reversed(range(seq_len)):
            h_i = self.h[:, i, :]
            if self.return_sequence:
                grad_i = do[:, i, :]
                Wo_gradient += np.dot(h_i.T, grad_i)
                o_bias_gradient += np.mean(grad_i, axis=0)
                hidden_error = np.dot(grad_i, self.Wo.T) + error_gradient
            else:
                if i == seq_len - 1:
                    grad_i = do
                    Wo_gradient += np.dot(h_i.T, grad_i)
                    o_bias_gradient += np.mean(grad_i, axis=0)
                    hidden_error = np.dot(grad_i, self.Wo.T)
                else:
                    hidden_error = error_gradient

            dh = self.tanh_derivative(h_i)
            h_grad_i = dh * hidden_error

            if i > 0:
                Wh_gradient += np.dot(self.h[:, i - 1, :].T, h_grad_i)
                h_bias_gradient += np.mean(h_grad_i, axis=0)

            input_x = self.x[:, i, :]
            Wi_gradient += np.dot(input_x.T, h_grad_i)

            error_gradient = np.dot(h_grad_i, self.Wh.T)

            dx[:, i, :] = np.dot(h_grad_i, self.Wi.T)

        self.Wi -= learning_rate * Wi_gradient / batch_size
        self.Wh -= learning_rate * Wh_gradient / batch_size
        self.Wo -= learning_rate * Wo_gradient / batch_size
        self.h_bias -= learning_rate * h_bias_gradient / batch_size
        self.o_bias -= learning_rate * o_bias_gradient / batch_size

        return dx
    def tanh(self, x):
        return 2 / (1 + np.exp(-2 * x)) - 1

    def tanh_derivative(self, x):
        return 1 - x ** 2

    def activation_function(self, x):
        return self.tanh(x)
