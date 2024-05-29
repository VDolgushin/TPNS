from lab3b.layers.Layer import Layer
import numpy as np


class Conv2DLayer(Layer):
    def __init__(self, input_size, output_size, kernel_size, stride, padding=0, wScale=0.01):
        self.x = None
        self.col = None
        self.input_size = input_size
        self.output_size = output_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = padding
        self.weights = np.random.randn(output_size, input_size, kernel_size, kernel_size) * wScale
        self.bias = np.random.randn(output_size) * wScale

    def calculate_output(self, x):
        x = np.pad(x, ((0, 0), (0, 0), (self.pad, self.pad), (self.pad, self.pad)), "constant", constant_values=0)
        self.x = x
        self.col = self.im2col(x)
        col_w = self.weights.reshape(self.output_size, -1).T
        out = np.dot(self.col, col_w) + self.bias
        n, c, h, w = x.shape
        out_h = (h - self.kernel_size) // self.stride + 1
        out_w = (w - self.kernel_size) // self.stride + 1
        out = out.reshape(n, out_h, out_w, -1).transpose(0, 3, 1, 2)
        return out

    def back_propagation(self, d_out, learning_rate):
        d_out = d_out.transpose(0, 2, 3, 1).reshape(-1, self.output_size)
        db = np.sum(d_out, axis=0)
        dw = np.dot(self.col.T, d_out)
        dw = dw.transpose(1, 0).reshape(self.output_size, self.input_size, self.kernel_size, self.kernel_size)
        self.weights -= learning_rate * dw
        self.bias -= learning_rate * db
        d_col = np.dot(d_out, self.weights.reshape(self.output_size, -1))
        dx = self.col2im(d_col, self.x.shape)
        return dx

    def im2col(self, input_data):
        n, c, h, w = input_data.shape

        out_h = (h - self.kernel_size) // self.stride + 1
        out_w = (w - self.kernel_size) // self.stride + 1
        img = input_data
        col = np.zeros((n, c, self.kernel_size, self.kernel_size, out_h, out_w))
        for y in range(self.kernel_size):
            y_max = y + self.stride * out_h
            for x in range(self.kernel_size):
                x_max = x + self.stride * out_w
                col[:, :, y, x, :, :] = img[:, :, y:y_max:self.stride, x:x_max:self.stride]
        col = col.transpose(0, 4, 5, 1, 2, 3).reshape(n * out_h * out_w, -1)
        return col

    def col2im(self, cols, input_shape):
        n, c, h, w = input_shape
        img = np.zeros((n, c, h, w), dtype=cols.dtype)
        out_h = (h - self.kernel_size) // self.stride + 1
        out_w = (w - self.kernel_size) // self.stride + 1
        cols_reshaped = cols.reshape(n, out_h, out_w, c, self.kernel_size, self.kernel_size).transpose(0, 3, 4, 5, 1, 2)
        for y in range(self.kernel_size):
            y_max = y + self.stride * out_h
            for x in range(self.kernel_size):
                x_max = x + self.stride * out_w
                img[:, :, y:y_max: self.stride, x:x_max: self.stride] += cols_reshaped[:, :, y, x, :, :]
        return img
