from lab3b.layers.Layer import Layer
import numpy as np

class AvgPool2DLayer(Layer):
    def __init__(self, pool_size, stride):
        self.pred = None
        self.pool_size = pool_size
        self.stride = stride

    def calculate_output(self, x):
        n, c, h, w = x.shape

        x_reshaped = x.reshape(n, c, h // self.pool_size, self.pool_size, w // self.pool_size, self.pool_size)
        x_strided = x_reshaped.transpose(0, 1, 2, 4, 3, 5).reshape(n, c, h // self.pool_size, w // self.pool_size, self.pool_size * self.pool_size)

        out = x_strided.mean(axis=4)
        self.pred = (x.shape, x_strided.shape, out.shape)
        return out

    def back_propagation(self, d_out, learning_rate):
        n, c, h, w = self.pred[0]
        _, _, out_h, out_w, _ = self.pred[1]

        dx_strided = np.zeros((n, c, out_h, out_w, self.pool_size * self.pool_size))

        dout_reshaped = d_out[:, :, :, :, np.newaxis]
        dx_strided[:] = dout_reshaped / (self.pool_size * self.pool_size)

        dx_strided = dx_strided.reshape(n, c, out_h, self.pool_size, out_w, self.pool_size).transpose(0, 1, 2, 4, 3, 5)
        dx = dx_strided.reshape(n, c, h , w )
        return dx
