import numpy as np

from lab3b.layers.Layer import Layer


class LSTMLayer(Layer):
    def __init__(self, input_size, hidden_size, output_size, return_sequence=False, wScale=0.2):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.return_sequence = return_sequence
        self.wScale = wScale

        self.x = None
        self.ht = None
        self.ct = None
        self.ot = None
        self.ft = None
        self.ins = None
        self.ct_ = None
        self.c_tanh = None

        self.Wi = np.random.uniform(-self.wScale, self.wScale, (self.input_size, self.hidden_size))  # noqa
        self.Ui = np.random.uniform(-self.wScale, self.wScale, (self.hidden_size, self.hidden_size))
        self.bi = np.zeros(self.hidden_size)

        self.Wf = np.random.uniform(-self.wScale, self.wScale, (self.input_size, self.hidden_size))
        self.Uf = np.random.uniform(-self.wScale, self.wScale, (self.hidden_size, self.hidden_size))
        self.bf = np.zeros(self.hidden_size)

        self.Wo = np.random.uniform(-self.wScale, self.wScale, (self.input_size, self.hidden_size))
        self.Uo = np.random.uniform(-self.wScale, self.wScale, (self.hidden_size, self.hidden_size))
        self.bo = np.zeros(self.hidden_size)

        self.Wc = np.random.uniform(-self.wScale, self.wScale, (self.input_size, self.hidden_size))
        self.Uc = np.random.uniform(-self.wScale, self.wScale, (self.hidden_size, self.hidden_size))
        self.bc = np.zeros(self.hidden_size)

        self.Wy = np.random.uniform(-self.wScale, self.wScale, (self.hidden_size, self.output_size))
        self.by = np.zeros(self.output_size)

    def calculate_output(self, x):

        batch_size, sequence_length, _ = x.shape
        self.x = x

        ht = np.zeros((batch_size, self.hidden_size))
        ct = np.zeros((batch_size, self.hidden_size))

        self.ht = np.zeros((batch_size, sequence_length, self.hidden_size))
        self.ct = np.zeros((batch_size, sequence_length, self.hidden_size))
        self.ins = np.zeros((batch_size, sequence_length, self.hidden_size))
        self.ot = np.zeros((batch_size, sequence_length, self.hidden_size))
        self.ft = np.zeros((batch_size, sequence_length, self.hidden_size))
        self.ct_ = np.zeros((batch_size, sequence_length, self.hidden_size))
        self.c_tanh = np.zeros((batch_size, sequence_length, self.hidden_size))

        for t in range(sequence_length):
            # Gates operations
            xt = x[:, t, :]
            it = self.sigmoid(xt @ self.Wi + ht @ self.Ui + self.bi)
            ft = self.sigmoid(xt @ self.Wf + ht @ self.Uf + self.bf)
            ot = self.sigmoid(xt @ self.Wo + ht @ self.Uo + self.bo)
            _c = self.tanh(xt @ self.Wc + ht @ self.Uc + self.bc)

            self.ins[:, t, :] = it
            self.ot[:, t, :] = ot
            self.ft[:, t, :] = ft
            self.ct_[:, t, :] = _c

            ct = ft * ct + it * _c
            self.ct[:, t, :] = ct

            self.c_tanh[:, t, :] = self.tanh(ct)
            ht = ot * self.c_tanh[:, t, :]
            self.ht[:, t, :] = ht

        if self.return_sequence:
            output = self.ht @ self.Wy + self.by
        else:
            final_hidden_state = self.ht[:, -1, :]
            output = final_hidden_state @ self.Wy + self.by

        return output

    def back_propagation(self, dY, learning_rate):
        batch_size, sequence_length, _ = self.x.shape

        dWi, dUi, dbi = np.zeros_like(self.Wi), np.zeros_like(self.Ui), np.zeros_like(self.bi)
        dWf, dUf, dbf = np.zeros_like(self.Wf), np.zeros_like(self.Uf), np.zeros_like(self.bf)
        dWo, dUo, dbo = np.zeros_like(self.Wo), np.zeros_like(self.Uo), np.zeros_like(self.bo)
        dWc, dUc, dbc = np.zeros_like(self.Wc), np.zeros_like(self.Uc), np.zeros_like(self.bc)
        dWy, dby = np.zeros_like(self.Wy), np.zeros_like(self.by)

        dh_next = np.zeros((batch_size, self.hidden_size))
        dc_next = np.zeros((batch_size, self.hidden_size))

        dX = np.zeros((batch_size, sequence_length, self.input_size))

        for t in reversed(range(sequence_length)):
            dy = dY[:, t, :] if self.return_sequence else (dY if t == sequence_length - 1 else np.zeros_like(dY))

            dWy += self.ht[:, t, :].T @ dy
            dby += np.sum(dy, axis=0)

            dh = dy @ self.Wy.T + dh_next
            dc = self.ot[:, t, :] * dh * self.tanh_derivative(self.ct[:, t, :]) + dc_next

            dot = self.sigmoid_derivative(self.ot[:, t, :]) * self.c_tanh[:, t, :] * dh

            if t > 0:
                dft = self.ct[:, t - 1, :] * dc * self.sigmoid_derivative(self.ft[:, t, :])
            else:
                dft = np.zeros_like(self.ft[:, t, :])

            dit = self.ct_[:, t, :] * dc * self.sigmoid_derivative(self.ins[:, t, :])

            dct = self.ins[:, t, :] * dc * self.tanh_derivative(self.ct_[:, t, :])

            dWi += self.x[:, t, :].T @ dit
            dbi += np.sum(dit, axis=0)

            dWf += self.x[:, t, :].T @ dft
            dbf += np.sum(dft, axis=0)

            dWo += self.x[:, t, :].T @ dot
            dbo += np.sum(dot, axis=0)

            dWc += self.x[:, t, :].T @ dct
            dbc += np.sum(dct, axis=0)

            if t > 0:
                dUi += self.ht[:, t - 1, :].T @ dit
                dUf += self.ht[:, t - 1, :].T @ dft
                dUo += self.ht[:, t - 1, :].T @ dot
                dUc += self.ht[:, t - 1, :].T @ dct

            dh_next = dit @ self.Ui.T + dft @ self.Uf.T + dot @ self.Uo.T + dct @ self.Uc.T
            dc_next = self.ft[:, t, :] * dc

            dX[:, t, :] = dit @ self.Wi.T + dft @ self.Wf.T + dot @ self.Wo.T + dct @ self.Wc.T

        self.Wi -= learning_rate * dWi / batch_size
        self.Ui -= learning_rate * dUi / batch_size
        self.bi -= learning_rate * dbi / batch_size
        self.Wf -= learning_rate * dWf / batch_size
        self.Uf -= learning_rate * dUf / batch_size
        self.bf -= learning_rate * dbf / batch_size
        self.Wo -= learning_rate * dWo / batch_size
        self.Uo -= learning_rate * dUo / batch_size
        self.bo -= learning_rate * dbo / batch_size
        self.Wc -= learning_rate * dWc / batch_size
        self.Uc -= learning_rate * dUc / batch_size
        self.bc -= learning_rate * dbc / batch_size
        self.Wy -= learning_rate * dWy / batch_size
        self.by -= learning_rate * dby / batch_size

        return dX

    def tanh(self, x):
        return 2 / (1 + np.exp(-2 * x)) - 1

    def tanh_derivative(self, x):
        return 1 - x ** 2

    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))
    def sigmoid_derivative(self,x):
        return x * (1 - x)

