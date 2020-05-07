import mxnet as mx

import sockeye.constants as C
from sockeye import utils
from sockeye.config import Config


import logging
logger = logging.getLogger(__name__)


def get_gcn(config, prefix):
    gcn = GCNCell(input_dim=config.input_dim,
                  output_dim=config.output_dim,
                  direction_num=config.direction_num,
                  num_blocks=config.num_blocks,
                  adj_norm=config.adj_norm,
                  dropout=config.dropout,
                  prefix=prefix)
    return gcn


class GCNConfig(Config):

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 direction_num: int,
                 num_blocks: int,
                 adj_norm: bool = True,
                 activation: str = 'relu',
                 dropout: float = 0.0,
                 dtype: str = C.DTYPE_FP32) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.direction_num = direction_num
        self.num_blocks = num_blocks
        self.adj_norm = adj_norm
        self.activation = activation
        self.dropout = dropout
        self.dtype = dtype


class GCNCell(object):
    def __init__(self,
                 input_dim,
                 output_dim,
                 direction_num,
                 num_blocks,
                 adj_norm=True,
                 prefix='gcn_',
                 activation='relu',
                 dropout=0.0):

        self._input_dim = input_dim
        self._output_dim = output_dim
        self._direction_num = direction_num
        self._num_blocks = num_blocks
        self._layers = []
        self._prefix = prefix

        self._activation = activation
        self._dropout = dropout
        self._norm = adj_norm

        self.reset()
        self._modified = False
        self._own_params = True

        if self._input_dim != self._output_dim:
            self._input_W = mx.symbol.Variable(self._prefix + '_input_weight',
                                               shape=(input_dim, output_dim))
            self._input_b = mx.symbol.Variable(self._prefix + '_input_bias',
                                               shape=(output_dim,))

        for i in range(self._num_blocks):
            self._layers.append(GraphConvolution(prefix="%s%d_1_" % (self._prefix, i),
                                                 sublayer_num=1,
                                                 output_dim=self._output_dim,
                                                 direction_num=self._direction_num,
                                                 dropout=self._dropout,
                                                 norm=self._norm,
                                                 activation=self._activation))



        # Layer Aggregation Params used for aggregating output from all the sub-blocks to form the final representation
        self._aggregate_W = mx.symbol.Variable(self._prefix + '_aggregate_weight',
                                               shape=(self._num_blocks * self._output_dim, self._output_dim))
        self._aggregate_b = mx.symbol.Variable(self._prefix + '_aggregate_bias',
                                               shape=(self._output_dim,))


    def convolve(self, adj, inputs, seq_len):
        layer_list = []

        if self._input_dim != self._output_dim:
            inputs = mx.sym.dot(inputs, self._input_W)
            inputs = mx.sym.broadcast_add(inputs, self._input_b)

        outputs = inputs
        for i in range(len(self._layers)):
            outputs = self._layers[i](adj=adj, inputs=outputs, seq_len=seq_len)
            layer_list.append(outputs)

        # aggregate the output to form the final representation
        aggregate_output = mx.sym.concat(*layer_list, dim=2)
        aggregate_output = mx.sym.dot(aggregate_output, self._aggregate_W)
        aggregate_output = mx.sym.broadcast_add(aggregate_output, self._aggregate_b)


        return aggregate_output

    def reset(self):
        pass


class GraphConvolution:

    def __init__(self,
                 prefix: str,
                 sublayer_num: int,
                 output_dim: int,
                 direction_num: int,
                 dropout: float,
                 norm: bool,
                 activation: str = 'relu'):

        self._prefix = prefix
        self._sublayer_num = sublayer_num
        self._output_dim = output_dim
        self._direction_num = direction_num
        utils.check_condition(output_dim % sublayer_num == 0,
                              "Number of sublayer_num (%d) must divide attention depth (%d)" % (sublayer_num, output_dim))
        self._hidden_dim = self._output_dim 
        self._dropout = dropout
        self._norm = norm
        self._activation = activation
        self._weight_list = []
        self._bias_list = []

        # Graph Convolution Params
        for i in range(sublayer_num):
            self._weight_list.append([mx.symbol.Variable(self._prefix + "_dense_" + str(i) + "_" + str(j) + "_weight",
                                                         shape=(self._output_dim + self._hidden_dim * i, self._hidden_dim))
                                      for j in range(self._direction_num)])
            self._bias_list.append([mx.symbol.Variable(self._prefix + "_dense_" + str(i) + "_" + str(j) + "_bias",
                                                       shape=(self._hidden_dim,))
                                    for j in range(self._direction_num)])


        # Attention Params

        self._att_weight = mx.symbol.Variable(self._prefix  + "_weight",
                                                     shape=(self._direction_num, 1))

        # Direction Params for aggregating the output of each direction
        self._direct_W = mx.symbol.Variable(self._prefix + '_direct_weight',
                                             shape=(self._hidden_dim,  self._hidden_dim))
        self._direct_b = mx.symbol.Variable(self._prefix + '_direct_bias',
                                             shape=(self._hidden_dim,))



    def __call__(self, adj, inputs, seq_len):
        outputs = inputs
        for i in range(self._sublayer_num):
            convolved = self._convolve(adj, outputs, i, seq_len)
            outputs = convolved
        if self._dropout != 0.0:
            outputs = mx.sym.Dropout(outputs, p=self._dropout)
        outputs = mx.sym.broadcast_add(outputs, inputs)

        return outputs

    def _convolve(self, adj, inputs, i, seq_len):
        direct_list = []
        for j in range(self._direction_num):
            k = i * self._direction_num + j

            weight = self._weight_list[i][j]
            bias = self._bias_list[i][j]

            output = mx.sym.dot(inputs, weight)
            output = mx.sym.broadcast_add(output, bias)
            label_id = j + 2
            mask = mx.sym.ones_like(adj) * label_id
            adji = (mask == adj)

            output = mx.sym.batch_dot(adji, output)
            output = mx.sym.expand_dims(output, axis=1)
            direct_list.append(output)

        outputs = mx.sym.concat(*direct_list, dim=1)
        outputs = mx.sym.reshape(outputs, shape=(-1, seq_len, self._hidden_dim, self._direction_num))
        outputs = mx.sym.dot(outputs, self._att_weight)
        outputs = mx.sym.reshape(outputs, shape=(-1, seq_len, self._hidden_dim))

        outputs = mx.sym.dot(outputs, self._direct_W)
        outputs = mx.sym.broadcast_add(outputs, self._direct_b)

        if self._norm:
            norm_adj = mx.sym.broadcast_not_equal(adj, mx.sym.zeros_like(adj))
            norm_factor = mx.sym.sum(norm_adj, axis=2, keepdims=True)
            outputs = mx.sym.broadcast_div(outputs, norm_factor)

        final_output = mx.sym.Activation(outputs, act_type=self._activation)
        return final_output
