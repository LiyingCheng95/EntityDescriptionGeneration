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
        """

        :param input_dim: the dimension of the input vector (word embedding)
        :param output_dim: the dimension of the output of the DCGCN model
        :param direction_num: number of directions of the graph (e.g. for AMR graph, we have direct, reverse and self)
        :param num_blocks: number of densely-connected blocks, each block has 2 sub-blocks
        :param adj_norm: normalize the adjacency matrix
        :param prefix: prefix of learned parameters
        :param activation: activation function used by the DCGCN model
        :param dropout: dropout rate in the graph convolution layer
        """

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

        # If the embeddings size is different from the output size of the DCGCN model
        if self._input_dim != self._output_dim:
            self._input_W = mx.symbol.Variable(self._prefix + '_input_weight',
                                               shape=(input_dim, output_dim))
            self._input_b = mx.symbol.Variable(self._prefix + '_input_bias',
                                               shape=(output_dim,))

        # Each block has two densely-connected sub-blocks, sub-blocks have different number of sub-layers (6 and 3)
        for i in range(self._num_blocks):
            self._layers.append(GraphConvolution(prefix="%s%d_6_" % (self._prefix, i),
                                                 sublayer_num=6,
                                                 output_dim=self._output_dim,
                                                 direction_num=self._direction_num,
                                                 dropout=self._dropout,
                                                 norm=self._norm,
                                                 activation=self._activation))
            self._layers.append(GraphConvolution(prefix="%s%d_3_" % (self._prefix, i),
                                                 sublayer_num=3,
                                                 output_dim=self._output_dim,
                                                 direction_num=self._direction_num,
                                                 dropout=self._dropout,
                                                 norm=self._norm,
                                                 activation=self._activation))

        # Layer Aggregation Params used for aggregating output from all the sub-blocks to form the final representation
        # self._aggregate_W = mx.symbol.Variable(self._prefix + '_aggregate_weight',
        #                                        shape=(self._num_blocks * 2 * self._output_dim, self._output_dim))
        # self._aggregate_b = mx.symbol.Variable(self._prefix + '_aggregate_bias',
        #                                        shape=(self._output_dim,))

    def convolve(self, adj, inputs, seq_len):
        layer_list = []

        if self._input_dim != self._output_dim:
            inputs = mx.sym.dot(inputs, self._input_W)
            inputs = mx.sym.broadcast_add(inputs, self._input_b)

        # graph convolution in each sub-block and store the output to a list.
        outputs = inputs
        for i in range(len(self._layers)):
            outputs = self._layers[i](adj=adj, inputs=outputs)
            # layer_list.append(outputs)

        # aggregate the output to form the final representation
        # aggregate_output = mx.sym.concat(*layer_list, dim=2)
        # aggregate_output = mx.sym.dot(aggregate_output, self._aggregate_W)
        # aggregate_output = mx.sym.broadcast_add(aggregate_output, self._aggregate_b)

        return outputs

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
        utils.check_condition(output_dim % self._sublayer_num == 0,
                              "Number of sublayers (%d) must divide attention depth (%d)" % (self._sublayer_num, output_dim))

        # hiddem_dim is the dimension of each layer in the sub-block, it is proportional to the number of layers in the sub-block
        self._hidden_dim = self._output_dim // self._sublayer_num
        self._dropout = dropout
        self._norm = norm
        self._activation = activation
        self._weight_list = []
        self._bias_list = []

        # Densely Connected Graph Convolution Params
        # Each layer in the sub-block takes the input from all previous layers, so the input dimension is increase by hidden_dim * i
        # Each direction has a separate DCGCN to encode (e.g. for AMR graph we have 3 DCGCNs, whose parameters are not share)
        for i in range(self._sublayer_num):
            self._weight_list.append([mx.symbol.Variable(self._prefix + "_dense_" + str(i) + "_" + str(j) + "_weight",
                                                         shape=(self._output_dim + self._hidden_dim * i, self._hidden_dim))
                                      for j in range(self._direction_num)])
            self._bias_list.append([mx.symbol.Variable(self._prefix + "_dense_" + str(i) + "_" + str(j) + "_bias",
                                                       shape=(self._hidden_dim,))
                                    for j in range(self._direction_num)])


        # Direction Params for aggregating the output of each direction
        # self._direct_W = [mx.symbol.Variable(self._prefix + str(i) + '_direct_weight',
        #                                      shape=(self._hidden_dim, self._hidden_dim))
        #                   for i in range(self._sublayer_num)]
        # self._direct_b = [mx.symbol.Variable(self._prefix + str(i) + '_direct_bias',
        #                                      shape=(self._hidden_dim,))
        #                   for i in range(self._sublayer_num)]

        # Linear Transform Params used in the sub-block
        self._linear_W = mx.symbol.Variable(self._prefix + '_linear_weight',
                                            shape=(self._output_dim, self._output_dim))
        self._linear_b = mx.symbol.Variable(self._prefix + '_linear_bias',
                                            shape=(self._output_dim,))

    def __call__(self, adj, inputs):
        outputs = inputs
        cache_list = [outputs]
        output_list = []

        # cache_list is used to store the output for each layer in the sub-block
        for i in range(self._sublayer_num):
            convolved = self._convolve(adj, outputs, i)
            cache_list.append(convolved)
            outputs = mx.sym.concat(*cache_list, dim=2)
            output_list.append(convolved)

        # concat the output to form the final representation
        outputs = mx.sym.concat(*output_list, dim=2)
        if self._dropout != 0.0:
            outputs = mx.sym.Dropout(outputs, p=self._dropout)

        # residual connection and linear transformation
        outputs = mx.sym.broadcast_add(outputs, inputs)
        outputs = mx.sym.dot(outputs, self._linear_W)
        outputs = mx.sym.broadcast_add(outputs, self._linear_b)

        return outputs

    def _convolve(self, adj, inputs, i):
        direct_list = []
        for j in range(self._direction_num):

            weight = self._weight_list[i][j]
            bias = self._bias_list[i][j]

            output = mx.sym.dot(inputs, weight)
            output = mx.sym.broadcast_add(output, bias)

            # retrieve the adj matrix for the jth direction
            label_id = j + 2
            mask = mx.sym.ones_like(adj) * label_id
            adji = (mask == adj)

            output = mx.sym.batch_dot(adji, output)
            output = mx.sym.expand_dims(output, axis=1)
            direct_list.append(output)

        outputs = mx.sym.concat(*direct_list, dim=1)
        outputs = mx.sym.sum(outputs, axis=1)
        # direct_W = self._direct_W[i]
        # direct_b = self._direct_b[i]

        # # direction aggregation
        # outputs = mx.sym.dot(outputs, direct_W)
        # outputs = mx.sym.broadcast_add(outputs, direct_b)

        # normalize the adj matrix
        if self._norm:
            norm_adj = mx.sym.broadcast_not_equal(adj, mx.sym.zeros_like(adj))
            norm_factor = mx.sym.sum(norm_adj, axis=2, keepdims=True)
            outputs = mx.sym.broadcast_div(outputs, norm_factor)

        final_output = mx.sym.Activation(outputs, act_type=self._activation)
        return final_output
