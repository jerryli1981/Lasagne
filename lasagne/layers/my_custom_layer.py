import theano.tensor as T
import theano

from .base import Layer
from .. import init

__all__ = [
    "AbsLayer",
    "RecursiveLayer",
]


class AbsLayer(Layer):
    def get_output_for(self, input, **kwargs):
        return T.abs_(input)


class RecursiveLayer(Layer):
    def __init__(self, incoming_dep_trees, input_dim, n_rel, **kwargs):
        super(RecursiveLayer, self).__init__(incoming_dep_trees, **kwargs)


        """
        WR_values = numpy.asarray(

            rng.uniform(
            	low=-np.sqrt(6. / (intput_dim + intput_dim)),
                high=np.sqrt(6. / (intput_dim + intput_dim)),
                size=(n_rel, intput_dim, intput_dim)
                ),

            dtype=theano.config.floatX)

        self.WR = theano.shared(value=WR_values, name='WR',
                          borrow=True)
        """

        WR = init.Uniform(range=np.sqrt(6. / (input_dim + input_dim)))
        self.WR = self.add_param(WR, (n_rel, input_dim, input_dim), name="WR")

        """
        WV_values = numpy.asarray(

            rng.uniform(
                    low=-np.sqrt(6. / (input_dim + input_dim)),
                    high=np.sqrt(6. / (input_dim + input_dim)),
                    size=(input_dim, input_dim)
                    ),

            dtype=theano.config.floatX)

        self.WV = theano.shared(value=WV_values, name='WV',
                          borrow=True)
        """

        WV = init.Uniform(range=np.sqrt(6. / (input_dim + input_dim)))
        self.WV = self.add_param(WV, (input_dim, input_dim), name="WV")


        """
        b_values=numpy.zeros((intput_dim,), 
                             dtype=theano.config.floatX)
        b = theano.shared(value=b_values, name='b',
                          borrow=True)
		"""

        b = init.Constant(0.)
        self.b = self.add_param(b, (input_dim), name="b")
	
        self.dep_tree = dep_tree
        self.input_dim = input_dim

    def get_output_for(self, tree, **kwargs):

        #because many training epoch. 
        tree.resetFinished()

        to_do = []
        to_do.append(tree.root)

        while to_do:
        
            curr = to_do.pop(0)
            curr.vec = self.L[:, curr.index]

            # node is leaf
            if len(curr.kids) == 0:

                # activation function is the normalized tanh
                #curr.hAct = norm_tanh(np.dot(curr.vec, self.WV) + self.b)
                curr.hAct = np.tanh(np.dot(curr.vec, self.WV) + self.b)

                curr.finished=True

            else:

                #check if all kids are finished
                all_done = True
                for index, rel in curr.kids:
                    node = tree.nodes[index]
                    if not node.finished:
                        to_do.append(node)
                        all_done = False

                if all_done:

                    sum = np.zeros(self.wvecDim)
                    for i, rel in curr.kids:
                        W_rel = self.WR[rel.index] # d * d
                        sum += np.dot(tree.nodes[i].hAct, W_rel) 

                    #curr.hAct = norm_tanh(sum + np.dot(curr.vec, self.WV) + self.b)
                    curr.hAct = np.tanh(sum + np.dot(curr.vec, self.WV) + self.b)
        
                    curr.finished = True

                else:
                    to_do.append(curr)
        
        return tree.root.hAct

    def get_output_shape_for(self, input_shape):
    	return (input_shape[0], self.input_dim)







