import theano.tensor as T
import theano

from .base import Layer
from .. import init

import numpy as np

import theano.typed_list

__all__ = [
    "AbsLayer",
    "RecursiveLayer",
]


class AbsLayer(Layer):
    def get_output_for(self, input, **kwargs):
        return T.abs_(input)


class RecursiveLayer(Layer):
    def __init__(self, incoming, input_dim, word2vecs, n_rel, input_shape, **kwargs):

        super(RecursiveLayer, self).__init__(incoming, **kwargs)

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

        vocab_size = word2vecs.shape[1]
        self.L = self.add_param(word2vecs, (input_dim, vocab_size), name="L")


        """
        b_values=numpy.zeros((intput_dim,), 
                             dtype=theano.config.floatX)
        b = theano.shared(value=b_values, name='b',
                          borrow=True)
		"""

        b = init.Constant(0.)
        self.b = self.add_param(b, (input_dim,), name="b")
	
        self.input_dim = input_dim

        (mini_batch, seq_len, n_children, n_ele) = input_shape

        self.mb = mini_batch
        self.seq_len = seq_len
        self.n_children = n_children
        self.n_ele = n_ele
        
    def get_output_for(self, input, **kwargs):

        for tree_idx in range(self.mb):
            
            root = input[tree_idx, -1]

            if root[0,0] != -1:
                print "correct"
            if root[0,-1] == -1:
                print "wrong"
            if root[0,-1] == -2:
                print "wrong"
            if root[0,-1] == 0:
                print "wrong"
            if root[0,0] == 1687:
                print "wrong"
    
            to_do = []
            to_do.append(root)

            #tree.resetFinished() -2: unfinished, -3:finished
            """
            for node_idx in range(self.seq_len):
                node = input[batch_idx, node_idx]
                T.set_subtensor(node[:,-1], -2)
            """
            #(govGlobalId, depLocalIdx, depGlobalIdx, relIdx, flag)
            #kid_hacts = []
            #kid_hacts = theano.typed_list.TypedListType(theano.tensor.fvector)()

            while to_do:
                print "1"
                curr = to_do.pop(0)

                curr_vec = self.L[:, curr[0,0]]

                #node is leaf
                #if len(curr.kids) == 0:
                if curr[0,1] == -1:
                    print "2"
                    # activation function is the normalized tanh
                    #curr.hAct = norm_tanh(np.dot(curr.vec, self.WV) + self.b)
                    curr_hAct = T.tanh(T.dot(curr_vec, self.WV) + self.b)

                    root_hAct = curr_hAct

                    #kid_hacts.append(curr_hAct)
                    #o = theano.typed_list.append(kid_hacts, curr_hAct)
                    #theano.typed_list.basic.append(kid_hacts, curr_hAct)

                    #curr.finished=True, if finished set flag = -3 else flag = -2
                    #curr[:,5] = -3
                    T.set_subtensor(curr[:,-1], -3)

                else:

                    print "3"
                    #check if all kids are finished
                    all_done = True

                    """
                    for index, rel in curr.kids:
                        node = tree.nodes[index]
                        if not node.finished:
                            to_do.append(node)
                            all_done = False
                    """

                    for kid_idx in range(self.n_children):

                        print "4"


                        if curr[kid_idx, 0] == -1:
                            continue

                        kid_local_idx = curr[kid_idx, 1] 

                        kid_node = input[tree_idx, kid_local_idx]

                        if kid_node[0, 4] == -2:

                            print "5"
                            to_do.append(kid_node_m)
                            all_done = False

                    if all_done:
                        print "6"

                        sum_ = T.zeros(self.input_dim)

                        """
                        for i, rel in curr.kids:
                            W_rel = self.WR[rel.index] # d * d
                            sum_ += T.dot(tree.nodes[i].hAct, W_rel)

                        """

                        """
                        real_kid_idx = 0
                        for kid_idx in range(max_children):
                            kid_local_idx = curr[kid_idx, 2] 
                            kid_global_idx = curr[kid_idx, 3]
                            if kid_local_idx != -1 and kid_global_idx != -1:
                                
                                kid_node = input[batch_idx, kid_local_idx]
                                rel_idx = kid_node[kid_idx, 4]
                                W_rel = self.WR[rel_idx]
                                kid_node_hAct = theano.typed_list.basic.getitem(kid_hacts, real_kid_idx)
                                real_kid_idx +=  1
                                sum_ += T.dot(kid_node_hAct, W_rel)
                        """

                        #kid_hacts = []

                        #curr.hAct = norm_tanh(sum + np.dot(curr.vec, self.WV) + self.b)
                        #curr.hAct = T.tanh(sum_ + T.dot(curr.vec, self.WV) + self.b)
            
                        #curr.finished = True
                        T.set_subtensor(curr[:,-1], -3)

                    else:
                        print "7"
                        to_do.append(curr)
            
            return root_hAct

            

    def get_output_shape_for(self, input_shape):
    	return (input_shape[0], self.input_dim)







