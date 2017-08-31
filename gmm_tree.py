from sklearn.mixture import GaussianMixture as GMM
import numpy as np
import pandas as pd

def load_fasttext_vectors(fname):
    """
    Loads a fasttext word vectors file, returns a DataFrame.
    """
    vocab = []
    vectors = None
    with open(fname) as f:
        header = f.readline()
        vocab_size, vector_size = list(map(int, header.split()))
        vectors = np.empty((vocab_size, vector_size), dtype=np.float)
        for line_no in range(vocab_size):
            linelst = f.readline().strip().split(' ')
            word, vectorlst = linelst[0], [float(x) for x in linelst[1:]]
            vector = np.array(vectorlst, np.float32)
            vocab.append(word)
            vectors[line_no] = vector
    return pd.DataFrame(vectors, index=vocab) 


def normalize(array, l=2, axis=1):
    """
    Normalizes a numpy array to have unit norm of 
    order 'l' along the specified axis.
    """
    div = np.linalg.norm(array, ord=l, axis=axis, keepdims=True)
    div[np.isclose(div, 0)] = 1.
    return array / div


class Node:
    """
    Class representing nodes in a tree, specified by left and
    right child nodes, a mean vector, a set of vectors that is
    associcated with the node and a codeword of bits.
    """
    def __init__(self, vecs, code=None, mean=None):
        self.l = None
        self.r = None
        self.mean = mean
        self.vecs = vecs
        self.code = code

        
class Tree:
    """
    Class representing a binary tree holding a set of word vectors
    at each node.
    """
    def __init__(self):
        self.root = None

    def get_root(self):
        """
        Return the root node of the tree.
        """
        return self.root

    def add(self, vecs, node=None, side=None, mean=None):
        """
        Add the vectors 'vecs' to the node 'node' as left or right
        child node, as specified by 'side'. Additionally provide 
        the mean vector 'mean'.
        """
        if(self.root == None):
            self.root = Node(vecs=vecs, code='0')
        else:
            if (side == 0):
                node.l = Node(vecs=vecs, code=node.code + str(side), mean=mean)
            else:
                node.r = Node(vecs=vecs, code=node.code + str(side), mean=mean)
            
    def get_terminal_vecs(self, node):
        """
        Return the vectors of the node 'node' if it is a terminal
        node, i.e. a leave of the tree.
        """
        if node.l is None and node.r is None:
            return {'vecs': node.vecs, 'code': node.code}
        else:
            return {}
            
    def get_terminal(self, node):
        """
        Get all nodes of the tree that are leaves.
        """
        terminals = []
        node_l = node.l
        node_r = node.r

        if node_l:
            terminals.append(self.get_terminal_vecs(node_l))
            t = self.get_terminal(node_l)
            terminals.extend(t)
        
        if node_r:
            terminals.append(self.get_terminal_vecs(node_r))
            t = self.get_terminal(node_r)
            terminals.extend(t)
        
        terminals = [t for t in terminals if t]
        return terminals
    
    def print_tree(self):
        """
        Recursively print the nodes of tree.
        """
        if(self.root != None):
            self._printTree(self.root)

    def _print_tree(self, node):
        if(node != None):
            self._printTree(node.l)
            print(str(node.vecs) + ' ' + str(node.code))
            self._printTree(node.r)
            
    def get_node(self, code):
        """
        Get the node associated with a binary codeword.
        """
        p = self.root
        for i in code[1:]:
            if i == '0':
                p = p.l
            else:
                p = p.r
        return p
    

def cluster_next_node(t, node, node_codes, verbose=0):
    """
    Given a tree 't' and a node 'node' in the tree, apply GMM based
    clustering to the vectors of the node and add the resulting child
    nodes to the tree. Adds the new codewords to the set 'node_codes'.
    """
    if len(node.vecs) < 2:
        return t, node_codes
    
    elif len(node.vecs) == 2:
        t.add(vecs=node.vecs.iloc[0:1], node=node, side=0, mean=node.vecs.iloc[0])
        t.add(vecs=node.vecs.iloc[1:2], node=node, side=1, mean=node.vecs.iloc[1])
        node_codes.add(node.code + str(0))
        node_codes.add(node.code + str(1))
        return t, node_codes
    else:  
        gmm = GMM(n_components=2, covariance_type='spherical', verbose=0, max_iter=10)
        gmm.fit(np.array(node.vecs))
        mean_vecs = gmm.means_
        classes = gmm.predict(np.array(node.vecs))
        # Add left node
        vecs_left = node.vecs.iloc[np.where(classes==0)]
        t.add(vecs=vecs_left, node=node, side=0, mean=mean_vecs[0])
        node_codes.add(node.code + str(0))
        # Add right node
        vecs_right = node.vecs.iloc[np.where(classes==1)]
        t.add(vecs=vecs_right, node=node, side=1, mean=mean_vecs[1])
        node_codes.add(node.code + str(1))

        if verbose > 0:
            print('Left node code word: {}'.format(node.code + str(0)))
            print('Words in left component: {}'.format(len(vecs_left)))
            print(list(vecs_left.index[:10]))
            print('')
            print('Right node code word: {}'.format(node.code + str(1)))
            print('Words in right component: {}'.format(len(vecs_right)))
            print(list(vecs_right.index[:10]))
            print('')
            print('------------------------------------------------------')
            print('')

        return t, node_codes
    
    
def is_clustering_finished(terminals):
    """
    Given a list of leave nodes, return True if all of them
    only contain a single word, otherwise return False.
    """
    leaf_sizes = [len(node['vecs']) for node in terminals]
    if max(leaf_sizes) > 1:
        return False
    else:
        return True
    
    
def path_to_root(code, nodes_to_index):
    """
    Given the binary codeword for a node, and a dictionary
    of codewords to an indexing of all nodes, return the 
    list of indices from the node to the root of the tree.
    """
    path = []
    try:
        for i in range(1, len(code)):
            path.append(nodes_to_index[code[:i]])
    except:
        print(code)
        print(i)
        
    return path


def save_code_words(leaves, nodes_to_index, out_file):
    """
    Given the leave nodes of the tree and a dictionary mapping
    codewords to an index of all nodes, save the word corresponding
    to the leaves together with the binary codeword and the path
    to the root in terms of the provided index to a file 'out_file'.
    """
    with open(out_file, 'w') as f:
        for n in leaves:
            assert(len(list(n['vecs'].index)) == 1)
            
            w = n['vecs'].index[0]
            c = n['code']
            path = path_to_root(c, nodes_to_index)
            
            # Leave out the '0' bit for the root as it doesn't signify a classification target
            f.write(w + ', ' + c[1:])
            for p in path[:-1]:
                f.write(', ' + str(p))
            f.write('\n')
            
            
def gmm_clustering(vec_file, save_file, verbose=0):
    """
    Given a fastText word vector file 'vec_file', apply hierarchical divisive clustering
    using a 2-component GMM at each step, and store the resulting binary tree to 'save_file'.
    """
    # Start with all vectors at the root
    vecs = load_fasttext_vectors(vec_file)
    vecs = pd.DataFrame(normalize(np.array(vecs)), index=vecs.index)
    nodes_index = dict()
    t = Tree()
    t.add(vecs=vecs)
    node_codes = set(['0'])

    terminals = [{'vecs': t.root.vecs, 'code': t.root.code}]
    while not is_clustering_finished(terminals):

        if verbose > 0:
            print('Number of leaves: {}\n'.format(len(terminals)))

        for next_terminal in terminals:
            t, node_codes = cluster_next_node(t, t.get_node(next_terminal['code']), node_codes, verbose=verbose)

        terminals = t.get_terminal(t.root)

    if verbose > -1:
        # Print final number of leaves, should be the same as the size of the vocabulary
        print('Final number of leaves: {}'.format(len(terminals)))
        print('Size of vocabulary: {}'.format(len(vecs)))
    
    # Remove the leave codes from the internal nodes
    for n in terminals:
        c = n['code']
        if c in node_codes:
            node_codes.remove(c)

    if verbose > -1:
        print('Number of internal nodes: {}'.format(len(node_codes)))
    
    # Enumerate node codes
    nodes_to_index = dict()
    for i, c in enumerate(list(node_codes)):
        nodes_to_index[c] = i
    
    # Compute mean code word lenghts
    code_lengths = [len(node['code']) for node in terminals]
    if verbose > -1:
        print('Mean code word length: {}'.format(np.mean(code_lengths)))
    
    save_code_words(terminals, nodes_to_index, save_file)
