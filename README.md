# hs-semantic-tree
Training word embeddings using hierarchical softmax with a semantic tree.

In hierarchical softmax, the regular softmax layer is replaced by a binary tree, where at each node of the tree a classifier is trained to select the child node containing the center word given its context (see Morin & Bengio (2005) for details). By default the implementations in word2vec and fastText use a Huffman tree based on word frequency. However, this does not contain any semantic information so that training consistent classifiers at each node might be hard.

Based on the work of Mnih and Hinton (2008) who showed that semantic trees can improve language models compared to random trees, we use GMM clustering of initial word vectors to derive a tree and feed this to fastText to replace the Huffman tree in a second round of training. In accordance with their results, this also improves the performance in word analogy and word similarity tasks.

### Code
The repository contains the iPyhton notebook <em>cbow_hs_huffman_vs_semantic_tree</em> to run the experiment. A modified version of <a href="https://github.com/facebookresearch/fastText">fastText</a> that reads the resulting tree file is provided <a href="https://github.com/mleimeister/fastText/tree/hs_precomputed_tree">here</a>. It needs to be compiled and the paths in the iPython notebook adjusted to point to the <em>fasttext</em> executable and training data file. The file <em>gmm_tree.py</em> contains functions for clustering a set of word vectors into a binary tree and storing it in a format that can be read by the modified fastText version.

More details on the experiment and results can be found in this <a href="https://blog.lateral.io/2017/08/semantic-trees-hierarchical-softmax">blog post</a>.

## References 

Morin, F., & Bengio, Y. (2005). Hierarchical Probabilistic Neural Network Language Model. Aistats, 5.

Mnih, A., & Hinton, G. E. (2008). A Scalable Hierarchical Distributed Language Model. Advances in Neural Information Processing Systems, 1–8.
