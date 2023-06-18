import easydict
import node2vec
import networkx as nx
from gensim.models import Word2Vec

def read_graph():
    '''
    Reads the input network in networkx.
    '''
    if args.weighted:
        G = nx.read_edgelist(args.input, nodetype=int, data=(('weight',float),), create_using=nx.Graph())
    else:
        G = nx.read_edgelist(args.input, nodetype=int, create_using=nx.Graph())
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1

    if not args.directed:
        G = G.to_undirected()

    return G
def learn_embeddings(walks):
    '''
    Learn embeddings by optimizing the Skipgram objective using SGD.
    '''
    walks = list(list(map(str, walk)) for walk in walks)
    model = Word2Vec(walks, vector_size=args.dimensions, window=args.window_size, min_count=0, sg=1, workers=args.workers, epochs=args.iter)
    model.wv.save_word2vec_format(args.output)
    #print walks
    return
def read_index(indexfile):
    index_dict = {}
    with open(indexfile,"r") as index:
        for line in index.readlines():
            index_dict[int(line.strip().split("\t")[0])] = str(line.strip().split("\t")[1])
    return index_dict
def main(args):
    '''
    Pipeline for representational learning for all nodes in a graph.
    '''
    print ("==read graph!!!==")
    nx_G = read_graph()
    G = node2vec.Graph(nx_G, args.directed,args.p,args.q)
    G.preprocess_transition_probs()
    walks = G.simulate_walks(args.num_walks, args.walk_length)
    #walks_semantic = walks

    print ("==learning embedding!!!==") 
    learn_embeddings(walks)
    
if __name__ == "__main__":
    args = easydict.EasyDict({
        "input": "../../data/ACTOR/actor_train.txt",
        "output": "../../baseEmb/ACTOR/actor_emb_node2vec.emb",
        "walk_length": 80,
        "dimensions": 64,
        "num_walks": 10,
        "window_size": 5,
        "iter": 5,
        "p":1,
        "q":1,
        "workers": 8,
        "weighted":False,
        "directed":False
    })
    main(args)