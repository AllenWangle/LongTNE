from __future__ import division
import numpy as np
import networkx as nx
import time
from tqdm import tqdm
import random

from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score
from sklearn.linear_model import LogisticRegression

#for the MLP task
import utils

class NTR:
    def __init__(self, emb_file, reconstruction_file, train_ratio, from_file) -> None:

        self.emb_file = emb_file
        self.reconstruction_file = reconstruction_file
        self.train_ratio = train_ratio
        self.from_file = from_file
        

        self.id2emb = {}
        if self.from_file:
            self.format_training_data_from_file()
        else:
            self.id2emb = self.emb_file

    def format_training_data_from_file(self):
        with open(self.emb_file, 'r') as reader:
            for line in reader.readlines():
                embeds = np.fromstring(line.strip(), dtype=float, sep=' ')
                self.id2emb[int(embeds[0])] = embeds[1:]   

    def network_temporal_reconstruction(self):
        nr_x = []
        nr_y = []
        with open(self.reconstruction_file,'r') as re_file:
            for line in re_file:
                tokens = line.strip().split('\t')
                i_emb = self.id2emb[int(tokens[0])]
                j_emb = self.id2emb[int(tokens[1])]
                nr_x.append(abs(i_emb-j_emb))
                nr_y.append(int(tokens[2]))

        x_train, x_valid, y_train, y_valid = train_test_split(nr_x, nr_y, test_size=1 - self.train_ratio, random_state=9)
        lr = LogisticRegression()
        lr.fit(x_train, y_train)
        y_valid_pred_prob = lr.predict_proba(x_valid)[:,1]
        # multi_label classification
        # y_valid_pred_prob = lr.predict_proba(x_valid)
        y_valid_pred_01 = lr.predict(x_valid)
        acc = accuracy_score(y_valid, y_valid_pred_01)
        f1 = f1_score(y_valid, y_valid_pred_01)
        # multi_label classification
        # auc = roc_auc_score(y_valid, y_valid_pred_prob, multi_class="ovr")
        auc = roc_auc_score(y_valid, y_valid_pred_prob)
        print ('The accuracy (ACC) of NTR is :{}'.format(acc))
        print ('The AUC of NTR is :{}'.format(auc))
        print ('The F1 of NTR is :{}'.format(f1))


class MLP:

    def __init__(self, train_file, test_file, test_neg_file, sample_num, emb_file, epoch) -> None:

        self.train_file = train_file
        self.test_file = test_file
        self.test_neg_file = test_neg_file
        self.sample_num = sample_num
        self.emb_file = emb_file
        self.epoch = epoch


    # for the multi-hop link prediction task
    def generate_neg_link(self, gtrain, gtest, start_node):
        neg_node = []
        node_index = list(gtrain.nodes())
        while len(neg_node) < self.sample_num:
            sample = random.randint(0,len(node_index)-1)
            if node_index[sample] != start_node and node_index[sample] not in gtest.neighbors(start_node) and node_index[sample] not in gtrain.neighbors(start_node):
                neg_node.append(node_index[sample])
        return neg_node[0]
    
    def dot_product_LP(self, edges, emb, EmbMap):
        score_res = []
        for i in range(len(edges)):
            score_res.append(np.dot(emb[EmbMap[str(int(edges[i][0]))]],emb[EmbMap[str(int(edges[i][1]))]]))
        test_label = np.array(score_res)
        median = np.median(test_label)
        index_pos = test_label >= median
        index_neg = test_label < median
        test_label[index_pos] = 1
        test_label[index_neg] = 0
        true_label = np.zeros(test_label.shape)
        true_label[0: len(true_label) // 2] = 1
        accuracy = accuracy_score(true_label, test_label)
        auc = roc_auc_score(true_label, score_res)
        if auc < 0.5:
            auc = 1 - auc
        return accuracy, auc

    def multihop_link_prediction(self, dim=64):
        acc = []
        auc = []
        e = 0

        G_train = nx.read_weighted_edgelist(self.train_file,nodetype=int,create_using=nx.Graph())
        n_node = len(G_train.nodes())
        G_test = nx.read_weighted_edgelist(self.test_file,nodetype=int,create_using=nx.Graph())

        while  e < self.epoch:
        
            print("Generate the negative samples")
            neg_sample_link = []
            for edge in tqdm(G_test.edges()):
                neg_sample_link.append([edge[0], self.generate_neg_link(G_train,G_test,edge[0])])
            np.savetxt(self.test_neg_file,np.asarray(neg_sample_link),fmt="%s",newline="\n",delimiter="\t")

            test_edge = utils.read_edges_from_file(self.test_file)
            test_edge_neg = utils.read_edges_from_file(self.test_neg_file)
            test_edge.extend(test_edge_neg)

            EMB,EMBMAP = utils.read_embeddings(self.emb_file,n_node,dim)

            acc_node, auc_node = self.dot_product_LP(test_edge,EMB,EMBMAP)
            print("The Acc is %f and AUC is %f in this round." % (acc_node, auc_node))
            acc.append(acc_node)
            auc.append(auc_node)
            e = e + 1
    
        print("Max value:")
        print("The max Acc is %f, and the max AUC is %f" % (max(acc),max(auc)))
        print("Mean value:")
        print("The mean Acc is %f, and the mean AUC is %f" % (float(sum(acc)/len(acc)),float(sum(auc)/len(auc))))

class TVR:
    def __init__(self, train_file, test_file, emb_file) -> None:
        self.train_file = train_file
        self.test_file = test_file
        self.emb_file = emb_file

    def euclidean_distance(self,start_node,node_vector_dict):
        distance_list = [] 
        start_node_vector = node_vector_dict[start_node]
        for i in node_vector_dict.keys():            
            # Euclidean distance
            if i != start_node:
                ith_node_vector = node_vector_dict[i]
                distance_value = np.linalg.norm(start_node_vector - ith_node_vector)
                distance_list.append([i,float(distance_value)])
        # sorted
        distance_list_sort = sorted(distance_list,key = lambda data:data[1])

        return distance_list_sort

    def remove_direct_neighbor(self,rec_list,node,g_train):
        for rec in rec_list:
            if rec in g_train.neighbors(node):
                rec_list.remove(rec)
                
        return rec_list


    def hits_precision_filter(self,node,g_test,recommend_dict,length):
        nbr = list(g_test.neighbors(node))
        count = 0
        for i in recommend_dict[node][0:length]:
            if i in nbr:
                count = count + 1
            else:
                continue
        hit_acc = float(count / len(nbr))
            
        return hit_acc

    def temporal_vertex_recommend(self):
        node_vector = {}
        recommend_dict = {}

        g = nx.read_weighted_edgelist(self.test_file, nodetype=int,create_using=nx.Graph())
        gtrain = nx.read_weighted_edgelist(self.train_file, nodetype=int,create_using=nx.Graph())

        with open (self.emb_file,"r") as f:
            for line in f.readlines():
                if int(float(line.strip().split(" ")[0])) in g.nodes():
                    vec = [float(i) for i in line.strip().split(" ")[1:]]
                    node_vector[int(float(line.strip().split(" ")[0]))] = np.asarray(vec)      
        
        for node in tqdm(g.nodes()):
            recommend_list = self.euclidean_distance(node,node_vector)
            recommend_dict[node] = [int(float(recommend[0])) for recommend in recommend_list]
            recommend_dict[node] = self.remove_direct_neighbor(recommend_dict[node],node,gtrain)
            
        print ("calculation for HIT@k")
        k = [5,10,15,20]
        for l in tqdm(k): 
            hit = []
            for node in g.nodes():
                hit.append(self.hits_precision_filter(node,g,recommend_dict,l))

            print ("the hit raion of top@%d is: %f" % (l,float(sum(hit)/len(hit))))


if __name__ == "__main__":
    print (time.asctime(time.localtime(time.time())))

    print("======Network Temporal Reconstruction!======")
    para_NTR = {
        'emb_file': 'output/ACTOR/actor_dynwalks_time_enhance.emb',
        'reconstruction_file': 'dataEval/ACTOR/actor_recons_1%.txt',
        'train_ratio': 0.8,
        'from_file': True} 

    ntr = NTR(
        emb_file=para_NTR["emb_file"],
        reconstruction_file=para_NTR["reconstruction_file"],
        train_ratio=para_NTR["train_ratio"],
        from_file=para_NTR["from_file"])
    ntr.network_temporal_reconstruction()

    print("======Multi-hop Link Prediction!======")
    para_MLP = {
        'train_file': 'data/ACTOR/actor_train.txt',
        'test_file': 'dataEval/ACTOR/actor_test.txt',
        'test_neg_file': "dataEval/ACTOR/actor_test_neg.txt",
        'emb_file': "output/ACTOR/actor_dynwalks_time_enhance.emb",
        'sample_num': 1,
        'epoch': 2}

    mlp = MLP(
        train_file=para_MLP["train_file"],
        test_file=para_MLP["test_file"],
        test_neg_file=para_MLP["test_neg_file"],
        emb_file=para_MLP["emb_file"],
        sample_num=para_MLP["sample_num"],
        epoch=para_MLP["epoch"])
    mlp.multihop_link_prediction()


    print("======Temporal Vertex Recommendation!======")
    para_TVR = {
        'train_file': 'data/ACTOR/actor_train.txt',
        'test_file': 'dataEval/ACTOR/actor_vertex_recommend.txt',
        'emb_file': "output/ACTOR/actor_dynwalks_time_enhance.emb"}

    tvr = TVR(
        train_file=para_TVR["train_file"],
        test_file=para_TVR["test_file"],
        emb_file=para_TVR["emb_file"])
    tvr.temporal_vertex_recommend()










    

