from __future__ import division
import numpy as np
import networkx as nx
import random
import math
import time
from tqdm import tqdm
from sklearn import preprocessing

class LongTNE:
    def __init__(self, net_path, emb_file, net_num, threshold, save_path, emb_size=64, directed=False, learning_rate=0.025):

        self.net_path = net_path
        self.emb_file = emb_file
        self.net_num = net_num
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.emb_size = emb_size
        self.save_path = save_path
        
    def vec2dict(self):
        VecDict = {}
        with open (self.emb_file,"r") as con_vec:
            for line in con_vec.readlines():
                VecDict[int(float(line.strip().split(" ")[0]))] = line.strip().split(" ")[1:]
        return VecDict

    def negative_sample(self, G, n):
        
        degree = len(list(nx.neighbors(G,n)))
        DFS_T = nx.dfs_tree(G, source=n)
        node_list = list(DFS_T)
        random.seed(n)
        negative = random.sample(node_list, math.ceil(int(degree**(0.75))))
        if n in negative: 
            negative.remove(n)

        return negative

    def sigmoid(self, inX):
        if inX >= 0:
            return 1.0 / (1 + np.exp(-inX))
        else:
            return np.exp(inX) / (1 + np.exp(inX))

    def vector_optimization_with_time(self, vec_dict, source, target, alpha, label, rho):

        vec_error = [0] * self.emb_size
        sum_dim = 0
        for ii in range (len(vec_dict[source])):
            sum_dim +=  alpha * (float(vec_dict[source][ii]) * float(vec_dict[target][ii]))
        
        g = (label - self.sigmoid(-sum_dim)) * rho
        
        for jj in range (len(vec_dict[target])):
            vec_error[jj] = vec_error[jj] + g * float(vec_dict[target][jj])
        for kk in range (len(vec_dict[target])):
            vec_dict[target][kk] = float(vec_dict[target][kk]) + g * float(vec_dict[source][kk])
    
        vec_dict[target] = vec_dict[target]
    
        return vec_dict, vec_error

    def Time_Update_main(self, vecdict):
        for filenum in range (self.net_num):  
            subTimeNet = nx.read_weighted_edgelist(self.net_path +str(filenum)+".txt",nodetype=int,create_using=nx.DiGraph()) 
            count,current_count = 0,0
            rho = self.learning_rate
            total_sample = len(subTimeNet.edges())

            for edge in tqdm(subTimeNet.edges()):
                source_node = edge[0]
                NegSample = self.negative_sample(subTimeNet, source_node)
                            
                for i in range (0, len(NegSample)+1):
                    if i == 0:
                        label = 1
                        alpha = 1
                        vecdict,error = self.vector_optimization_with_time(vecdict, source_node, edge[1], alpha, label, rho)
                    else:
                        target_node = NegSample[i-1]
                        try:
                            hop = nx.shortest_path_length(subTimeNet,source_node, target_node)
                            if hop <= self.threshold:
                                label = 1
                                alpha = float(1 / hop)
                                vecdict,error = self.vector_optimization_with_time(vecdict, source_node, target_node, alpha, label, rho)
                            else:
                                label = 0
                                alpha = 1
                                vecdict,error = self.vector_optimization_with_time(vecdict, source_node, target_node, alpha, label, rho)
                        except:
                            label = 0
                            alpha = 1
                            vecdict,error = self.vector_optimization_with_time(vecdict, source_node, target_node, alpha, label, rho)
                            
                count += 1
                        
                if count - current_count > 5000:
                    current_count = count
                    rho = rho * (1 - current_count/ total_sample)
                    if rho < 0.0001: rho = 0.0001  
                else:
                    continue
                
                for hh in range (len(vecdict[source_node])):
                    vecdict[source_node][hh] = float((float(vecdict[source_node][hh]) + error[hh])/(len(NegSample)+1))
                    
        return  vecdict

    def output_embedding_Time(self):
        VECDICT = self.vec2dict()
        Time_embedding = self.Time_Update_main(VECDICT)
        
        ID = list(Time_embedding.keys())
        print ("======Saving Embedding=========")
        f = open(self.save_path,"w")
        for i in tqdm(range (len(ID))):
            f.write(str(ID[i]) + " ")
            for j in Time_embedding[ID[i]]:
                f.write(str(j) + " ")
            f.write("\n")
        
        f.close()


    def embed_normalization(self, output_emb):
        ID = []
        emb_norm = []
        with open (output_emb, "r") as emb:
            for line in tqdm(emb.readlines()):
                vec = []
                ID.append(line.strip().split(" ")[0])
                vec.append(line.strip().split(" ")[1:])
                vec_norm = preprocessing.normalize(np.asarray(vec)).tolist()
                emb_norm.append(vec_norm[0])
                
        return ID,emb_norm



if __name__ == '__main__':

    print (time.asctime(time.localtime(time.time())))
    para_dict = {
        'net_path': 'data/COVID/COVID_Graph_',
        'emb_file': 'baseEmb/COVID/covid_emb_distmult.emb',
        'save_path': 'output/COVID/covid_emb_distmult_enhance.emb',
        'net_num': 3,
        'threshold': 5,
        'learning_rate': 0.025,
        'emb_size': 200,
        'directed': False}

    print ('parameters: \r\n{}'.format(para_dict))


    longtne = LongTNE(net_path=para_dict['net_path'],
                  emb_file=para_dict['emb_file'],
                  save_path=para_dict['save_path'],
                  net_num=para_dict['net_num'],
                  threshold=para_dict['threshold'],
                  learning_rate=para_dict['learning_rate'],
                  emb_size=para_dict['emb_size'],
                  directed=para_dict['directed'])

    longtne.output_embedding_Time()

    # # if neccessary
    # ID, emb_norm = longtne.embed_normalization(para_dict['save_path'])
    # f = open("output/ACTOR/actor_MMDNE_time_enhance_norm.emb","w")
    # for i in tqdm(range (len(ID))):
    #     f.write(str(ID[i]) + " ")
    #     for j in emb_norm[i]:
    #         f.write(str(j) + " ")
    #     f.write("\n")
        
    # # f.close()   
