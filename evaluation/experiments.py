
# coding: utf-8

# In[1]:


# get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:



# coding: utf-8

# In[ ]:



# coding: utf-8

# In[27]:


import csv
import numpy as np
from collections import defaultdict
import os.path as osp
from tqdm import trange, tqdm
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
import heapq
from scipy.sparse import load_npz
import pickle
import argparse


class Model():

    def __init__(self, data_dir):
        self.data_dir = data_dir
        # self.node_pairs_dir = osp.join(self.data_dir, 'train_test_node_pairs')
        self.node_pairs_dir = data_dir

    def get_test_node_pairs(self):
        with open(osp.join(self.node_pairs_dir, 'test_node_pairs.p'), mode='rb') as tnp_file:
            self.test_node_pairs_list, self.y_tests = pickle.load(tnp_file)

        # with open(osp.join(self.node_pairs_dir, 'test_pos_dict.p'), mode='rb') as tpd_file:
        #     test_pos_dict = pickle.load(tpd_file)
        # with open(osp.join(self.node_pairs_dir, 'all_businesses.p'), mode='rb') as ab_file:
        #     all_businesses = pickle.load(ab_file)
        # self.test_node_pairs_list = []
        # self.y_tests = []
        # for business1 in tqdm(test_pos_dict, desc='Getting test node pairs (1st loop)'):
        #     node_pairs = []
        #     y_test = np.empty(0)
        #     for business2 in tqdm(all_businesses, desc='Getting test node pairs (2nd loop)'):
        #         # feature_vec = self.get_feature_vec((business1, business2), vec_func)
        #         node_pairs.append((business1, business2))
        #         y_test = np.append(y_test, 1 if business2 in test_pos_dict[business1] else 0)
        #     self.test_node_pairs_list.append(node_pairs)
        #     self.y_tests.append(y_test)

        # return X_tests, y_tests


class EmbeddingModel(Model):

    def __init__(self, embedding_size, baseline, dataset, data_dir):
        super(EmbeddingModel, self).__init__(data_dir)
        self.embedding_size = embedding_size
        self.baseline = baseline
        self.dataset = dataset
        self.data_dir = data_dir
    
    def construct_embd_dict(self):
        self.embedding_dict = defaultdict(list)
        if self.baseline == 'esim':
            with open(self.data_dir + 'esim_embedding.txt') as embedding_file:
                csv_reader = csv.reader(embedding_file, delimiter=',')
                for row in csv_reader:
                    if len(row) != self.embedding_size + 1:
                        del row[1]
                    self.embedding_dict[row[0]] = [float(e) for e in row[1:]]
        elif self.baseline == 'pte':
            with open('/shared/data/xikunz2/autopath/hin2vec/embedding.txt') as embedding_file:
                csv_reader = csv.reader(embedding_file, delimiter='\t')
                for row in csv_reader:
                    self.embedding_dict[row[0]] = [float(e) for e in row[1].split(' ')[:-1]]
        elif self.baseline == 'metapath2vec':
            with open('/shared/data/xikunz2/autopath/metapath2vec/results/' + self.dataset +
                      '_embedding_w100_l10.txt') as embedding_file:
                csv_reader = csv.reader(embedding_file, delimiter=' ')
                for i, row in enumerate(csv_reader):
                    if i > 0:
                        self.embedding_dict[row[0][1:]] = [float(e) for e in row[1:-1]]


    # In[23]:


    def get_feature_vec(self, bs_pair, vec_func):
        embedding1 = np.array(self.embedding_dict[bs_pair[0]])
        embedding2 = np.array(self.embedding_dict[bs_pair[1]])
        if vec_func == 'hadamard':
            feature_vec = np.multiply(embedding1, embedding2)
        elif vec_func == 'concatenation':
            feature_vec = np.concatenate((embedding1, embedding2))
        else:
            feature_vec = None
        return feature_vec

    def get_train_feature_vecs(self, vec_func='hadamard'):
        train_feature_vecs_list = []
        for filename in tqdm(['train_pos_node_pairs.txt', 'train_neg_node_pairs.txt'],
                             desc='Getting training feature vectors (1st loop)'):
            train_feature_vecs = np.empty((0, self.embedding_size))
            with open(osp.join(self.node_pairs_dir, filename), newline='') as np_file:
                csv_reader = csv.reader(np_file, delimiter='\t')
                for business_pair in tqdm(csv_reader, desc='Getting training feature vectors (2nd loop)'):
                    feature_vec = self.get_feature_vec(business_pair, vec_func)
                    train_feature_vecs = np.append(train_feature_vecs, feature_vec.reshape((1, -1)), axis=0)
            train_feature_vecs_list.append(train_feature_vecs)
        X_train = np.concatenate((train_feature_vecs_list[0], train_feature_vecs_list[1]))
        y_train = np.concatenate((np.ones(train_feature_vecs_list[0].shape[0]),
                                 np.zeros(train_feature_vecs_list[1].shape[0])))
        return X_train, y_train

    def get_test_feature_vecs(self, vec_func='hadamard'):
        X_tests = []
        for node_pairs in tqdm(self.test_node_pairs_list, desc='Getting testing feature vectors (1st loop)'):
            X_test = np.empty((0, self.embedding_size))
            # for node_pair in tqdm(node_pairs[:700], desc='Getting testing feature vectors (2nd loop)'):
            for node_pair in tqdm(node_pairs, desc='Getting testing feature vectors (2nd loop)'):
                feature_vec = self.get_feature_vec(node_pair, vec_func)
                X_test = np.append(X_test, feature_vec.reshape((1, -1)), axis=0)
            X_tests.append(X_test)

        # return X_tests, [y_test[:700] for y_test in self.y_tests]
        return X_tests, self.y_tests

# class PathSimModel(Model):
#
#     def sample_node_pairs(self, pos_or_neg, train_size):
#         if pos_or_neg == 'pos':
#             test_indices = np.random.choice(num_pos_node_pairs,
#                                             size=int(num_pos_node_pairs * (1 - train_size)), replace=False)
#         else:
#             test_indices = np.random.choice(num_pos_node_pairs * neg_sampling_ratio,
#                                             size=int(num_pos_node_pairs * neg_sampling_ratio * (1 - train_size)),
#                                             replace=False)
#         node_pair_list = []
#         with open(osp.join(self.data_dir, pos_or_neg + '_node_pairs.txt'), newline='') as pnp_file:
#             csv_reader = csv.reader(pnp_file, delimiter='\t')
#             for i, business_pair in tqdm(enumerate(csv_reader), desc='Processing ' + pos_or_neg + '_node_pairs.txt'):
#                 if i in test_indices:
#                     node_pair_list.append(business_pair)
#         return node_pair_list

# In[ ]:



def precision_at_k(y_true, y_score, k):
    k_hightest_id = heapq.nlargest(k, range(len(y_score)), y_score.take)
    k_labels = y_true[k_hightest_id]
    return np.sum(k_labels) / k

def recall_at_k(y_true, y_score, k):
    k_hightest_id = heapq.nlargest(k, range(len(y_score)), y_score.take)
    k_labels = y_true[k_hightest_id]
    return np.sum(k_labels) / np.sum(y_true)

def precision_recall_list(y_true, y_score):
    precisions = [precision_at_k(y_true, y_score, k) for k in range(5, 35, 5)]
    recalls = [recall_at_k(y_true, y_score, k) for k in range(5, 35, 5)]
    return precisions, recalls

def calculate_results(baseline, dataset, data_dir):
    if baseline == 'pathsim':
        metapaths = []
        metapath_weights = []
        with open('../data/'+dataset+'/path.dat') as path_file:
            for line in path_file:
                toks = line.strip().split(" ")
                metapaths.append(toks[0])
                metapath_weights.append(float(toks[1]))

        with open(data_dir + '/' + dataset + '_node_hash.p', mode='rb') as node_hash_file:
            node_hash = pickle.load(node_hash_file)
        # pos_node_pair_list = sample_node_pairs('pos', train_size)
        # neg_node_pair_list = sample_node_pairs('neg', train_size)

        pathsim_model = Model(data_dir)
        pathsim_model.get_test_node_pairs()

        # 3-D list
        pathsim_list = []
        for metapath in tqdm(metapaths, desc='Calculating "single_pathsim_list" for a specific metapath'):
            node_type = metapath[0]
            # pathsim list for a specific metapath (2-D)
            single_pathsim_list = []
            cmt_mtx = load_npz(data_dir + '/' + dataset + '_cmt_mtx_' + metapath + '.npz')
            for node_pairs in tqdm(pathsim_model.test_node_pairs_list, desc='Processing a node pair list'):
                single_pathsim_list.append([])
                for node_pair in tqdm(node_pairs, desc='Processing a node pair'):
                    i1 = node_hash[node_type][node_pair[0]]
                    i2 = node_hash[node_type][node_pair[1]]
                    single_pathsim = 2 * cmt_mtx[i1, i2]
                    if cmt_mtx[i1, i1] + cmt_mtx[i2, i2] == 0:
                        single_pathsim = 0
                    else:
                        single_pathsim /= cmt_mtx[i1, i1] + cmt_mtx[i2, i2]
                    single_pathsim_list[-1].append(single_pathsim)
            pathsim_list.append(single_pathsim_list)

        #2-D numpy array
        y_scores = np.average(pathsim_list, axis=0, weights=metapath_weights)
        y_tests = pathsim_model.y_tests

    elif baseline == 'autopath':
        all_node_name_file = '../data/'+data_dir+'/node.dat'
        test_node_name_file = '../data/'+data_dir+'/test_nodes.txt'
        score_file = data_dir+'/rank_list.pkl'
        node_names = []
        test_node_names = []
        with open(all_node_name_file, 'r') as f:
            for line in f:
                tokens = line.strip().split('\t')
                if tokens[1] == 'm':
                    node_names.append(tokens[0])
        with open(test_node_name_file, 'r') as f:
            for line in f:
                test_node_names.append(line.strip())



    else:
        embedding_size = 50

        embedding_model = EmbeddingModel(embedding_size, baseline, dataset, data_dir)
        embedding_model.construct_embd_dict()
        X_train, y_train = embedding_model.get_train_feature_vecs()

        # X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size)

        classifier = svm.LinearSVC()
        classifier.fit(X_train, y_train)

        embedding_model.get_test_node_pairs()
        X_tests, y_tests = embedding_model.get_test_feature_vecs()
        y_scores = [classifier.decision_function(X_test) for X_test in X_tests]

    precisions_list = []
    recalls_list = []
    auc_list = []
    for y_test, y_score in zip(y_tests, y_scores):
        precisions, recalls = precision_recall_list(y_test, y_score)
        auc = roc_auc_score(y_test, y_score)
        precisions_list.append(precisions)
        recalls_list.append(recalls)
        auc_list.append(auc)

    precisions = np.average(precisions_list, axis=0)
    recalls = np.average(recalls_list, axis=0)
    auc = np.average(auc_list)
    auc_std = np.std(auc_list)

    print(baseline + ' precision list: ')
    print(precisions)
    print(baseline + ' recall list: ')
    print(recalls)
    print(baseline + ' roc curve auc: ')
    print(auc)
    print(baseline + ' roc curve auc standard deviation: ')
    print(auc_std)

    return precisions, recalls



# In[ ]:





# In[ ]:


if __name__ == '__main__':
    # dataset = 'yelp'
    # data_dir = '/shared/data/xikunz2/autopath/yelp_data'
    # num_pos_node_pairs = 40000
    # neg_sampling_ratio = 1
    # generate_node_pairs(num_pos_node_pairs, neg_sampling_ratio)
    parser = argparse.ArgumentParser(
        description='Evaluation. '
    )

    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='dataset to run pathsim on',
        choices=['yelp', 'imdb', 'dblp']
                        )

    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='directory of the dataset'
    )
    # parser.add_argument(
    #     '--num_pos_node_pairs',
    #     type=int,
    #     required=True,
    #     help='number of positive node pairs'
    # )
    # parser.add_argument(
    #     '--neg_sampling_ratio',
    #     type=int,
    #     default=1,
    #     help='sampling ratio of negative node pairs'
    # )
    args = parser.parse_args()

    # In[4]:

    # train_size = 0.7
    #baselines = ['esim', 'metapath2vec', 'pathsim']
    baselines = ['pathsim', 'esim']
    dataset = 'imdb'
    baseline_performance = dict()
    for bsl in tqdm(baselines, desc='Running a specific baseline'):
        # tuple of length 3
        baseline_performance[bsl] = calculate_results(bsl, args.dataset, args.data_dir)

   


    # In[ ]:


    # Plot all precision curves
    fig1 = plt.figure()
    for baseline in baselines:
        plt.plot(range(5, 35, 5), baseline_performance[baseline][0], label=baseline)
    plt.xlabel('k')
    plt.ylabel('Precision @ k')
    #plt.title(dataset + ' - Precision @ k')
    # plt.legend(loc="lower right")
    plt.legend()



    # In[ ]:


    #Plot all recall curves

    fig2 = plt.figure()
    for baseline in baselines:
        plt.plot(range(5, 35, 5), baseline_performance[baseline][1], label=baseline)
    plt.xlabel('k')
    plt.ylabel('Recall @ k')
    #plt.title(dataset + ' - Recall @ k')
    # plt.legend(loc="lower right")
    plt.legend()



# plt.plot(fpr["micro"], tpr["micro"],
#          label='micro-average ROC curve (area = {0:0.2f})'
#                ''.format(roc_auc["micro"]),
#          color='deeppink', linestyle=':', linewidth=4)
#
# plt.plot(fpr["macro"], tpr["macro"],
#          label='macro-average ROC curve (area = {0:0.2f})'
#                ''.format(roc_auc["macro"]),
#          color='navy', linestyle=':', linewidth=4)
#
# colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
# for i, color in zip(range(n_classes), colors):
#     plt.plot(fpr[i], tpr[i], color=color, lw=lw,
#              label='ROC curve of class {0} (area = {1:0.2f})'
#                    ''.format(i, roc_auc[i]))
#
# plt.plot([0, 1], [0, 1], 'k--', lw=lw)
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])


    # plt.show()
    fig1.savefig(dataset + '_pre.png')
    fig2.savefig(dataset + '_rec.png')
