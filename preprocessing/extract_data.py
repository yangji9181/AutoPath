import json
import argparse
import os
import os.path as osp
import numpy as np
import csv
import pickle
from collections import defaultdict
from tqdm import tqdm, trange


class DataSet():
    def __init__(self, data_dir, output_dir, pos_pair_num, neg_sampling_ratio):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.pos_pair_num = pos_pair_num
        self.neg_sampling_ratio = neg_sampling_ratio


class YelpDataSet(DataSet):

    def print_node_attr_distr(self):
        node_attr_counter = defaultdict(int)
        with open(osp.join(self.data_dir, 'business.json')) as bs_json_file:
            for line in bs_json_file:
                line_contents = json.loads(line)
                for attr in get_column_names(line_contents).keys():
                    node_attr_counter[attr] += 1
        print(node_attr_counter)
        with open(osp.join('node_attr_distr.csv'), mode='w') as nad_file:
            csv_writer = csv.writer(nad_file)
            csv_writer.writerow(['node attribute', 'count'])
            for node_count_pair in sorted(node_attr_counter.items(), key=lambda x: x[1], reverse=True):
                csv_writer.writerow(node_count_pair)


    def get_nested_value(self, d, key):
        """Return a dictionary item given a dictionary `d` and a flattened key from `get_column_names`.

        Example:

            d = {
                'a': {
                    'b': 2,
                    'c': 3,
                    },
            }
            key = 'a.b'

            will return: 2

        """
        if '.' not in key:
            if key not in d:
                return None
            return d[key]
        base_key, sub_key = key.split('.', 1)
        if base_key not in d:
            return None
        sub_dict = d[base_key]
        return self.get_nested_value(sub_dict, sub_key)


    def preprocess(self):
        with open(osp.join(self.output_dir, 'node.dat'), mode='w') as node_file:
            with open(osp.join(self.output_dir, 'link.dat'), mode='w') as link_file:
                node_attrs = ['attributes.GoodForKids', 'attributes.RestaurantsTakeOut', 'attributes.OutdoorSeating',
                              'attributes.RestaurantsGoodForGroups', 'attributes.RestaurantsDelivery',
                              'attributes.RestaurantsReservations']

                self.attr_business_dict = {k: [[], []] for k in node_attrs}
                self.business_attr_dict = defaultdict(dict)
                self.businesses = set()
                with open(osp.join(self.data_dir, 'business.json')) as bs_json_file:
                    cities = set()
                    categories = set()
                    stars = set()
                    for line in bs_json_file:
                        line_contents = json.loads(line)
                        line_contents['business_id'] = 'b' + line_contents['business_id']
                        taken = False
                        for node_attr in node_attrs:
                            val = self.get_nested_value(line_contents, node_attr)
                            if val is not None:
                                taken = True
                                self.business_attr_dict[line_contents['business_id']][node_attr] = val
                                if val:
                                    self.attr_business_dict[node_attr][0].append(line_contents['business_id'])
                                else:
                                    self.attr_business_dict[node_attr][1].append(line_contents['business_id'])
                        if not taken:
                            continue

                        line_contents['city'] = line_contents['city'].replace(' ', '_')
                        line_contents['categories'] = [category.replace(' ', '_') for category in
                                                       line_contents['categories']]

                        self.businesses.add(line_contents['business_id'])
                        cities.add(line_contents['city'])
                        categories.update(line_contents['categories'])
                        stars.add(line_contents['stars'])
                        link_file.write(line_contents['business_id'] + '\t' + line_contents['city'] + '\n')
                        link_file.write(line_contents['city'] + '\t' + line_contents['business_id'] + '\n')
                        link_file.write(line_contents['business_id'] + '\t' + str(line_contents['stars']) + '\n')
                        link_file.write(str(line_contents['stars']) + '\t' + line_contents['business_id'] + '\n')
                        for category in line_contents['categories']:
                            link_file.write(line_contents['business_id'] + '\t' + category + '\n')
                            link_file.write(category + '\t' + line_contents['business_id'] + '\n')
                    for star in stars:
                        node_file.write(str(star) + '\ts\n')
                    for city in cities:
                        node_file.write(city + '\tl\n')
                    for category in categories:
                        node_file.write(category + '\tc\n')

                # for k, v in node_attr_dict.items():
                #     print(k + ': ' + str([len(v[0]), len(v[1])]))
                # return

                users = set()
                with open(osp.join(self.data_dir, 'review.json')) as rvw_json_file:
                    bs_usr_pairs = set()
                    for line in rvw_json_file:
                        line_contents = json.loads(line)
                        line_contents['business_id'] = 'b' + line_contents['business_id']
                        line_contents['user_id'] = 'u' + line_contents['user_id']
                        if line_contents['business_id'] in self.businesses:
                            users.add(line_contents['user_id'])
                            bs_usr_pairs.add((line_contents['business_id'], line_contents['user_id']))
                    for bs_usr_pair in bs_usr_pairs:
                        link_file.write(bs_usr_pair[0] + '\t' + bs_usr_pair[1] + '\n')
                        link_file.write(bs_usr_pair[1] + '\t' + bs_usr_pair[0] + '\n')

                # with open(osp.join(self.data_dir, 'user.json')) as usr_json_file:
                #     friend_pairs = set()
                #     # users2 = set()
                #     for line in usr_json_file:
                #         line_contents = json.loads(line)
                #         if line_contents['user_id'] in users:
                #             for friend in line_contents['friends']:
                #                 # users2.add(friend)
                #                 if friend in users:
                #                     friend_pairs.add((line_contents['user_id'], friend))
                #                     friend_pairs.add((friend, line_contents['user_id']))
                #
                # for friend_pair in friend_pairs:
                #     link_file.write(friend_pair[0] + '\t' + friend_pair[1] + '\n')

                # users.update(users2)
                for user in users:
                    node_file.write(user + '\tu\n')

                for business in self.businesses:
                    node_file.write(business + '\tb\n')


    def write_node_classes(self):
        node_classes_dir = osp.join(self.output_dir, 'node_classes')
        if not osp.exists(node_classes_dir):
            os.makedirs(node_classes_dir)

        for cls in self.node_attr_dict:
            with open(osp.join(node_classes_dir, cls + '.txt'), mode='w') as node_class_file:
                for pos_node in self.attr_business_dict[cls][0]:
    #                 print(node)
                    node_class_file.write(pos_node + '\t' + '1' + '\n')
                for neg_node in self.attr_business_dict[cls][1]:
    #                 print(node)
                    node_class_file.write(neg_node + '\t' + '0' + '\n')


    def gen_train_test_pairs(self, test_sampling_ratio):
        node_pairs_dir = osp.join(self.output_dir, 'train_test_node_pairs')
        if not osp.exists(node_pairs_dir):
            os.makedirs(node_pairs_dir)
        with open(osp.join(node_pairs_dir, 'train_pos_node_pairs.txt'), mode='w', newline='') as pnp_file:
            with open(osp.join(node_pairs_dir, 'train_neg_node_pairs.txt'), mode='w', newline='') as nnp_file:
                pos_csv_writer = csv.writer(pnp_file, delimiter='\t')
                neg_csv_writer = csv.writer(nnp_file, delimiter='\t')
                for i in range(self.pos_pair_num):
                    attr = np.random.choice(list(self.attr_business_dict.keys()))
                    business_pair = np.random.choice(self.attr_business_dict[attr][np.random.randint(2)], 2)
                    pos_csv_writer.writerow(business_pair)
                    for j in range(self.neg_sampling_ratio):
                        attr = np.random.choice(list(self.attr_business_dict.keys()))
                        business1 = np.random.choice(self.attr_business_dict[attr][0])
                        business2 = np.random.choice(self.attr_business_dict[attr][1])
                        neg_csv_writer.writerow([business1, business2])

        test_businesses = np.random.choice(list(self.businesses), size=int(len(self.businesses) * test_sampling_ratio),
                                           replace=False)
        test_pos_dict = defaultdict(set)
        for business in test_businesses:
            for node_attr in self.business_attr_dict[business]:
                test_pos_dict[business].update(self.attr_business_dict[node_attr]\
                    [0 if self.business_attr_dict[business][node_attr] else 1])

        # with open(osp.join(self.node_pairs_dir, 'test_pos_dict.p'), mode='rb') as tpd_file:
        #     test_pos_dict = pickle.load(tpd_file)
        # with open(osp.join(self.node_pairs_dir, 'all_businesses.p'), mode='rb') as ab_file:
        #     all_businesses = pickle.load(ab_file)
        test_node_pairs_list = []
        y_tests = []
        for business1 in tqdm(test_pos_dict, desc='Generating test node pairs (1st loop)'):
            node_pairs = []
            y_test = np.empty(0)
            for business2 in tqdm(self.businesses, desc='Generating test node pairs (2nd loop)'):
                # feature_vec = self.get_feature_vec((business1, business2), vec_func)
                node_pairs.append((business1, business2))
                y_test = np.append(y_test, 1 if business2 in test_pos_dict[business1] else 0)
            test_node_pairs_list.append(node_pairs)
            y_tests.append(y_test)

        with open(osp.join(node_pairs_dir, 'test_node_pairs.p'), mode='wb') as tnp_file:
            pickle.dump([test_node_pairs_list, y_tests], tnp_file)
        # with open(osp.join(node_pairs_dir, 'test_pos_dict.p'), mode='wb') as tpd_file:
        #     pickle.dump(test_pos_dict, tpd_file)
        # with open(osp.join(node_pairs_dir, 'all_businesses.p'), mode='wb') as ab_file:
        #     pickle.dump(self.businesses, ab_file)






class IMDbDataSet(DataSet):

    def print_node_attr_distr(self):
        node_attr_counter = defaultdict(int)
        genres = [None]
        with open(osp.join(self.data_dir, 'genre_list.txt')) as gl_file:
            for line in gl_file:
                genres.append(line.strip().split('\t')[1])
        with open(osp.join(self.data_dir, 'movie_genre_rel.txt')) as mgr_file:
            for line in mgr_file:
                node_attr_counter[genres[int(line.strip().split('\t')[1])]] += 1
        print(node_attr_counter)
        with open(osp.join(self.data_dir, 'node_attr_distr.csv'), mode='w') as nad_file:
            csv_writer = csv.writer(nad_file)
            csv_writer.writerow(['node attribute', 'count'])
            for node_count_pair in sorted(node_attr_counter.items(), key=lambda x: x[1], reverse=True):
                csv_writer.writerow(node_count_pair)

    def preprocess(self):
        with open(osp.join(self.output_dir, 'node.dat'), mode='w') as node_file:
            with open(osp.join(self.output_dir, 'link.dat'), mode='w') as link_file:
                for filename in ['actor_list.txt', 'director_list.txt', 'movie_list.txt']:
                    num_lines = sum((1 for line in open(osp.join(self.data_dir, filename))))
                    if filename == 'movie_list.txt':
                        self.num_movies = num_lines
                    for i in range(num_lines):
                        node_file.write(filename[0] + str(i + 1) + '\t' + filename[0] + '\n')
                for filename in ['movie_actor_rel.txt', 'movie_director_rel.txt']:
                    with open(osp.join(self.data_dir, filename)) as fo:
                        for line in fo:
                            line = filename[0] + line.replace('\t', '\t' + filename[6])
                            link_file.write(line)
                users = set()
                for filename in ['train_ratings.txt', 'test_ratings.txt']:
                    with open(osp.join(self.data_dir, filename), newline='') as fo:
                        csv_reader = csv.reader(fo, delimiter='\t')
                        for row in csv_reader:
                            link_file.write('m' + row[0] + '\t' + 'u' + row[1] + '\n')
                            users.add(row[1])
                for user in users:
                    node_file.write('u' + user + '\t' + 'u' + '\n')

    def write_node_classes(self):
        node_classes_dir = osp.join(self.output_dir, 'node_classes')
        if not osp.exists(node_classes_dir):
            os.makedirs(node_classes_dir)

        self.genre_movie_dict = defaultdict(list)
        self.movie_genre_dict = defaultdict(list)
        with open(osp.join(self.data_dir, 'movie_genre_rel.txt')) as mgr_file:
            for line in mgr_file:
                movie_genre_pair = line.strip().split('\t')
                self.genre_movie_dict[int(movie_genre_pair[1])].append(int(movie_genre_pair[0]))
                self.movie_genre_dict[int(movie_genre_pair[0])].append(int(movie_genre_pair[1]))

        for cls in self.genre_movie_dict:
            with open(osp.join(node_classes_dir, 'genre_' + str(cls) + '.txt'), mode='w') as node_class_file:
                for node in self.genre_movie_dict[cls]:
    #                 print(node)
                    node_class_file.write('m' + str(node) + '\n')

    def gen_train_test_pairs(self, test_sampling_ratio):
        node_pairs_dir = osp.join(self.output_dir, 'train_test_node_pairs')
        if not osp.exists(node_pairs_dir):
            os.makedirs(node_pairs_dir)

        with open(osp.join(node_pairs_dir, 'train_pos_node_pairs.txt'), mode='w', newline='') as pnp_file:
            with open(osp.join(node_pairs_dir, 'train_neg_node_pairs.txt'), mode='w', newline='') as nnp_file:
                pos_csv_writer = csv.writer(pnp_file, delimiter='\t')
                neg_csv_writer = csv.writer(nnp_file, delimiter='\t')
                for i in range(self.pos_pair_num):
                    genre = np.random.choice(list(self.genre_movie_dict.keys()))
                    movie_pair = np.random.choice(self.genre_movie_dict[genre], 2)
                    pos_csv_writer.writerow(['m' + str(movie) for movie in movie_pair])
                    for j in range(self.neg_sampling_ratio):
                        genre = np.random.choice(list(self.genre_movie_dict.keys()))
                        movie1 = np.random.choice(self.genre_movie_dict[genre])
                        movie2 = np.random.choice(list(set(range(1, self.num_movies + 1))
                                                       - set(self.genre_movie_dict[genre])))
                        neg_csv_writer.writerow(['m' + str(movie1), 'm' + str(movie2)])

        test_movies = np.random.randint(1, self.num_movies + 1, size=int(self.num_movies * test_sampling_ratio))
        test_pos_dict = defaultdict(set)
        for movie in test_movies:
            for genre in self.movie_genre_dict[movie]:
                test_pos_dict[movie].update(self.genre_movie_dict[genre])

        test_node_pairs_list = []
        y_tests = []
        for movie1 in tqdm(test_pos_dict, desc='Generating test node pairs (1st loop)'):
            node_pairs = []
            y_test = np.empty(0)
            for movie2 in trange(1, self.num_movies + 1, desc='Generating test node pairs (2nd loop)'):
                # feature_vec = self.get_feature_vec((business1, business2), vec_func)
                node_pairs.append(('m' + str(movie1), 'm' + str(movie2)))
                y_test = np.append(y_test, 1 if movie2 in test_pos_dict[movie1] else 0)
            test_node_pairs_list.append(node_pairs)
            y_tests.append(y_test)

        with open(osp.join(node_pairs_dir, 'test_node_pairs.p'), mode='wb') as tnp_file:
            pickle.dump([test_node_pairs_list, y_tests], tnp_file)


class DBLPDataSet(DataSet):

    def preprocess(self):
        with open(osp.join(self.output_dir, 'node.dat'), mode='w') as node_file:
            with open(osp.join(self.output_dir, 'link.dat'), mode='w') as link_file:
                papers = set()
                venues = set()
                authors = set()
                years = set()
                citation_pairs = set()
                with open(osp.join(self.data_dir, 'dblp.txt')) as fo:
                    paper = dict()
                    for line in fo:
                        if line == '\n':
                            papers.add(paper['#index'])
                            papers.update(paper.get('#%', []))
                            try:
                                venues.add(paper['#c'])
                            except KeyError:
                                print(paper)
                                return
                            authors.update(paper.get('#@', []))
                            years.add(paper['#t'])
                            for citation in paper.get('#%', []):
                                citation_pairs.update([(paper['#index'], citation), (citation, paper['#index'])])
                            for author in paper.get('#@', []):
                                link_file.write(paper['#index'] + '\t' + author + '\n')
                                link_file.write(author + '\t' + paper['#index'] + '\n')
                            link_file.write(paper['#index'] + '\t' + paper['#c'] + '\n')
                            link_file.write(paper['#c'] + '\t' + paper['#index'] + '\n')
                            link_file.write(paper['#index'] + '\t' + paper['#t'] + '\n')
                            link_file.write(paper['#t'] + '\t' + paper['#index'] + '\n')
                            paper.clear()
                        else:
                            line = line.strip().replace(' ', '_')

                            if line[1] == 'i':
                                paper[line[:6]] = line[6:]
                            elif line[1] == '%':
                                if line[:2] not in paper:
                                    paper[line[:2]] = []
                                paper[line[:2]].append(line[2:])
                            elif line[1] == '@':
                                paper[line[:2]] = line[2:].split(',_')
                            else:
                                paper[line[:2]] = line[2:]

                for paper in papers:
                    node_file.write(paper + '\tp\n')

                for venue in venues:
                    node_file.write(venue + '\tv\n')

                for author in authors:
                    node_file.write(author + '\ta\n')

                for year in years:
                    node_file.write(year + '\ty\n')

                for citation_pair in citation_pairs:
                    link_file.write(citation_pair[0] + '\t' + citation_pair[1] + '\n')



if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Preprocess HIN datasets.'
    )

    parser.add_argument('--dataset',
                        type=str,
                        required=True,
                        help='dataset to preprocess',
                        choices=['yelp', 'imdb', 'dblp']
                        )
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='directory of the dataset'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        help='directory to store output'
    )
    parser.add_argument(
        '--pos_pair_num',
        type=int,
        help='number of postive node pairs to train and test on'
    )
    parser.add_argument(
        '--neg_sampling_ratio',
        type=int,
        default=1,
        help='sampling ratio of negative pairs compared to positive pairs'
    )
    parser.add_argument(
        '--test_sampling_ratio',
        type=float,
        default=0.001,
        help='sampling ratio of negative pairs compared to positive pairs'
    )
    args = parser.parse_args()
    # print(dir(parser))
    if args.output_dir is None:
        args.output_dir = args.data_dir
    if not osp.exists(args.output_dir):
        os.makedirs(args.output_dir)

    ds = None
    if args.dataset == 'yelp':
        ds = YelpDataSet(args.data_dir, args.output_dir, args.pos_pair_num, args.neg_sampling_ratio)
    elif args.dataset == 'imdb':
        ds = IMDbDataSet(args.data_dir, args.output_dir, args.pos_pair_num, args.neg_sampling_ratio)
    elif args.dataset == 'dblp':
        ds = DBLPDataSet(args.data_dir, args.output_dir, args.pos_pair_num, args.neg_sampling_ratio)
    ds.print_node_attr_distr()
    # ds.preprocess()
    # ds.write_node_classes()
    # ds.gen_train_test_pairs(args.test_sampling_ratio)
