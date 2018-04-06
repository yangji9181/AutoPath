import json
import argparse
import os
import os.path as osp
import numpy as np
import csv
import pickle
from collections import defaultdict


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
                self.node_attr_dict = {k: [[], []] for k in node_attrs}
                businesses = set()
                with open(osp.join(self.data_dir, 'business.json')) as bs_json_file:
                    cities = set()
                    categories = set()
                    stars = set()
                    for line in bs_json_file:
                        line_contents = json.loads(line)
                        taken = False
                        for node_attr in node_attrs:
                            val = self.get_nested_value(line_contents, node_attr)
                            if val is not None:
                                taken = True
                                if val:
                                    self.node_attr_dict[node_attr][0].append(line_contents['business_id'])
                                else:
                                    self.node_attr_dict[node_attr][1].append(line_contents['business_id'])
                        if not taken:
                            continue

                        line_contents['city'] = line_contents['city'].replace(' ', '_')
                        line_contents['categories'] = [category.replace(' ', '_') for category in
                                                       line_contents['categories']]
                        businesses.add(line_contents['business_id'])
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
                        if line_contents['business_id'] in businesses:
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

                for business in businesses:
                    node_file.write(business + '\tb\n')


    def write_node_classes(self):
        node_classes_dir = osp.join(self.output_dir, 'node_classes')
        if not osp.exists(node_classes_dir):
            os.makedirs(node_classes_dir)

        for cls in self.node_attr_dict:
            with open(osp.join(node_classes_dir, cls + '.txt'), mode='w') as node_class_file:
                for pos_node in self.node_attr_dict[cls][0]:
    #                 print(node)
                    node_class_file.write(pos_node + '\t' + '1' + '\n')
                for neg_node in self.node_attr_dict[cls][1]:
    #                 print(node)
                    node_class_file.write(neg_node + '\t' + '0' + '\n')


    def gen_train_test_pairs(self):
        node_pairs_dir = osp.join(self.output_dir, 'node_pairs')
        if not osp.exists(node_pairs_dir):
            os.makedirs(node_pairs_dir)
        with open(osp.join(node_pairs_dir, 'pos_node_pairs.txt'), mode='w', newline='') as pnp_file:
            with open(osp.join(node_pairs_dir, 'neg_node_pairs.txt'), mode='w', newline='') as nnp_file:
                pos_csv_writer = csv.writer(pnp_file, delimiter='\t')
                neg_csv_writer = csv.writer(nnp_file, delimiter='\t')
                for i in range(self.pos_pair_num):
                    attr = np.random.choice(list(self.node_attr_dict.keys()))
                    business_pair = np.random.choice(self.node_attr_dict[attr][np.random.randint(2)], 2)
                    pos_csv_writer.writerow(business_pair)
                    for j in range(self.neg_sampling_ratio):
                        attr = np.random.choice(list(self.node_attr_dict.keys()))
                        business1 = np.random.choice(self.node_attr_dict[attr][0])
                        business2 = np.random.choice(self.node_attr_dict[attr][1])
                        neg_csv_writer.writerow([business1, business2])


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

        self.genre_dict = defaultdict(list)
        with open(osp.join(self.data_dir, 'movie_genre_rel.txt')) as mgr_file:
            for line in mgr_file:
                movie_genre_pair = line.strip().split('\t')
                self.genre_dict[int(movie_genre_pair[1])].append(int(movie_genre_pair[0]))

        for cls in self.genre_dict:
            with open(osp.join(node_classes_dir, 'genre_' + str(cls) + '.txt'), mode='w') as node_class_file:
                for node in self.genre_dict[cls]:
    #                 print(node)
                    node_class_file.write('m' + str(node) + '\n')
    #             for neg_node in self.node_attr_dict[cls][1]:
    # #                 print(node)
    #                 node_class_file.write(neg_node + '\t' + '0' + '\n')

    def gen_train_test_pairs(self):
        node_pairs_dir = osp.join(self.output_dir, 'node_pairs')
        if not osp.exists(node_pairs_dir):
            os.makedirs(node_pairs_dir)

        with open(osp.join(node_pairs_dir, 'pos_node_pairs.txt'), mode='w', newline='') as pnp_file:
            with open(osp.join(node_pairs_dir, 'neg_node_pairs.txt'), mode='w', newline='') as nnp_file:
                pos_csv_writer = csv.writer(pnp_file, delimiter='\t')
                neg_csv_writer = csv.writer(nnp_file, delimiter='\t')
                for i in range(self.pos_pair_num):
                    genre = np.random.choice(list(self.genre_dict.keys()))
                    movie_pair = np.random.choice(self.genre_dict[genre], 2)
                    pos_csv_writer.writerow(['m' + str(movie) for movie in movie_pair])
                    for j in range(self.neg_sampling_ratio):
                        genre = np.random.choice(list(self.genre_dict.keys()))
                        movie1 = np.random.choice(self.genre_dict[genre])
                        movie2 = np.random.choice(list(set(range(1, self.num_movies + 1)) - set(self.genre_dict[genre])))
                        neg_csv_writer.writerow(['m' + str(movie1), 'm' + str(movie2)])


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
        help='sampling ratio of negative pairs compared to positive pairs'
    )
    args = parser.parse_args()
    # print(dir(parser))
    if args.output_dir is None:
        args.output_dir = args.data_dir
    if not osp.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.dataset == 'yelp':
        ds = YelpDataSet(args.data_dir, args.output_dir, args.pos_pair_num, args.neg_sampling_ratio)
    elif args.dataset == 'imdb':
        ds = IMDbDataSet(args.data_dir, args.output_dir, args.pos_pair_num, args.neg_sampling_ratio)
    else:
        ds = None
    # ds.print_node_attr_distr()
    ds.preprocess()
    ds.write_node_classes()
    ds.gen_train_test_pairs()
