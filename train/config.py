import argparse
import os


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--device_id', type=int, default=1, help=None)
	parser.add_argument('--max_neighbor', type=int, default=20, help=None)
	parser.add_argument('--num_node', type=int, default=-1, help='Total number of nodes')
	parser.add_argument('--num_attribute', type=int, default=-1, help='Total number of attributes')
	parser.add_argument('--num_type', type=int, default=-1, help='Total number of node types')
	parser.add_argument('--embed_dim', type=int, default=64, help=None)
	parser.add_argument('--feature_dim', type=int, default=-1, help=None)
	parser.add_argument('--hidden_dim', type=list, default=[64], help=None)
	parser.add_argument('--reconstruct_hidden_dim', type=list, default=[64], help=None)
	parser.add_argument('--learning_rate', type=float, default=1e-3, help=None)
	parser.add_argument('--keep_prob', type=float, default=0.5, help='Used for dropout')
	parser.add_argument('--clip_epsilon', type=float, default=1e-1, help=None)
	parser.add_argument('--c_value', type=float, default=0.2, help='Coefficient for value function loss')
	parser.add_argument('--batch_size', type=int, default=20, help='Number of trajectories sampled')
	parser.add_argument('--trajectory_length', type=int, default=2, help=None)
	parser.add_argument('--epoch', type=int, default=2, help=None)
	parser.add_argument('--classification_step', type=int, default=1, help='Number of rounds of mini batch SGD per epoch for classification')
	parser.add_argument('--PPO_step', type=int, default=1, help='Number of rounds of mini batch SGD per epoch for PPO')
	parser.add_argument('--step', type=int, default=2, help=None)
	parser.add_argument('--c_classification', type=float, default=1.0, help='Classification coefficient')
	parser.add_argument('--c_reconstruction', type=float, default=0.1, help='Reconstruction coefficient')
	parser.add_argument('--num_trial', type=int, default=2, help='Number of random walks during planning')
	parser.add_argument('--top_k', type=int, default=10, help='Top k for precision and recall')
	parser.add_argument('--num_process', type=int, default=4, help='Number of processes for ranking')
	return parser.parse_args()


def init_dir(args):
	args.data_dir = '../data/imdb/'
	args.model_dir = os.getcwd() + '/model/'
	args.log_dir = os.getcwd() + '/log/'
	args.node_file = args.data_dir + 'node.dat'
	args.link_file = args.data_dir + 'link.dat'
	args.train_files = [args.data_dir + 'node_classes/genre_' + str(i) + '.txt' for i in range(1, 24)]
	args.test_file = args.data_dir + 'test_nodes.txt'
	args.embedding_file = args.data_dir + 'pretrained_emb.txt'
	args.plot_file = args.model_dir + 'reward.png'
	args.model_file = args.model_dir + 'model.ckpt'
	args.rank_list_file = '../evaluation/imdb/' + 'rank_list.pkl'

args = parse_args()
init_dir(args)
