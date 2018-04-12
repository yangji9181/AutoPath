import operator
from config import *
from Environment import *
from AutoPath import *


# user choices are raw movie names
def rank(user_choices):
	for choice in user_choices:
		if choice not in movie_to_id or movie_to_id[choice] not in id_to_genre:
			print('Invalid movie')
			return

	start_ids = np.array([environment.name_to_id[movie_to_id[name]] for name in user_choices])
	start_state = np.stack([start_ids, start_ids], axis=1)
	rank_lists = agent.plan(sess, start_state)
	for movie, rank_list in rank_lists.items():
		print(id_to_movie[movie], id_to_genre[movie])
		sorted_rank = sorted(rank_list.items(), key=operator.itemgetter(1), reverse=True)
		for related in sorted_rank[: args.top_k]:
			name = related[0]
			if name != movie and name in id_to_movie and id_to_genre:
				print(id_to_movie[related[0]], id_to_genre[related[0]])
		print('--------')


if __name__ == '__main__':
	id_to_movie, movie_to_id, id_to_genre = utils.load_movie_genre(args.movie_file, args.genre_files, args.genre_name_file)
	environment = Environment(args)
	tf.reset_default_graph()
	os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device_id)
	with tf.device('/gpu:' + str(args.device_id)):
		agent = AutoPath(environment.params, environment)
		saver = tf.train.Saver()
		with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
			saver.restore(sess, args.model_file)
			while True:
				user_choices = raw_input('Please enter movie titles, separated by space\n')
				rank(user_choices.rstrip().split())
