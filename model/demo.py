import operator
from config import *
from Environment import *
from AutoPath import *


# user choices are raw movie names
def rank(user_choices, sid):
	if len(user_choices) == 0:
		raw_input('Invalid input. Example input: 0 or 0 1 2 (separated by space). Please press enter to continue.')
		return
	for choice in user_choices:
		try:
			correct = 0 <= int(choice) and int(choice) < len(sid)
		except ValueError, IndexError:
			raw_input('Invalid input. Example input: 0 or 0 1 2 (separated by space). Please press enter to continue.')
			return
		if not correct:
			raw_input('Invalid input. Example input: 0 or 0 1 2 (separated by space). Please press enter to continue.')
			return

	start_ids = np.array([environment.name_to_id[sid[int(choice)]] for choice in user_choices])
	start_state = np.stack([start_ids, start_ids], axis=1)
	rank_lists = agent.plan(sess, start_state)
	for movie, rank_list in rank_lists.items():
		print('Top %d related movies for <%s>, %s' % (args.top_k, id_to_movie[movie], id_to_genre[movie]))
		sorted_rank = sorted(rank_list.items(), key=operator.itemgetter(1), reverse=True)
		for related in sorted_rank[: args.top_k]:
			name = related[0]
			if name != movie and name in id_to_movie and name in id_to_genre:
				print('<%s>, %s' % (id_to_movie[name], id_to_genre[name]))
		print('----------------')
	raw_input('Above are the results. Please press enter to continue.')


def interactive():
	movies = environment.test_data
	sid = []
	for m in movies:
		sid.append(environment.id_to_name[m])
	while True:
		for movie in movies:
			movie_id = environment.id_to_name[movie]
			print(str(sid.index(movie_id))+': <'+str(id_to_movie[movie_id])+'>, '+str(id_to_genre[movie_id]))
		user_choices = raw_input('Please type in any movie id(s) from the above example list (use space to separate multiple ones): \n')
		rank(user_choices.rstrip().split(), sid)


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
			interactive()

