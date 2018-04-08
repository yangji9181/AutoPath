from config import *
from Environment import *
from AutoPath import *

if __name__ == '__main__':
	environment = Environment(args)
	tf.reset_default_graph()
	os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device_id)
	with tf.device('/gpu:' + str(args.device_id)):
		agent = AutoPath(environment.params, environment)
		saver = tf.train.Saver()
		with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
			if os.path.exists(args.model_file + '.meta'):
				saver.restore(sess, args.model_file)
			else:
				sess.run(tf.global_variables_initializer())
			agent.train(sess)
			saver.save(sess, args.model_file)
			print('Node type accuracy: %f' % agent.accuracy(sess))
			rank_lists = agent.plan(sess)
