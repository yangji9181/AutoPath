import gc
import pickle
from copy import deepcopy
from random import shuffle
from multiprocessing import *

import numpy as np
from tqdm import tqdm

import utils
from NN import *


class AutoPath(object):
	def __init__(self, params, environment):
		self.params = params
		self.environment = environment
		self.build()

	def build(self):
		self.training = tf.placeholder(tf.bool)
		# last index is None embedding
		self.embedding = embedding('Embedding', [self.params.num_node, self.params.embed_dim])
		self.indices = tf.placeholder(tf.int32, [None])
		self.labels = tf.placeholder(tf.int32, [None])
		self.neighbors = tf.placeholder(tf.int32, [None, None])
		self.state = tf.placeholder(tf.int32, [None, 2])
		self.action = tf.placeholder(tf.int32, [None])
		self.next_state = tf.placeholder(tf.int32, [None, 2])
		self.reward = tf.placeholder(tf.float32, [None])
		self.future_reward = tf.placeholder(tf.float32, [None])

		self.build_classification()
		self.build_PPO()

		for variable in tf.trainable_variables():
			print(variable.name, variable.get_shape())
			# tf.summary.histogram(variable.name, variable)
		self.merged_summary_op = tf.summary.merge_all()

	def build_classification(self):
		embedding = dropout(tf.nn.embedding_lookup(self.embedding, self.indices), self.params.keep_prob, self.training)
		logits = fully_connected(embedding, self.params.num_type, 'Classification', activation='linear')
		self.prediction = tf.argmax(logits, axis=1)
		loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
			labels=tf.one_hot(self.labels, self.params.num_type), logits=logits), axis=0)
		optimizer = tf.train.AdamOptimizer(self.params.learning_rate)
		self.classification_step = optimizer.minimize(loss)

		del embedding, logits, loss, optimizer
		gc.collect()

	def build_PPO(self):
		state_embedding = tf.reshape(tf.nn.embedding_lookup(self.embedding, self.state), [-1, 2 * self.params.embed_dim])
		next_embedding = tf.reshape(tf.nn.embedding_lookup(self.embedding, self.next_state), [-1, 2 * self.params.embed_dim])
		with tf.variable_scope('value_policy', reuse=tf.AUTO_REUSE):
			hidden = self.value_policy(state_embedding)
			hidden_next = self.value_policy(next_embedding)
			policy = self.policy(hidden)
			value = tf.squeeze(self.value(hidden))
			value_next = tf.squeeze(self.value(hidden_next))

		# do not use scaled std of embedding vectors as policy std to avoid overflow
		sigma = tf.ones(self.params.embed_dim, dtype=tf.float32) / self.params.embed_dim
		self.build_train(tf.nn.embedding_lookup(self.embedding, self.action), value, value_next, policy, sigma)
		self.build_plan(policy, sigma)

		del state_embedding, hidden, value, policy, sigma
		gc.collect()

	def build_train(self, action, value, value_next, policy_mean, sigma):
		advantage = self.reward + tf.stop_gradient(value_next) - tf.stop_gradient(value)
		# Gaussian policy with identity matrix as covariance mastrix
		log_pi = -0.5 * tf.reduce_sum(tf.square((action - policy_mean) / sigma), axis=1)
		actor_loss = -tf.reduce_mean(log_pi * advantage, axis=0)
		critic_loss = tf.reduce_mean(tf.squared_difference(self.future_reward, value), axis=0)
		loss = actor_loss + self.params.c_value * critic_loss
		optimizer = tf.train.AdamOptimizer(self.params.learning_rate)
		self.global_step = tf.Variable(0, trainable=False)
		self.PPO_step = optimizer.minimize(loss, global_step=self.global_step)

		with tf.control_dependencies([tf.Assert(tf.is_finite(loss), [loss])]):
			tf.summary.scalar('value', tf.reduce_mean(value, axis=0))
			tf.summary.scalar('actor_loss', actor_loss)
			tf.summary.scalar('critic_loss', critic_loss)

		del advantage, log_pi, actor_loss, critic_loss, loss, optimizer
		gc.collect()

	def build_plan(self, policy_mean, sigma):
		policy = tf.distributions.Normal(policy_mean, sigma)
		action_embed = policy.sample()
		l2_diff = tf.squared_difference(tf.expand_dims(action_embed, axis=1),
		                                tf.nn.embedding_lookup(self.embedding, self.neighbors))
		self.decision = tf.argmin(tf.reduce_sum(l2_diff, axis=-1), axis=1)

		del policy, action_embed, l2_diff
		gc.collect()

	def value_policy(self, state):
		hidden = state
		for i, dim in enumerate(self.params.hidden_dim):
			hidden = fully_connected(hidden, dim, 'policy_value_' + str(i))
		return hidden

	def value(self, hidden):
		return fully_connected(hidden, 1, 'value_o', activation='linear')

	def policy(self, hidden):
		return fully_connected(hidden, self.params.embed_dim, 'policy_o', activation='tanh')

	# the number of trajectories sampled is equal to batch size
	def collect_trajectory(self, sess, start_state):
		feed_state = deepcopy(start_state)
		states, actions, nexts = [], [], []
		for i in range(self.params.trajectory_length):
			states.append(deepcopy(feed_state))
			feed_neighbor = self.environment.get_neighbors(feed_state[:, 1])
			# action contains indices of actual node IDs
			action_indices = sess.run(self.decision,
			                          feed_dict={self.state: feed_state, self.neighbors: feed_neighbor})
			action = feed_neighbor[np.array(range(len(start_state))), action_indices]
			actions.append(action)
			feed_state[:, 1] = action
			nexts.append(deepcopy(feed_state))
		states = np.transpose(np.array(states), axes=(1, 0, 2)).tolist()
		actions = np.transpose(np.array(actions)).tolist()
		nexts = np.transpose(np.array(nexts), axes=(1, 0, 2)).tolist()
		return states, actions, nexts

	def PPO_epoch(self, sess):
		start_state = self.environment.initial_state()
		states, actions, nexts = self.collect_trajectory(sess, start_state)
		states, actions, rewards, nexts, future_rewards = self.environment.compute_reward(states, actions, nexts)
		self.average_reward.append(np.average(future_rewards))
		total_size = self.params.trajectory_length * len(start_state)
		assert len(states) == total_size and len(actions) == total_size and \
		       len(rewards) == total_size and len(future_rewards) == total_size
		indices = range(total_size)
		shuffle(indices)
		sample_size = total_size / self.params.step
		for _ in tqdm(range(self.params.PPO_step), ncols=100):
			for i in range(self.params.step):
				batch_indices = indices[i * sample_size: (i + 1) * sample_size]
				_, summary = sess.run([self.PPO_step, self.merged_summary_op],
				                      feed_dict={self.state: states[batch_indices],
				                                 self.action: actions[batch_indices],
				                                 self.reward: rewards[batch_indices],
				                                 self.next_state: nexts[batch_indices],
				                                 self.future_reward: future_rewards[batch_indices]})
				self.summary_writer.add_summary(summary, global_step=sess.run(self.global_step))

	def sample_classification(self):
		indices, labels = [], []
		types = self.environment.type_to_id.keys()
		for t in types:
			labels += [self.environment.type_to_id[t]] * self.params.batch_size
			indices += list(np.random.choice(self.environment.type_to_node[t], self.params.batch_size))
		return np.array(indices), np.array(labels)

	def classification_epoch(self, sess):
		for _ in tqdm(range(self.params.classification_step), ncols=100):
			indices, labels = self.sample_classification()
			sess.run(self.classification_step, feed_dict={self.indices: indices, self.labels: labels, self.training: True})

	# do not initialize variables here
	def train(self, sess):
		self.summary_writer = tf.summary.FileWriter(self.params.log_dir, graph=tf.get_default_graph())
		self.average_reward = []
		for _ in tqdm(range(self.params.epoch), ncols=100):
			self.classification_epoch(sess)
			self.PPO_epoch(sess)
		utils.plot(self.average_reward, self.params.plot_file)


	def accuracy(self, sess):
		indices, labels = [], []
		types = self.environment.type_to_id.keys()
		for t in types:
			labels += [self.environment.type_to_id[t]] * len(self.environment.type_to_node[t])
			indices += list(self.environment.type_to_node[t])
		predictions = sess.run(self.prediction, feed_dict={self.indices: np.array(indices), self.training: False})
		correct = 0
		for label, prediction in zip(labels, predictions):
			if label == prediction:
				correct += 1
		return float(correct) / len(indices)

	def plan(self, sess):
		start_state = self.environment.initial_test()
		trials = []
		for _ in tqdm(range(self.params.num_trial), ncols=100):
			_, actions, _ = self.collect_trajectory(sess, start_state)
			trials.append(np.array(actions))
		trials = np.concatenate(trials, axis=1)
		start_state = start_state[:, 0]
		assert len(start_state) == len(trials)

		manager = Manager()
		rank_lists = manager.dict()

		processes = []
		for i in range(self.params.num_process):
			process = Process(target=self.worker, args=(i, self.params.num_process, len(start_state), start_state, trials, rank_lists,))
			process.start()
			processes.append(process)
		for process in processes:
			process.join()
		result_rank_lists = {key: value for key, value in rank_lists.items()}
		pickle.dump(result_rank_lists, open(self.params.rank_list_file, 'wb'))
		return result_rank_lists

	def worker(self, process_id, num_process, num_data, start_state, trials, rank_lists):
		for data_id, state, action in zip(range(num_data), start_state, trials):
			if data_id % num_process == process_id:
				state_type = self.environment.node_to_type[state]
				frequency = np.zeros(self.params.num_node, dtype=np.int32)
				for a in action:
					frequency[a] -= 1
				rank_lists[self.environment.id_to_name[state]] = {self.environment.id_to_name[node_id]: count for node_id, count in enumerate(frequency) if self.environment.node_to_type[node_id] == state_type}



