import tensorflow as tf
import json
import os
from bandits.abstract_bandit import ContextualBandit


class TFLinUCB(ContextualBandit):
    def __init__(self, context_dimension=None, items=None, alpha=0.1, restore=False, path_config=None,
             path_dir='checkpoints/', path_ckpt=''):
        super().__init__(context_dimension)
        if restore and path_config:
            self.restore_config(path_config)
            self.inverse_mapping = {v: k for k, v in self.mapping.items()}
        elif context_dimension is not None and items is not None:
            self.mapping = {integer: item for integer, item in enumerate(items)}
            self.context_dimension = context_dimension
            self.alpha = alpha
        else:
            raise Exception('You should provide context_dimension and items or restore config and tensorflow graph!')
        self.inverse_mapping = {v: k for k, v in self.mapping.items()}
        number_items = len(self.mapping)
        self.A, self.A_inv, self.b, self.theta = self.initialize_model(self.context_dimension, number_items)
        self.checkpoint = tf.train.Checkpoint(A=self.A, A_inv=self.A_inv, b=self.b, theta=self.theta)
        if restore:
            self.restore_tf(path_dir, path_ckpt)

    def __str__(self):
        print('Contextual bandit for personalized recommendations in tensorflow.\n')

    def __repr__(self):
        print("TFLinUCB(context_dimension={0},alpha={1})".format(self.context_dimension,
                                                                 self.alpha))

    def save(self, path, path_config):
        self.checkpoint.save(path)
        config = {'context_dimension': self.context_dimension,
                  'alpha': self.alpha,
                  'mapping': self.mapping}
        with open(path_config, 'w') as f:
            json.dump(config, f)

    def set_alpha(self, alpha):
        self.alpha = alpha

    def restore_tf(self, path_dir='checkpoints/', path_ckpt=''):
        if not path_ckpt:
            self.checkpoint.restore(tf.train.latest_checkpoint(path_dir))
        else:
            self.checkpoint.restore(os.path.join(path_dir, path_ckpt))

    def restore_all(self, path_config, path_dir='checkpoints/', path_ckpt=''):
        self.restore_config(path_config)
        self.restore_tf(path_dir, path_ckpt)

    def restore_config(self, path_config):
        with open(path_config, 'r') as f:
            config = json.load(f)
        self.context_dimension = config['context_dimension']
        self.alpha = config['alpha']
        mapping = config['mapping']
        self.mapping = {int(k): v for k, v in mapping.items()}

    def initialize_model(self, context_dimension, number_items):
        A = tf.Variable(tf.eye(num_columns=context_dimension, num_rows=context_dimension, batch_shape=[number_items]),
                        name='A')
        A_inv = tf.Variable(tf.identity(A), name='A_inv')
        b = tf.Variable(tf.zeros(shape=(number_items, context_dimension, 1)), name='b')
        theta = tf.Variable(tf.zeros(shape=(number_items, context_dimension, 1)), name='theta')
        return A, A_inv, b, theta

    def predict(self, context, N=1,
                return_stats=False):
        context = tf.expand_dims(context, axis=-1)
        estimated_reward = tf.matmul(tf.transpose(self.theta, perm=[0, 2, 1]), context, name='estimated_reward')
        uncertainty = self.alpha * tf.sqrt(
            tf.matmul(tf.matmul(tf.transpose(context, perm=[0, 2, 1]), self.A_inv), context),
            name='uncertainty')
        scores = tf.add(uncertainty, estimated_reward, name='scores')
        action = tf.argsort(tf.reshape(scores, shape=(-1,)), direction='DESCENDING')[:N].numpy()
        score = tf.gather(scores, action).numpy()
        item = [self.mapping.get(i) for i in action]
        if N == 1:
            action = action[0]
            item = item[0]
            score = score[0]
        if return_stats:
            return action, item, score, estimated_reward.numpy(), uncertainty.numpy(), scores.numpy()
        else:
            return action, item, score

    def predict_on_batch(self, batch_context, N=1, return_stats=False):
        if return_stats:
            action, item, score, estimated_reward, uncertainty = tf.map_fn(lambda x: self.predict(x, N, return_stats),
                                                                           batch_context,
                                                                           dtype=(tf.float32, tf.string, tf.float32,
                                                                                  tf.float32, tf.float32))
            item = [i.decode() for i in item.numpy()]
            return action.numpy(), item, score.numpy(), estimated_reward.numpy(), uncertainty.numpy()

        else:
            action, item, score = tf.map_fn(lambda x: self.predict(x, N, return_stats), batch_context,
                                            dtype=(tf.float32, tf.string, tf.float32))
            item = [i.decode() for i in item.numpy()]
            return action.numpy(), item, score.numpy()

    def update(self, action, context, reward):
        context = tf.reshape(context, shape=(-1, 1))
        prod = tf.matmul(context, tf.transpose(context))
        self.A = tf.compat.v1.scatter_add(self.A, [action], prod)
        self.b = tf.compat.v1.scatter_add(self.b, [action], reward * context)
        self.A_inv = tf.compat.v1.scatter_update(self.A_inv, [action], tf.linalg.inv(self.A[action]))
        self.theta = tf.compat.v1.scatter_update(self.theta, [action], tf.matmul(self.A_inv[action], self.b[action]))

    def add_new_items(self, items: list):
        diff = set(items).difference(self.mapping.values())
        items = [i for i in items if i in diff]
        number_items = len(items)
        maximum_num = max(self.mapping.keys())
        self.mapping.update({maximum_num + 1 + c: item for c, item in enumerate(items)})
        self.inverse_mapping = {v: k for k, v in self.mapping.items()}
        A, A_inv, b, theta = self.initialize_model(self.context_dimension, number_items)
        self.A = tf.Variable(tf.concat([self.A, A], axis=0), name='A')
        self.A_inv = tf.Variable(tf.concat([self.A_inv, A_inv], axis=0), name='A_inv')
        self.b = tf.Variable(tf.concat([self.b, b], axis=0), name='b')
        self.theta = tf.Variable(tf.concat([self.theta, theta], axis=0), name='theta')

    def del_from_tensor(self, matrix, to_del_idx):
        shape = matrix.shape
        if to_del_idx == 0:
            matrix = tf.slice(matrix, [1, *[0 for i in range(len(shape[1:]))]], [shape[0] - 1, *shape[1:]])
        elif to_del_idx == shape[0] - 1:
            matrix = tf.slice(matrix, [0, *[0 for i in range(len(shape[1:]))]], [shape[0] - 1, *shape[1:]])
        else:
            slice1 = tf.slice(matrix, [0, *[0 for i in range(len(shape[1:]))]], [to_del_idx, *shape[1:]])
            slice2 = tf.slice(matrix, [to_del_idx + 1, *[0 for i in range(len(shape[1:]))]],
                              [shape[0] - to_del_idx - 1, *shape[1:]])
            matrix = tf.Variable(tf.concat([slice1, slice2], axis=0))
        return matrix

    def remove_items(self, items: list):
        diff = set(self.mapping.values()).difference(items)
        items = [i for i in items if not i in diff]
        for item in items:
            to_del_idx = self.inverse_mapping.get(item)
            self.mapping.pop(to_del_idx)
            self.A = self.del_from_tensor(self.A, to_del_idx)
            self.A_inv = self.del_from_tensor(self.A_inv, to_del_idx)
            self.b = self.del_from_tensor(self.b, to_del_idx)
            self.theta = self.del_from_tensor(self.theta, to_del_idx)
        self.mapping = {integer: item for integer, item in enumerate(self.mapping.values())}
        self.inverse_mapping = {v: k for k, v in self.mapping.items()}
