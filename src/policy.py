import numpy as np
import tensorflow as tf
import sentencepiece as spm
from tqdm import tqdm
import utils
import pickle
import os
import random
from env import JerichoEnv

# For Inference Step (CPU Mode)
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
np.random.seed(1)


class Policy(tf.keras.layers.Layer):
    def __init__(self, args, learning_rate=0.001):
        super(Policy, self).__init__()

        self.args = args
        self.game_name = args.game_name
        self.round = args.round
        self.learning_rate = learning_rate
        self.uct_type = args.uct_type
        
        self.seed = args.seed

        self.sp = spm.SentencePieceProcessor()
        self.sp.Load('spm_models/unigram_8k.model')

        self.data_path = args.data_path.replace('GAME', args.game_name)

        self.max_len_obs = 150
        self.max_len_look = 150
        self.max_len_inv = 50
        self.max_len_action = 12
        self.hidden_dim = 128

        self.sess = tf.keras.backend.get_session()

        self.state_obs_ph = tf.keras.layers.Input(shape=(self.max_len_obs,))
        self.state_look_ph = tf.keras.layers.Input(shape=(self.max_len_look,))
        self.state_inv_ph = tf.keras.layers.Input(shape=(self.max_len_inv,))
        self.prev_action_ph = tf.keras.layers.Input(shape=(self.max_len_action,))
        self.action_ph = tf.keras.layers.Input(shape=(self.max_len_action,))
        self.reward_ph = tf.keras.layers.Input(shape=(10,))
        self.label_ph = tf.keras.layers.Input(shape=(1,))

        self.embed = tf.keras.layers.Embedding(input_dim=8000, output_dim=64, mask_zero=True,
                               embeddings_initializer='uniform')

        self.lstm = tf.keras.layers.LSTM(self.hidden_dim)

        self.dense_obs = tf.keras.layers.Dense(self.hidden_dim, activation='relu')
        self.dense_look = tf.keras.layers.Dense(self.hidden_dim, activation='relu')
        self.dense_inv = tf.keras.layers.Dense(self.hidden_dim, activation='relu')
        self.dense_prev_act = tf.keras.layers.Dense(self.hidden_dim, activation='relu')
        self.dense_act = tf.keras.layers.Dense(self.hidden_dim, activation='relu')

        self.dense1 = tf.keras.layers.Dense(self.hidden_dim, activation='relu')
        self.dense2 = tf.keras.layers.Dense(self.hidden_dim, activation='relu')

        self.dense_out = tf.keras.layers.Dense(1, activation='sigmoid')

        h_obs = self.dense_obs(self.lstm(self.embed(self.state_obs_ph)))
        h_look = self.dense_look(self.lstm(self.embed(self.state_look_ph)))
        h_inv = self.dense_inv(self.lstm(self.embed(self.state_inv_ph)))

        h_prev_act = self.dense_prev_act(self.lstm(self.embed(self.prev_action_ph)))
        h_act = self.dense_act(self.lstm(self.embed(self.action_ph)))

        h = self.dense1(tf.keras.layers.Concatenate()([h_obs, h_look, h_inv, h_prev_act, self.reward_ph]))
        h = self.dense2(tf.keras.layers.Concatenate()([h, h_act]))
        self.outputs = self.dense_out(h)

        loss = tf.keras.losses.binary_crossentropy(self.label_ph, self.outputs)

        self.model = tf.keras.models.Model(inputs=[self.state_obs_ph, self.state_look_ph, self.state_inv_ph, self.prev_action_ph, self.action_ph, self.reward_ph, self.label_ph],
                                           outputs=self.outputs)

        self.model.summary()

        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        self.model.add_loss(loss)
        self.model.compile(optimizer=self.optimizer)

    def train(self, batch_size=256, epochs=3000):
        obs, look, inv, prev_action, action, reward, label = self.load_data()

        max_data_num = look.shape[0]
        data_num = (max_data_num // batch_size) * batch_size

        best_loss = 100
        best_score = -100
        for step in range(epochs):
            batch_obs, batch_look, batch_inv, batch_prev_action, batch_action, batch_reward, batch_label = self.sample_batch(obs, look, inv, prev_action, action,
                                                                                   reward, label, data_num=max_data_num,
                                                                                   batch_size=data_num)

            history = self.model.fit(x=[batch_obs, batch_look, batch_inv, batch_prev_action, batch_action, batch_reward, batch_label], y=[],
                           batch_size=batch_size, epochs=1, shuffle=True, verbose=1)

            loss = np.mean(history.history['loss'][-1])
            score = self.evaluate()

            if score > best_score:
                best_score = score
                print('Best score model updated!: iter=%04d, score=%d' % (step, best_score))
                save_dir = 'weights/%s/round_%d/' % (self.game_name, self.round)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                self.save(save_dir + '%s_weight_policy_best_seed%d.pickle' % (self.uct_type, self.seed))
                # self.save(save_dir + '%s_weight_policy_score_%s.pickle' % (self.uct_type, best_score))

            else:
                print('iter=%04d, score=%d' % (step, score))
            
        f = open('outputs/eval_result_%s_%s.txt'%(self.args.game_name, self.args.uct_type), 'a')
        f.write("- Round %d (learning_seed%d) : max_ep_return=%d\n" % \
                (self.args.round, self.seed, best_score))
        f.close()

    def sample_batch(self, obs, look, inv, prev_action, action, reward, label, data_num, batch_size=128):
        batch_idx = np.random.choice(np.arange(data_num), batch_size, replace=False)
        obs, look, inv, prev_action, action, reward, label = obs[batch_idx], look[batch_idx], inv[batch_idx], prev_action[batch_idx], action[batch_idx], reward[batch_idx], label[batch_idx]

        return obs, look, inv, prev_action, action, reward, label

    def calculate_probs(self, obs, look, inv, prev_action, reward, valid_actions):
        num_valid = len(valid_actions)

        obs = np.tile(obs, (num_valid, 1))
        look = np.tile(look, (num_valid, 1))
        inv = np.tile(inv, (num_valid, 1))
        prev_action = np.tile(prev_action, (num_valid, 1))
        reward = np.tile(reward, (num_valid, 1))
        action = []

        for valid_action in valid_actions:
            valid_action = self.sp.EncodeAsIds(valid_action)
            while len(valid_action) != self.max_len_action:
                valid_action.append(0)
            action.append(valid_action)

        action = np.array(action)

        probs = self.sess.run(self.outputs, feed_dict={
            self.state_obs_ph: obs,
            self.state_look_ph: look,
            self.state_inv_ph: inv,
            self.prev_action_ph: prev_action,
            self.action_ph: action,
            self.reward_ph: reward
        })

        probs = utils.softmax(probs, T=1.0)
        return probs

    def load_data(self):
        path = 'data/%s/%s_trial_%d/round_%d/merged_mcts_log_%02d.txt' % (self.game_name, self.args.uct_type, self.seed, self.round, self.args.max_depth)
        f = open(path, 'r')
        a = f.readlines()

        all_ids_obs, all_ids_look, all_ids_inv, all_ids_prev_act, all_ids_act, all_rewards, all_labels = [[] for _ in range(7)]

        max_obs, max_look, max_inv, max_act = 0, 0, 0, 0

        for line in tqdm(a):
            str_obs = line.split('[OBS]')[-1].split('[LOOK]')[0].strip()
            str_look = line.split('[OBS]')[-1].split('[LOOK]')[1].split('[INV]')[0].strip()
            str_inv = line.split('[OBS]')[-1].split('[LOOK]')[1].split('[INV]')[1].split('[VALID_ACTION]')[0].strip()
            str_prev_act = line.split('[PREV_ACTION]')[-1].split('[REWARD]')[0].strip()
            str_act = line.split('[ACTION]')[-1].split('[PREV_ACTION]')[0].strip()
            str_valid_acts = line.split(' [ACTION]')[0].split(' [VALID_ACTION] ')[1:]

            for str_valid_act in str_valid_acts:
                ids_obs = self.sp.EncodeAsIds(str_obs)
                ids_look = self.sp.EncodeAsIds(str_look)
                ids_inv = self.sp.EncodeAsIds(str_inv)
                ids_prev_act = self.sp.EncodeAsIds(str_prev_act)
                ids_act = self.sp.EncodeAsIds(str_act)
                ids_valid_act = self.sp.EncodeAsIds(str_valid_act)

                str_reward = line.split('[REWARD] ')[-1].strip()
                reward = np.zeros((10,))
                for i in range(len(str_reward)):
                    reward[i] = int(str_reward[i])

                if max_obs < len(ids_obs):
                    max_obs = len(ids_obs)
                if max_look < len(ids_look):
                    max_look = len(ids_look)
                if max_inv < len(ids_inv):
                    max_inv = len(ids_inv)
                if max_act < len(ids_act):
                    max_act = len(ids_act)

                all_ids_obs.append(ids_obs)
                all_ids_look.append(ids_look)
                all_ids_inv.append(ids_inv)
                all_ids_prev_act.append(ids_prev_act)
                all_ids_act.append(ids_act if str_valid_act == str_act else ids_valid_act)
                all_rewards.append(reward)
                all_labels.append(1 if str_valid_act == str_act else 0)

        all_ids_obs = self.pad_sequences(all_ids_obs, self.max_len_obs)
        all_ids_look = self.pad_sequences(all_ids_look, self.max_len_look)
        all_ids_inv = self.pad_sequences(all_ids_inv, self.max_len_inv)
        all_ids_prev_act = self.pad_sequences(all_ids_prev_act, self.max_len_action)
        all_ids_act = self.pad_sequences(all_ids_act, self.max_len_action)

        all_rewards = np.array(all_rewards)
        all_labels = np.array(all_labels)

        return all_ids_obs, all_ids_look, all_ids_inv, all_ids_prev_act, all_ids_act, all_rewards, all_labels

    def pad_sequences(self, ids_list, max_len):
        results = []
        for ids in ids_list:
            if len(ids) > max_len:
                results.append(ids[:max_len])
            else:
                while len(ids) < max_len:
                    ids.append(0)
                results.append(ids)

        return np.array(results)

    def get_parameters(self):
        parameters = []
        weights = self.get_weights()
        assert len(self.trainable_variables) == len(weights)
        for idx, variable in enumerate(self.trainable_variables):
            weight = weights[idx]
            parameters.append((variable.name, weight))
        return parameters

    def save(self, filepath):
        parameters = self.get_parameters()
        with open(filepath, 'wb') as f:
            pickle.dump(parameters, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_weights(self, filepath, exact_match=False):
        with open(filepath, 'rb') as f:
            parameters = pickle.load(f)
        assert len(parameters) == len(self.weights)
        weights = []
        for variable, parameter in zip(self.weights, parameters):
            name, value = parameter
            if exact_match:
                if name != variable.name:
                    print(name, variable.name)
                assert name == variable.name
            weights.append(value)
        self.set_weights(weights)

    def evaluate(self, max_episode_len=50, save_result=False, learning_seed=0):
        scores = []
        rom_path = self.args.rom_path #+ utils.game_file(self.args.game_name)

        maxlen_obs = 150
        maxlen_look = 150
        maxlen_inv = 50
        max_len_action = 12

        for _ in range(1):            
            seed = random.randint(0,1000)
            env = JerichoEnv(rom_path, seed, self.args.env_step_limit)
            env.create()

            obs, info = env.reset()
            cum_reward = 0
            step = 0
            prev_action = '<s>'

            for _ in range(50):
                obs, look, inv, prev_action, score = utils.state_representation(obs, info['look'], info['inv'],
                                                                                prev_action, info['score'],
                                                                                maxlen_obs, maxlen_look, maxlen_inv,
                                                                                max_len_action)
                probs = self.calculate_probs(obs, look, inv, prev_action, score, info['valid'])
                idx = np.argmax(probs)
                action = info['valid'][idx]

                obs, reward, done, info = env.step(action)
                cum_reward += reward
                step += 1

                prev_action = action

                score = info['score']

                next_obs_text = obs + info['look'] + info['inv']

                if '*** You have won ***' in next_obs_text:
                    done = True
                    score = int(next_obs_text.split('you scored ')[1].split(' out of')[0])

            scores.append(score)

        return np.mean(scores)
