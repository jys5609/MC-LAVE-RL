import numpy as np
import tensorflow as tf
import sentencepiece as spm
from env import JerichoEnv
from tqdm import tqdm
import os
import time
import pickle
import random

# For Inference Step (CPU Mode)
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def pad_sequences(ids_list, max_len):
    results = []
    for ids in ids_list:
        if len(ids) > max_len:
            results.append(ids[:max_len])
        else:
            while len(ids) < max_len:
                ids.append(0)
            results.append(ids)

    return np.array(results)

class ReplayBuffer(object):
    def __init__(self, path, sp=None, prioritized=False, alpha=1.0):
        f = open(path, 'r')
        a = f.readlines()

        self.max_len_obs = 150
        self.max_len_look = 150
        self.max_len_inv = 50
        self.max_len_action = 12
        self.max_valid_len = 0
        self.max_bin_score = 10

        self.prioritized = prioritized
        self.alpha = alpha
        self.sp = sp

        all_ids_obs, all_ids_look, all_ids_inv, all_ids_score, all_ids_prev_act, all_ids_act, all_ids_valid_acts, \
            all_rewards, all_dones, all_ids_next_obs, all_ids_next_look, all_ids_next_inv, all_ids_next_score = [[] for _ in range(13)]
    
        max_obs, max_look, max_inv, max_act = 0, 0, 0, 0

        for line in tqdm(a):
            if '[OBS]' in line:
                str_obs = line.split('[OBS]')[-1].split('[LOOK]')[0].strip()
                str_look = line.split('[LOOK]')[1].split('[INV]')[0].strip()
                try:
                    str_inv = line.split('[INV]')[1].split('[SCORE]')[0].strip()
                except:
                    continue
                str_score = line.split('[SCORE]')[-1].split('[PREV_ACTION]')[0].strip()
                # ids_score = self.sp.EncodeAsIds(str_score)                
                ids_score = np.zeros((10,))
                for i in range(len(str_score)):
                    ids_score[i] = int(str_score[i])

                str_prev_act = line.split('[PREV_ACTION]')[-1].split('[ACTION]')[0].strip()
                str_act = line.split('[ACTION]')[-1].split('[VALID_ACTION]')[0].strip()
                str_valid_acts = line.split('[ACTION]')[-1].split(' [REWARD]')[0].split(' [VALID_ACTION] ')[1:]
                str_next_obs = line.split('[NEXT_OBS]')[-1].split('[NEXT_LOOK]')[0].strip()
                str_next_look = line.split('[NEXT_LOOK]')[-1].split('[NEXT_INV]')[0].strip()
                str_next_inv = line.split('[NEXT_INV]')[-1].split('[NEXT_SCORE]')[0].strip()                
                str_next_score = line.split('[NEXT_SCORE]')[-1].strip()                
                # ids_next_score = self.sp.EncodeAsIds(str_next_score)
                ids_next_score = np.zeros((10,))
                for i in range(len(str_next_score)):
                    ids_next_score[i] = int(str_next_score[i])

                ids_obs = self.sp.EncodeAsIds(str_obs)
                ids_look = self.sp.EncodeAsIds(str_look)
                ids_inv = self.sp.EncodeAsIds(str_inv)                

                ids_prev_act = self.sp.EncodeAsIds(str_prev_act)
                ids_act = self.sp.EncodeAsIds(str_act)
                ids_next_obs = self.sp.EncodeAsIds(str_next_obs)
                ids_next_look = self.sp.EncodeAsIds(str_next_look)
                ids_next_inv = self.sp.EncodeAsIds(str_next_inv)

                ids_valid_acts = []
                for str_valid_act in str_valid_acts:
                    ids_valid_acts.append(self.sp.EncodeAsIds(str_valid_act)) 
                ids_valid_acts = pad_sequences(ids_valid_acts, self.max_len_action)

                str_reward = line.split('[REWARD]')[-1].split('[DONE]')[0].strip()
                reward = int(str_reward)
                str_done = line.split('[DONE]')[-1].split('[NEXT_OBS]')[0].strip()
                done = int(str_done)
                
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
                all_ids_act.append(ids_act)
                all_ids_valid_acts.append(ids_valid_acts)
                all_rewards.append(reward)
                all_ids_score.append(ids_score)
                all_dones.append(done)

                all_ids_next_obs.append(ids_next_obs)
                all_ids_next_look.append(ids_next_look)
                all_ids_next_inv.append(ids_next_inv)
                all_ids_next_score.append(ids_score)

        self.all_ids_obs = pad_sequences(all_ids_obs, self.max_len_obs)
        self.all_ids_look = pad_sequences(all_ids_look, self.max_len_look)
        self.all_ids_inv = pad_sequences(all_ids_inv, self.max_len_inv)
        self.all_ids_prev_act = pad_sequences(all_ids_prev_act, self.max_len_action)
        self.all_ids_act = pad_sequences(all_ids_act, self.max_len_action)
        self.all_ids_next_obs = pad_sequences(all_ids_next_obs, self.max_len_obs)
        self.all_ids_next_look = pad_sequences(all_ids_next_look, self.max_len_look)
        self.all_ids_next_inv = pad_sequences(all_ids_next_inv, self.max_len_inv)
        self.all_ids_score = pad_sequences(all_ids_score, self.max_bin_score)
        self.all_ids_next_score = pad_sequences(all_ids_next_score, self.max_bin_score)
        
        self.all_rewards = np.array(all_rewards)
        self.all_dones = np.array(all_dones)

        self.max_valid_len = np.max([len(va) for va in all_ids_valid_acts])

        def pad_valid_acts(valid_actions):
            padded_valid_actions = []
            for va in valid_actions:
                padding = np.zeros([self.max_valid_len - len(va), self.max_len_action])
                padded_va = np.concatenate([va, padding])
                padded_valid_actions.append(padded_va)
            
            return np.array(padded_valid_actions)

        self.all_ids_valid_acts = pad_valid_acts(all_ids_valid_acts)
        self.size = len(self.all_ids_obs)

    def sample(self, batch_size):
        if not self.prioritized:
            batch_idx = np.random.choice(np.arange(self.size), batch_size, replace=False)
        else:
            raise NotImplementedError

        obs = self.all_ids_obs[batch_idx]
        look = self.all_ids_look[batch_idx]
        inv = self.all_ids_inv[batch_idx]
        prev_action = self.all_ids_prev_act[batch_idx]
        bin_score = self.all_ids_score[batch_idx]

        action = self.all_ids_act[batch_idx]
        valid_actions = self.all_ids_valid_acts[batch_idx]
        reward = self.all_rewards[batch_idx]
        
        done = self.all_dones[batch_idx]
        next_obs = self.all_ids_next_obs[batch_idx]
        next_look = self.all_ids_next_look[batch_idx]
        next_inv = self.all_ids_next_inv[batch_idx]
        next_bin_score = self.all_ids_next_score[batch_idx]

        return obs, look, inv, bin_score, prev_action, action, valid_actions, reward, done, next_obs, next_look, next_inv, next_bin_score


class QNetwork(tf.keras.layers.Layer):
    def __init__(self, hidden_dim=128):
        super(QNetwork, self).__init__()
        self.embed = tf.keras.layers.Embedding(input_dim=8000, output_dim=hidden_dim, mask_zero=True,
                            embeddings_initializer='uniform')

        self.lstm = tf.keras.layers.LSTM(hidden_dim)

        self.dense_obs = tf.keras.layers.Dense(hidden_dim, activation='relu')
        self.dense_look = tf.keras.layers.Dense(hidden_dim, activation='relu')
        self.dense_inv = tf.keras.layers.Dense(hidden_dim, activation='relu')
        self.dense_prev_act = tf.keras.layers.Dense(hidden_dim, activation='relu')
        self.dense_act = tf.keras.layers.Dense(hidden_dim, activation='relu')

        self.dense1 = tf.keras.layers.Dense(hidden_dim, activation='relu')
        self.dense2 = tf.keras.layers.Dense(hidden_dim, activation='relu')
        self.dense_out = tf.keras.layers.Dense(1, activation=None)

    def call(self, inputs, **kwargs):
        obs, look, inv, prev_action, score, action = inputs

        h_obs = self.dense_obs(self.lstm(self.embed(obs)))
        h_look = self.dense_look(self.lstm(self.embed(look)))
        h_inv = self.dense_inv(self.lstm(self.embed(inv)))

        h_prev_act = self.dense_prev_act(self.lstm(self.embed(prev_action)))
        h_act = self.dense_act(self.lstm(self.embed(action)))

        h = self.dense1(tf.keras.layers.Concatenate()([h_obs, h_look, h_inv, h_prev_act, score]))
        h = self.dense2(tf.keras.layers.Concatenate()([h, h_act]))
        q = self.dense_out(h)

        return q


class DRRN(tf.keras.layers.Layer):
    def __init__(self, args, learning_rate=0.005):
        super(DRRN, self).__init__()

        self.args = args
        self.round = args.round
        self.rom_path = args.rom_path
        self.game_name = args.game_name
        
        self.seed = args.seed
        self.env_step_limit = args.env_step_limit
        self.learning_rate = learning_rate
        self.step = 0

        self.sp = spm.SentencePieceProcessor()
        self.sp.Load('spm_models/unigram_8k.model')

        log_dir = 'data/%s/%s_trial_%d/round_%d/' % (args.game_name, self.args.uct_type, args.seed, args.round)
        path = log_dir + 'merged_replay.txt'                    

        self.eval_mode = args.evaluate
        if not self.eval_mode:
            self.replay_buffer = ReplayBuffer(path, sp=self.sp, prioritized=False, alpha=1.0)
            self.max_valid_len = self.replay_buffer.max_valid_len
        else:
            self.replay_buffer = None
            self.max_valid_len = 30

        self.max_len_obs = 150
        self.max_len_look = 150
        self.max_len_inv = 50
        self.max_len_action = 12
        self.max_bin_score = 10
        
        self.gamma = args.discount_factor
        self.tau = 0.01

        self.best_episode_return = -10000.

        self.sess = tf.keras.backend.get_session()

        self.state_obs_ph = tf.keras.layers.Input(shape=(self.max_len_obs,))
        self.state_look_ph = tf.keras.layers.Input(shape=(self.max_len_look,))
        self.state_inv_ph = tf.keras.layers.Input(shape=(self.max_len_inv,))
        self.state_score_ph = tf.keras.layers.Input(shape=(self.max_bin_score,))        
        self.prev_action_ph = tf.keras.layers.Input(shape=(self.max_len_action,))

        self.action_ph = tf.keras.layers.Input(shape=(self.max_len_action,))
        self.reward_ph = tf.keras.layers.Input(shape=(1,))        
        self.done_ph = tf.keras.layers.Input(shape=(1,))

        self.next_obs_ph = tf.keras.layers.Input(shape=(self.max_len_obs,))
        self.next_look_ph = tf.keras.layers.Input(shape=(self.max_len_look,))
        self.next_inv_ph = tf.keras.layers.Input(shape=(self.max_len_inv,))
        self.next_score_ph = tf.keras.layers.Input(shape=(self.max_bin_score,))

        self.q_target_best_action_ph = tf.keras.layers.Input(shape=(self.max_len_action,))
        self.q_target_ph = tf.keras.layers.Input(shape=(1,))

        self.q = QNetwork(hidden_dim=64)
        self.q_target = QNetwork(hidden_dim=64)

        optimizer_variables = []

        self.q_ = self.q([self.state_obs_ph, self.state_look_ph, self.state_inv_ph, self.prev_action_ph, self.state_score_ph, self.action_ph])
        self.q_target_ = self.q_target([self.next_obs_ph, self.next_look_ph, self.next_inv_ph, self.action_ph, self.next_score_ph, self.q_target_best_action_ph])

        q_backup = self.reward_ph + (1 - self.done_ph) * self.gamma * self.q_target_ph
        q_loss = tf.keras.losses.mse(self.q_, q_backup)
        q_optimizer = tf.train.AdamOptimizer(self.learning_rate)

        self.q_train_op = q_optimizer.minimize(q_loss, var_list=self.q.trainable_variables)
        self.step_ops = [self.q_train_op] + \
                        [tf.reduce_mean(q_loss), tf.reduce_mean(self.q_)]

        optimizer_variables += q_optimizer.variables()

        source_params = self.q.trainable_variables
        target_params = self.q_target.trainable_variables
        self.target_update_op = [tf.assign(target, (1 - self.tau) * target + self.tau * source) for target, source in zip(target_params, source_params)]
    
        self.sess.run(tf.variables_initializer(optimizer_variables))
        self.sess.run(tf.variables_initializer(target_params))
        self.q_target.set_weights(self.q.get_weights())
        
        self.load_path = args.load_path
        if self.load_path is not None:
            self.load_weights(self.load_path)

    def sample_batch(self, obs, look, inv, bin_score, prev_action, action, valid_actions, reward, done, next_obs, next_look, next_inv, next_bin_score, batch_size):
        batch_idx = np.random.choice(np.arange(len(obs)), batch_size, replace=False)

        obs = obs[batch_idx]
        look = look[batch_idx]
        inv = inv[batch_idx]
        bin_score = bin_score[batch_idx]

        prev_action = prev_action[batch_idx]
        action = action[batch_idx]        
        valid_actions = valid_actions[batch_idx]
        reward = reward[batch_idx]
        done = done[batch_idx]

        next_obs = next_obs[batch_idx]
        next_look = next_look[batch_idx]
        next_inv = next_inv[batch_idx]
        next_bin_score = next_bin_score[batch_idx]

        return obs, look, inv, bin_score, prev_action, action, valid_actions, reward, done, next_obs, next_look, next_inv, next_bin_score

    def train(self, batch_size=128, epochs=100):
        np.random.seed(self.seed)

        start_time = time.time()
        episode_rewards = [0.0]
        eval_steps = 100
        target_update_steps = 1
        print_steps = 100

        for step in tqdm(range(epochs), desc='DRRN', ncols=70):
            batch_data = self.replay_buffer.sample(batch_size)
            step_info = self._train_step(batch_data)
            
            if step % print_steps == 0:
                print("\n=== Train Info. ===")
                print("- Avg. TD Error : %.3f" % step_info[0])
                print("- Avg. Q-Value  : %.3f" % step_info[1])

            if step % target_update_steps == 0:
                # print("\n=== Target Q Update ===")
                self.sess.run([self.target_update_op], feed_dict={})                

            if step % eval_steps == 0:
                self.step = step
                self.evaluate()

        f = open('outputs/eval_result_%s_%s.txt'%(self.args.game_name, self.args.uct_type), 'a')        
        f.write("- Round %d (qlearning_seed%d) : max_ep_return=%d\n" % (self.round, self.seed, self.best_episode_return))
        f.close()

    def get_best_q_target_action(self, next_obs, next_look, next_inv, next_score, prev_action, valid_actions):
        batch_size = len(valid_actions)

        next_obss  = np.tile(next_obs, self.max_valid_len).reshape([batch_size * self.max_valid_len, self.max_len_obs])
        next_looks = np.tile(next_look, self.max_valid_len).reshape([batch_size * self.max_valid_len, self.max_len_look])
        next_invs  = np.tile(next_inv, self.max_valid_len).reshape([batch_size * self.max_valid_len, self.max_len_inv])
        next_scores  = np.tile(next_score, self.max_valid_len).reshape([batch_size * self.max_valid_len, self.max_bin_score])
        prev_actions = np.tile(prev_action, self.max_valid_len).reshape([batch_size * self.max_valid_len, self.max_len_action])
        
        valid_actions_ = valid_actions.reshape([batch_size * self.max_valid_len, self.max_len_action])

        # Double Q-learning Implementation
        online_q = self.sess.run(self.q_, feed_dict={
                           self.state_obs_ph: next_obss,
                           self.state_look_ph : next_looks, 
                           self.state_inv_ph : next_invs,                            
                           self.prev_action_ph: prev_actions, 
                           self.state_score_ph : next_scores,
                           self.action_ph: valid_actions_})
        
        online_q = online_q.reshape([batch_size, self.max_valid_len])
        best_online_q_action_idx = np.argmax(online_q, axis=-1)
        best_online_q_action = np.array([valid_actions[i, act_idx] for i, act_idx in enumerate(best_online_q_action_idx)])

        best_q_target = self.sess.run(self.q_target_, feed_dict={
                            self.next_obs_ph: next_obs,
                            self.next_look_ph : next_look, 
                            self.next_inv_ph : next_inv,                             
                            self.action_ph: prev_action, 
                            self.next_score_ph : next_score, 
                            self.q_target_best_action_ph: best_online_q_action})
                            
        # q_target = q_target.reshape([batch_size, self.max_valid_len])
        # best_qtarget = np.max(q_target, axis=-1).reshape([-1,1])
        # best_q_target_action_idx = np.argmax(q_target, axis=-1)

        return best_online_q_action_idx, best_q_target

    def _train_step(self, batch_data):
        obs, look, inv, score, prev_action, action, valid_actions, reward, done, next_obs, next_look, next_inv, next_score = batch_data
        q_target_best_action, q_target = self.get_best_q_target_action(next_obs, next_look, next_inv, next_score, action, valid_actions)

        step_result = self.sess.run(self.step_ops, feed_dict={
            self.state_obs_ph:              obs,
            self.state_look_ph:             look,
            self.state_inv_ph:              inv,
            self.state_score_ph:            score,            
            self.prev_action_ph:            prev_action,
            self.action_ph:                 action,
            self.reward_ph:                 reward.reshape(-1,1),            
            self.done_ph:                   done.reshape(-1,1),
            self.next_obs_ph:               next_obs,
            self.next_look_ph:              next_look,
            self.next_inv_ph:               next_inv,
            self.next_score_ph:             next_score,
            self.q_target_ph:               q_target
        })
        # print('training step time')
        # print(time.time() - start_time)

        return step_result[1:]

    def get_q_value(self, str_obs, str_look, str_inv, str_prev_act, int_score, action):
        obs = self.sp.EncodeAsIds(str_obs)
        look = self.sp.EncodeAsIds(str_look)
        inv = self.sp.EncodeAsIds(str_inv)
        prev_act = self.sp.EncodeAsIds(str_prev_act)

        str_sign = '0' if int(int_score) >= 0 else '1'
        str_score = '{0:09b}'.format(abs(int(int_score)))
        str_score = '%s%s' % (str_sign, str_score)
        score = np.zeros((10,))
        for i in range(len(str_score)):
            score[i] = int(str_score[i])
        # score = self.sp.EncodeAsIds(str_score)

        obs  = pad_sequences([obs], self.max_len_obs)
        look = pad_sequences([look], self.max_len_look)
        inv  = pad_sequences([inv], self.max_len_inv)
        prev_act = pad_sequences([prev_act], self.max_len_action)
        score = pad_sequences([score], self.max_bin_score)

        action = self.sp.EncodeAsIds(action)
        action = pad_sequences([action], self.max_len_action)

        q_value = self.sess.run(self.q_, feed_dict={
                                    self.state_obs_ph: obs,
                                    self.state_look_ph : look,
                                    self.state_inv_ph : inv,
                                    self.prev_action_ph: prev_act,
                                    self.state_score_ph: score,
                                    self.action_ph: action})

        return q_value[0][0]

    def select_action(self, str_obs, str_look, str_inv, str_prev_act, int_score, valid_actions):
        obs = self.sp.EncodeAsIds(str_obs)
        look = self.sp.EncodeAsIds(str_look)
        inv = self.sp.EncodeAsIds(str_inv)        
        prev_act = self.sp.EncodeAsIds(str_prev_act)

        str_sign = '0' if int(int_score) >= 0 else '1'
        str_score = '{0:09b}'.format(abs(int(int_score)))
        str_score = '%s%s' % (str_sign, str_score)

        score = np.zeros((10,))
        for i in range(len(str_score)):
            score[i] = int(str_score[i])
        # score = self.sp.EncodeAsIds(str_score)

        num_valids = len(valid_actions)

        obs  = pad_sequences([obs] * num_valids, self.max_len_obs) 
        look = pad_sequences([look] * num_valids, self.max_len_look)
        inv  = pad_sequences([inv] * num_valids, self.max_len_inv)
        prev_act = pad_sequences([prev_act] * num_valids, self.max_len_action)
        score = pad_sequences([score] * num_valids, self.max_bin_score)

        valid_acts = []
        for va in valid_actions:
            va = self.sp.EncodeAsIds(va)
            valid_acts.append(va)
        
        valid_acts = pad_sequences(valid_acts, self.max_len_action)
        qvalues = self.sess.run(self.q_, feed_dict={
                                    self.state_obs_ph: obs, 
                                    self.state_look_ph : look, 
                                    self.state_inv_ph : inv, 
                                    self.prev_action_ph: prev_act,
                                    self.state_score_ph: score,
                                    self.action_ph: valid_acts})

        print("** OBS : ", str_obs)
        print("** LOOK : ", str_look)
        print("** INV: ", str_inv)
        print("** SCORE: ", str_score)
        print("** PREV_ACT: ", str_prev_act)
        
        for i in range(num_valids):
            print("[%02d] Valid : %s" % (i, valid_actions[i]))
            print("[%02d] Q-Value : %.4f" % (i, qvalues[i]))

        act = valid_actions[np.argmax(qvalues)]
        print("** BEST ACT : ", act)
        
        return act

    def evaluate(self, max_episode_len=50, save_result=False, learning_seed=0):
        seed = random.randint(0,1000)
        env = JerichoEnv(self.rom_path, seed, self.env_step_limit)
        env.create()
        
        eps_returns = []
        max_eval_iter = 1
        # max_episode_len = 30

        print("-- Evaluation Start --")

        for it in range(max_eval_iter):
            i = 0
            eps_return = 0
            done = False
            ob, info = env.reset()
            prev_action = "<s>"

            print("== Episode %d ==" % (it+1))
            print("** OBS : ", ob)
            print("** LOOK : ", info["look"])
            print("** INVEN : ", info["inv"])
            print("** SCORE : ", info["score"])
            print("** PREV_ACT : ", prev_action)
            
            while i < max_episode_len and not done:
                action = self.select_action(ob, info['look'], info['inv'], prev_action, info["score"], info['valid'])
                ob, reward, done, info = env.step(action, valid_out=True)
                prev_action = action
                eps_return += reward
                i += 1

            eps_returns.append(eps_return)

        print("-- Evaluation Ends --") 
        avg_return = np.mean(eps_returns)
        print("-- Avg. %d Episode return : %.5f" % (max_eval_iter, avg_return))
        
        if not self.eval_mode:
            save_dir = "weights/%s/round_%d/" % (self.game_name, self.round)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            # weights_filepath = save_dir + "%s_weight_return%d_seed%d.pickle" % (self.args.uct_type, avg_return, self.seed)
            # self.save(weights_filepath)
            
            if self.best_episode_return < avg_return:
                self.best_episode_return = avg_return
                best_weights_filepath = save_dir + "%s_weight_q_best_seed%d.pickle" % (self.args.uct_type, self.seed)
                self.save(best_weights_filepath)

            print("-- Best Episode return : %.5f" % self.best_episode_return)

        
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
