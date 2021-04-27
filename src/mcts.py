import numpy as np
from tqdm import tqdm
import utils
from collections import defaultdict
from qnetwork import DRRN


class StateNode:
    def __init__(self, reward=0, done=False):
        self.ob = None
        self.look = None
        self.inv = None
        self.state = None
        self.prev_action = None
        self.id = None
        self.valid_actions = None

        self.N = 0
        self.children = []
        self.children_probs = []
        self.reward = reward
        self.score = 0
        self.done = done


class ActionNode:
    def __init__(self, action):
        self.action = action
        self.N = 0
        self.Q = 0
        self.Q_hat = 0
        self.Rs = []
        self.children = []
        self.children_text = []


class MCTSAgent:
    def __init__(self, args, env, policy=None, name='MCTS', uct_type='UCT', valid_action_dict=None, actions_info=None, log_dir=None, visited_transitions=None, replay_file=None):
        self.env = env
        self.name = name

        self.uct_type = uct_type
        self.seed = args.seed
        self.round = args.round

        self.exploration_constant = args.exploration_constant
        self.bonus_constant = args.bonus_constant
        self.max_depth = args.max_depth
        self.simulation_per_act = args.simulation_per_act
        self.discount_factor = args.discount_factor
        self.visited_transitions = visited_transitions

        self.action_selection_temp = 0.1 / (self.round + 1)

        self.policy = policy
        self.actions = [] if actions_info is None else actions_info[0]
        self.actions_e = [] if actions_info is None else actions_info[1]

        self.action_values = defaultdict(set)   # Ex: {north: [3.01, 2.00, 5.01]}

        self.maxlen_obs = 150
        self.maxlen_look = 150
        self.maxlen_inv = 50
        self.maxlen_action = 12
        
        if self.uct_type == 'MC-LAVE':
            self.q_network = DRRN(args)
        else:
            self.q_network = None

        self.valid_action_dict = {} if valid_action_dict is None else valid_action_dict
        self.state_dict = {}
        self.action_embedding = {}
        self.replay_file = replay_file

    def search(self, ob, info, cur_depth):
        '''
        Search the best action with probs
        :return: best action
        '''
        self.root = self.build_state(ob, info)
        for _ in tqdm(range(self.simulation_per_act * len(self.root.children))):
            copy_env = self.env.copy()
            self.simulate(self.root, copy_env, 0)

        # select best action by Q-value
        best_action_node = self.greedy_action_node(self.root, 0, 0)
        # select best action by Count
        # best_action_node = self.max_visit_action_node(self.root)
        
        return self.root, best_action_node.action, self.visited_transitions

    def build_state(self, ob, info, reward=0, prev_action='<s>'):
        state = StateNode()
        state.ob = ob
        state.look = info['look']
        state.inv = info['inv']
        state.state = ob + info['look'] + info['inv']
        state.reward = reward
        state.score = info['score']
        state.prev_action = prev_action
        state.id = ob + info['look'] + info['inv'] + str(reward) + str(info['score']) + prev_action
        state.valid_actions = info['valid']

        if self.policy is None:
            state.children_probs = np.ones((len(info['valid']),)) / len(info['valid'])

        elif state.id in self.state_dict.keys():
            state.children_probs = self.state_dict[state.id].children_probs
            
        else:
            obs, look, inv, prev_action, score = utils.state_representation(state.ob, state.look, state.inv,
                                                          state.prev_action, state.score,
                                                          self.maxlen_obs, self.maxlen_look, self.maxlen_inv, self.maxlen_action)
            state.children_probs = self.policy.calculate_probs(obs, look, inv, prev_action, score, info['valid'])[:,0]
            self.state_dict[state.id] = state

        for valid_action in info['valid']:
            state.children.append(ActionNode(valid_action))

        return state

    def write_buffer(self, state_node, best_action_node, ob, reward, done, info):
        # Example of transition strings :
        # [OBS] Outer Court 
        # [LOOK] Outer Court High walls surround this court, and a long pathway leads from the gates to the east to the throne room to the west.  Most of the palace is locked securely, but the courtyard is still open, for the very reason of allowing a Dwarven Reclaimer to have access to the Throne Room.
        # [INV] You are carrying: a piece of coal a lantern (providing light) a magical torch (providing light) King's Order
        # [VALID_ACTION] examine west [VALID_ACTION] west [VALID_ACTION] east [VALID_ACTION] put coal down [VALID_ACTION] put magical down [VALID_ACTION] put order down [VALID_ACTION] put lantern down 
        # [ACTION] east 
        # [PREV_ACTION] east 
        # [REWARD] 0000011100
        
        def clean_string(input_str):
            return " ".join(input_str.split('\n')).strip()

        obs = '[OBS] ' + clean_string(state_node.ob)
        look = '[LOOK] ' + clean_string(state_node.look)
        inven = '[INV] ' + clean_string(state_node.inv)
        sign = '0' if int(state_node.score) >= 0 else '1'
        score_string = '{0:09b}'.format(abs(int(state_node.score)))
        score = '[SCORE] %s%s ' % (sign, score_string)

        prev_action = '[PREV_ACTION] ' + clean_string(state_node.prev_action)
        action = '[ACTION] ' + clean_string(best_action_node.action)
        valid_actions = '[VALID_ACTION] ' + ' [VALID_ACTION] '.join(state_node.valid_actions)
        
        next_sign = '0' if int(state_node.score + reward) >= 0 else '1'
        next_score_string = '{0:09b}'.format(abs(int(state_node.score  + reward)))

        reward = '[REWARD] %d ' % reward
        next_score = '[NEXT_SCORE] %s%s ' % (next_sign, next_score_string)
        
        done = '[DONE] %d ' % done
        next_obs = '[NEXT_OBS] ' + clean_string(ob)
        next_look = '[NEXT_LOOK] ' + clean_string(info['look'])
        next_inven = '[NEXT_INV] ' + clean_string(info['inv'])

        transition = [obs, look, inven, score, prev_action, action, valid_actions, reward, done, next_obs, next_look, next_inven, next_score]
        transition_str = " ".join(transition) + '\n'
        
        if self.replay_file is not None:
            self.replay_file.write(transition_str)
        
    def simulate(self, state_node, copy_env, depth):
        if state_node.done or depth == self.max_depth or (state_node.look == 'unknown' and state_node.inv == 'unknown'):
            return 0
        
        best_action_node = self.greedy_action_node(state_node, self.exploration_constant, self.bonus_constant)

        rollout_next = False

        ob, reward, done, info = copy_env.step(best_action_node.action, valid_out=False)
        next_state_text = ob + info['look'] + info['inv']

        if '*** You have won ***' in next_state_text or '*** You have died ***' in next_state_text:
            score = int(next_state_text.split('you scored ')[1].split(' out of')[0])
            reward = score - state_node.score
            info['score'] = score

        self.write_buffer(state_node, best_action_node, ob, reward, done, info)

        if next_state_text in best_action_node.children_text:
            index = best_action_node.children_text.index(next_state_text)
            next_state_node = best_action_node.children[index]

            if next_state_node.N == 0:
                rollout_next = True
            next_state_node.N += 1

        else:
            if next_state_text in self.valid_action_dict.keys():
                info['valid'] = self.valid_action_dict[next_state_text]
            else:
                info['valid'] = copy_env.get_valid(ob)
                self.valid_action_dict[next_state_text] = info['valid']
            next_state_node = self.build_state(ob, info, reward, prev_action=best_action_node.action)
            best_action_node.children.append(next_state_node)
            best_action_node.children_text.append(next_state_node.state)
            rollout_next = True

        if rollout_next:
            R = reward + self.discount_factor * self.rollout(next_state_node, copy_env, depth+1)
        else:
            R = reward + self.discount_factor * self.simulate(next_state_node, copy_env, depth+1)

        state_node.N += 1
        best_action_node.N += 1

        if self.uct_type == 'MC-LAVE':
            if not best_action_node.action in self.action_embedding.keys():
                embed_vector = utils.vectorize(best_action_node.action)
                self.action_embedding[best_action_node.action] = embed_vector

            action_value = self.q_network.get_q_value(ob, info['look'], info['inv'], state_node.prev_action, info['score'], best_action_node.action)
            self.action_values[best_action_node.action].add(action_value)

        best_action_node.Rs.append(R)
        best_action_node.Q = np.sum(np.array(best_action_node.Rs) * utils.softmax(best_action_node.Rs, T=10))

        return R

    def max_visit_action_node(self, state_node):
        children_count = []

        for i in range(len(state_node.children)):
            child = state_node.children[i]
            children_count.append(child.N)

        children_count = children_count / np.max(children_count)
        count_based_probs = children_count ** (1/self.action_selection_temp) / (np.sum(children_count ** (1/self.action_selection_temp)))
        return np.random.choice(state_node.children, p=count_based_probs)

    def greedy_action_node(self, state_node, exploration_constant, bonus_constant):
        best_value = -np.inf
        best_children = []

        for i in range(len(state_node.children)):
            child = state_node.children[i]
            child_prob = state_node.children_probs[i]

            if exploration_constant == 0:
                ucb_value = child.Q
            elif self.uct_type == 'UCT':
                ucb_value = child.Q + exploration_constant * np.sqrt(np.log(state_node.N + 1) / (child.N + 1))
            elif self.uct_type == 'PUCT':
                ucb_value = child.Q + exploration_constant * child_prob * np.sqrt(state_node.N + 1) / (child.N + 1)
            elif self.uct_type == 'MC-LAVE':
                if child.action in self.action_embedding.keys():
                    action_e = self.action_embedding[child.action]
                else:
                    action_e = utils.vectorize(child.action)
                    self.action_embedding[child.action] = action_e

                actions = list(self.action_values.keys())
                if child.action in actions:
                    actions.pop(actions.index(child.action))

                actions_e = []
                for a in actions:
                    actions_e.append(self.action_embedding[a])

                near_act, near_idx = utils.find_near_actions(action_e, actions, np.array(actions_e), threshold=0.8)
                if len(near_idx) == 0:
                    child.Q_hat = 0
                else:
                    near_Qs = set()
                    for a in near_act:
                        near_Qs.add(np.mean(list(self.action_values[a])))
                    near_Qs = list(near_Qs)
                    child.Q_hat = utils.softmax_value(near_Qs)

                ucb_value = child.Q \
                            + exploration_constant * np.sqrt(state_node.N + 1) / (child.N + 1) * child_prob \
                            + bonus_constant * np.sqrt(state_node.N + 1) / (child.N + 1) * child.Q_hat

            else:
                raise NotImplementedError

            if ucb_value == best_value:
                best_children.append(child)
            elif ucb_value > best_value:
                best_value = ucb_value
                best_children = [child]

        return np.random.choice(best_children)

    def rollout(self, state_node, copy_env, depth):
        if state_node.done or depth == self.max_depth or (state_node.look == 'unknown' and state_node.inv == 'unknown'):
            return 0

        action_node = np.random.choice(state_node.children, 1, p=state_node.children_probs)[0]
        action = action_node.action

        ob, reward, done, info = copy_env.step(action, valid_out=False)
        next_state_text = ob + info['look'] + info['inv']

        if '*** You have won ***' in next_state_text or '*** You have died ***' in next_state_text:
            score = int(next_state_text.split('you scored ')[1].split(' out of')[0])
            reward = score - state_node.score
            info['score'] = score

        self.write_buffer(state_node, action_node, ob, reward, done, info)

        if next_state_text in action_node.children_text:
            index = action_node.children_text.index(next_state_text)
            next_state_node = action_node.children[index]
        else:
            if next_state_text in self.valid_action_dict.keys():
                info['valid'] = self.valid_action_dict[next_state_text]
            else:
                info['valid'] = copy_env.get_valid(ob)
                self.valid_action_dict[next_state_text] = info['valid']
            next_state_node = self.build_state(ob, info, reward, prev_action=action)
            action_node.children.append(next_state_node)
            action_node.children_text.append(next_state_node.state)

        return reward + self.discount_factor * self.rollout(next_state_node, copy_env, depth+1)

