import argparse
import numpy as np
from src.mcts import MCTSAgent
from src.env import JerichoEnv
import src.utils as utils
from src.policy import Policy
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rom_path', default='envs/jericho-game-suite/', type=str)
    parser.add_argument('--game_name', default='zork1', type=str)
    parser.add_argument('--data_path', default='data/GAME', type=str)
    parser.add_argument('--env_step_limit', default=100000, type=int)
    parser.add_argument('--trial', default=0, type=int)
    parser.add_argument('--process_id', default=0, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--exploration_constant', default=50, type=int)
    parser.add_argument('--bonus_constant', default=1, type=int)
    parser.add_argument('--max_episode_len', default=50, type=int)
    parser.add_argument('--max_depth', default=10, type=int)
    parser.add_argument('--round', default=0, type=int)
    parser.add_argument('--simulation_per_act', default=50, type=int)
    parser.add_argument('--discount_factor', default=0.95, type=float)
    parser.add_argument('--uct_type', default='MC-LAVE', type=str)
    parser.add_argument('--save_cache', action='store_true', default=False)
    parser.add_argument('--load_cache', action='store_true', default=False)
    parser.add_argument('--load_path', default=None, type=str)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--evaluate', default=True)
    return parser.parse_args()


def main():    
    args = parse_args()
    print(args)

    args.rom_path = args.rom_path + utils.game_file(args.game_name)
    data_path = args.data_path.replace('GAME', args.game_name)

    if args.seed is None:
        import random
        args.seed = random.randint(0,1000)
    
    np.random.seed(args.seed)

    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.Load('spm_models/unigram_8k.model')

    log_dir = data_path + '/%s_trial_%d/round_%d/' % (args.uct_type, args.trial, args.round)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    env = JerichoEnv(args.rom_path, args.seed, args.env_step_limit)
    env.create()
    visited_transitions = []
    
    ob, info = env.reset()
    
    done = False
    cum_reward = info['score']
    step = 0

    if args.load_cache:
        try:
            valid_action_dict = np.load('cache/%s_valid_action_dict.npy' % args.game_name, allow_pickle=True)[()]
        except EOFError:
            print("EOFError: skip loading cache..")
            valid_action_dict = None        
        except OSError:
            print("OSError: skip loading cache..")
            valid_action_dict = None    
        
    else:
        valid_action_dict = None

    actions_info = None

    prev_action = '<START>'

    if args.round == 0:
        policy = None
    elif args.round > 0:
        policy = Policy(args)
        policy.load_weights('weights/%s/round_%s/%s_weight_policy_best_seed%d.pickle' % (args.game_name, args.round - 1, args.uct_type, args.trial))
        args.load_path = 'weights/%s/round_%s/%s_weight_q_best_seed%d.pickle' % (args.game_name, args.round - 1, args.uct_type, args.trial)
    else:
        raise NotImplementedError

    import time
    start = time.time()

    log_file = log_dir + 'mcts_log_d%02d_s%d_e%d_%02d.txt'\
               % (args.max_depth, args.simulation_per_act, args.exploration_constant, args.seed)    
    data = open(log_file, 'w')
    replay_buffer_filename = log_dir + 'mcts_replay_d%02d_%02d.txt' % (args.max_depth, args.seed)
    replay_buffer_file = open(replay_buffer_filename, 'w')

    for cur_depth in range(args.max_episode_len):
        agent = MCTSAgent(args, env.copy(), policy, uct_type=args.uct_type, valid_action_dict=valid_action_dict, actions_info=actions_info, log_dir=log_dir, visited_transitions=visited_transitions, replay_file=replay_buffer_file)
        prev_action_str = '[PREV_ACTION] ' + prev_action + '\n'
        root_node, action, visited_transitions = agent.search(ob, info, cur_depth)

        data.write('#######################################################\n')
        state_str = '[OBS] ' + ob + '\n' + '[LOOK] ' + info['look'] + '\n' + '[INV] ' + info['inv'] + '\n'
        valid_action_strs = ['[VALID_ACTION] ' + valid + '\n' for valid in info['valid']]
        action_str = '[ACTION] ' + action + '\n'

        data.write(state_str)
        for valid_action_str in valid_action_strs:
            data.write(valid_action_str)
        data.write(action_str)
        data.write(prev_action_str)

        ob, reward, done, info = env.step(action)

        cum_reward += reward
        score = info['score']
        step += 1

        next_ob_text = ob + info['look'] + info['inv']

        if '*** You have won ***' in next_ob_text or '*** You have died ***' in next_ob_text:
            score = int(next_ob_text.split('you scored ')[1].split(' out of')[0])
            reward = score - cum_reward

        data.write('Reward: %d, Cum_reward: %d \n' % (reward, score))

        for action_node in root_node.children:
            data.write('%s Q_val: %f Q_hat: %f count: %d \n' % (action_node.action, action_node.Q, action_node.Q_hat, action_node.N))

        prev_action = action

        print('##########################')
        print('STEP: %s' % step)
        print(root_node.state)
        print()
        print('BEST_ACTION: ', action)
        print()
        print('Valid actions:', [action.action for action in root_node.children])
        print('Q-values', [action.Q for action in root_node.children])
        print('Q-hat', [action.Q_hat for action in root_node.children])
        print('Final Q', [action.Q + action.Q_hat for action in root_node.children])
        print('Maximum Q', [0 if len(action.Rs) == 0 else max(action.Rs) for action in root_node.children])
        print('Count of actions', [action.N for action in root_node.children])
        print('Action Probs:', [prob for prob in root_node.children_probs])
        print()
        print('Reward: %s, CUM_Reward: %s' % (reward, score))
        print()
        print(ob + info['look'] + info['inv'])
        print(flush=True)

        valid_action_dict = agent.valid_action_dict
        actions_info = [agent.actions, agent.actions_e]
    
        if args.save_cache:
            np.save('cache/%s_valid_action_dict.npy' % args.game_name, valid_action_dict)

        if '*** You have won ***' in next_ob_text or '*** You have died ***' in next_ob_text:
            break

    print('TOTAL TIME: ', time.time() - start)
    data.close()
    replay_buffer_file.close()

if __name__ == "__main__":
    main()
