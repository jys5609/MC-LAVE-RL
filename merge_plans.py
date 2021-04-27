import glob
import argparse
import numpy as np


def parse_data(data_path):
    save_data = []
    save_reward = ''
    with open(data_path, 'r') as file:
        lines = file.readlines()

        max_reward = 0
        for rew_line in ' '.join(lines).split('Cum_reward:')[1:]:
            r = int(rew_line.split('\n')[0])
            max_reward = r if r > max_reward else max_reward

        max_score = max_reward

    with open(data_path, 'r') as file:
        new_line = []
        count = 0
        for line in file:
            if '#####' in line:
                new_line = []
            elif '[VALID_ACTION]'in line:
                if line.strip() != '':
                    new_line.append(line.strip())
            elif '[ACTION]' in line:
                if line.strip() != '':
                    new_line.append(line.strip())
            elif '[PREV_ACTION]' in line:
                if '<START>' in line:
                    new_line.append(line.replace('<START>', '<s>').strip())
                else:
                    new_line.append(line.strip())
            elif 'Reward' in line:
                save_reward = line.split('Cum_reward: ')[-1]
                sign = '0' if int(save_reward) >= 0 else '1'
                binary_reward = '{0:09b}'.format(abs(int(save_reward)))
                new_line.append('[REWARD] %s%s' % (sign, binary_reward))

                save_data.append(' '.join(new_line))
                count += 1
                print(count, ' '.join(new_line))
            else:
                if line.strip() != '':
                    new_line.append(line.strip())

    return save_data, save_reward

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--game_name', default='zork1', type=str)
    parser.add_argument('--round', default=0, type=int)
    parser.add_argument('--trial', default=0, type=int)
    parser.add_argument('--num_trajectory', default=100, type=int)
    parser.add_argument('--num_simulation', default=100, type=int)
    parser.add_argument('--exploration_constant', default=50, type=int)
    parser.add_argument('--max_depth', default=10, type=int)
    parser.add_argument('--uct_type', default='MC-LAVE', type=str)
    parser.add_argument('--replay_max_files', default=5, type=int)

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    round = args.round
    num_trajectory = args.num_trajectory
    num_simulation = args.num_simulation
    exploration_c = args.exploration_constant
    max_depth = args.max_depth
    task = args.game_name
    uct_type = args.uct_type

    print('task: ', task)
    print('selfplay_round: ', round)
    print('num of trajectory: ', num_trajectory)
    print('max_depth: ', max_depth)
    print('======================================')

    # Merge plans into 1 file
    log_dir = 'data/%s/%s_trial_%d/round_%d/' % (task, uct_type, args.trial, round)
    path = log_dir + 'mcts_log_d%02d_s%d_e%d_*.txt' % (max_depth, num_simulation, exploration_c)
    num_log_files = len(glob.glob(path))
    print(path)

    if num_log_files != num_trajectory:
        print('The number of log files %d is differnt from the num_condor_nodes %d' % (num_log_files, num_trajectory))
    # assert num_log_files == num_trajectory

    save_path = log_dir + 'merged_mcts_log_%02d.txt' % (max_depth)
    score_path = log_dir + 'score_mcts_log_%02d.txt' % (max_depth)

    save_file = open(save_path, 'w')
    save_score = open(score_path, 'w')

    all_data = []
    rewards = []

    for idx in range(num_trajectory):
        print (idx, num_trajectory)                            
        dir_name = log_dir + 'mcts_log_d%02d_s%d_e%d_%02d.txt' % (max_depth, num_simulation, exploration_c, idx)
        data, reward = parse_data(dir_name)
        rewards.append(reward)

        all_data.extend(data)
        save_score.write(reward)

    # all_data = list(set(all_data))
    save_file.write('\n'.join(all_data))

    save_file.close()
    save_score.close()   

    # Merge Replay buffer into 1 txt file
    file_num = min(num_trajectory, args.replay_max_files)

    f = open(log_dir + 'merged_replay.txt', 'w')
    save = []

    for i in range(file_num):
        file_path = log_dir + 'mcts_replay_d%02d_%02d.txt' % (max_depth, i)
        f_tmp = open(file_path, 'r')
        lines = f_tmp.readlines()
        for line in lines:
            if not line in save:
                save.append(line)
                f.write(line + '\n')
        f_tmp.close()
    f.close()

    # Record Evaluation Result on Planning
    f = open('outputs/eval_result_%s_%s.txt'%(task, args.uct_type), 'a')
    rewards = [int(r) for r in rewards] 
    f.write("- Round %d (planning_seed%d) : num_plans=%d, mean_ep_return=%.3f, std_ep_return=%.3f, min_ep_return=%.3f, max_ep_return=%.3f\n" % (int(args.round), int(args.trial), len(rewards), np.mean(rewards), np.std(rewards), np.min(rewards), np.max(rewards)))
    f.close()
