import argparse
import numpy as np
from env import JerichoEnv
import sentencepiece as spm
from policy import Policy
import utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rom_path', default='../envs/jericho-game-suite/', type=str)
    parser.add_argument('--game_name', default='ztuu', type=str)
    parser.add_argument('--data_path', default='data/GAME/round', type=str)
    parser.add_argument('--env_step_limit', default=100000, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--exploration_constant', default=50, type=int)
    parser.add_argument('--max_depth', default=20, type=int)
    parser.add_argument('--round', default=0, type=int)
    parser.add_argument('--simulation_per_act', default=1000, type=int)
    parser.add_argument('--min_simulation_per_act', default=30, type=int)
    parser.add_argument('--discount_factor', default=0.95, type=float)
    parser.add_argument('--backup_type', default='softmax', type=str)
    parser.add_argument('--uct_type', default='PLAVE', type=str)
    parser.add_argument('--epoch', default=1000, type=int)
    parser.add_argument('--max_episode_len', default=35, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    return parser.parse_args()


def main():
    args = parse_args()
    print(args)

    maxlen_obs = 150
    maxlen_look = 150
    maxlen_inv = 50
    max_len_action = 12

    sp = spm.SentencePieceProcessor()
    sp.Load('../spm_models/unigram_8k.model')

    rom_path = args.rom_path + utils.game_file(args.game_name)

    policy = Policy(args)
    # policy.model.load_weights('weights/%s_%s_round%s.5000.h5' % (args.game_name, args.uct_type, args.round))
    policy.load_weights('gcp/weights/%s/round_%d/%s_weight_policy_best_seed%d.pickle' % (args.game_name, args.round, args.uct_type, args.seed))
    # 63 / 100
    env = JerichoEnv(rom_path, 1, args.env_step_limit)
    env.create()

    scores = []

    for seed in range(5):
        env = JerichoEnv(rom_path, seed, args.env_step_limit)
        env.create()

        obs, info = env.reset()
        cum_reward = 0
        step = 0
        prev_action = '<s>'

        # livingroom_steps = ["S", "E"]
        #
        # for action in livingroom_steps:
        #     obs, reward, done, info = env.step(action)
        # prev_action = action

        for _ in range(args.max_episode_len):
            print('#################################################')
            print('STEP: %s' % step)
            print()
            print(info['look'])
            print()
            print(info['inv'])
            print()

            obs, look, inv, prev_action, score = utils.state_representation(obs, info['look'], info['inv'],
                                                                            prev_action, info['score'],
                                                                            maxlen_obs, maxlen_look, maxlen_inv, max_len_action)
            probs = policy.calculate_probs(obs, look, inv, prev_action, score, info['valid'])
            print(info['valid'])
            print(probs)
            idx = np.argmax(probs)
            # idx = int(np.random.choice([i for i in range(probs.shape[0])], 1, p=probs[:,0]))
            action = info['valid'][idx]

            obs, reward, done, info = env.step(action)
            cum_reward += reward
            step += 1

            print('ACTION: %s' % action)
            print()
            print('Reward: %s, Score: %s' % (reward, info['score']))
            print()
            print(obs + info['look'] + info['inv'])
            print()

            prev_action = action

        scores.append(info['score'])

    print(scores)
    print('AVERAGE SCORE: %s' % np.mean(scores))

    f = open('outputs/eval_result_%s_%s.txt'%(args.game_name, args.uct_type), 'a')
    f.write("- Round %d (learning) : num_eval=%d, mean_ep_return=%.3f, std_ep_return=%.3f\n" % \
            (args.round, len(scores), np.mean(scores), np.std(scores)))
    f.close()


if __name__ == "__main__":
    main()
