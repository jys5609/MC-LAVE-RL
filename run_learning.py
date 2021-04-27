import argparse
from src.qnetwork import DRRN
from src.policy import Policy
import src.utils as utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--game_name', default='zork1')
    parser.add_argument('--rom_path', default='envs/jericho-game-suite/')
    parser.add_argument('--qnet_iter', default=10000, type=int)
    parser.add_argument('--policy_epoch', default=100, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--load_path', default=None, type=str)
    parser.add_argument('--data_path', default='data/GAME/round', type=str)
    parser.add_argument('--env_step_limit', default=100000, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--exploration_constant', default=50, type=int)
    parser.add_argument('--max_depth', default=10, type=int)
    parser.add_argument('--max_episode_len', default=50, type=int)
    parser.add_argument('--round', default=0, type=int)
    parser.add_argument('--simulation_per_act', default=50, type=int)
    parser.add_argument('--discount_factor', default=0.95, type=float)
    parser.add_argument('--uct_type', default='MC-LAVE', type=str)
    parser.add_argument('--evaluate', default=False)
    parser.add_argument('--process_id', default=0, type=int)
    parser.add_argument('--mode', default=0, type=int) # 0 : train DRRN, 1 : behavior clone, 2 : both

    return parser.parse_args()


def main():
    args = parse_args()
    args.rom_path = args.rom_path + utils.game_file(args.game_name)

    print(args)

    if args.mode != 1:
        drrn = DRRN(args)
        drrn.train(batch_size=args.batch_size, epochs=args.qnet_iter)        
        
    elif args.mode != 0:
        policy = Policy(args)
        policy.train(epochs=args.policy_epoch)           

    else:
        print("Argument Error!!")


if __name__ == "__main__":
    main()
