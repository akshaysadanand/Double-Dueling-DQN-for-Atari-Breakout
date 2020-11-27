import argparse
from test import test
from environment import Environment
import sys
import numpy as np

def parse():
    parser = argparse.ArgumentParser(description="DS595/CS525 RL Project 3")
    parser.add_argument('--env_name', default=None, help='environment name')
    parser.add_argument('--train_dqn', action='store_true', help='whether train DQN')
    parser.add_argument('--test_dqn', action='store_true', help='whether test DQN')
    try:
        from argument import add_arguments
        parser = add_arguments(parser)
    except Exception as e:
        print(e)
        pass
    args = parser.parse_args()
    return args


def run(args):
    if args.train_dqn:
        
        env_name ='BreakoutNoFrameskip-v4'
        env = Environment(env_name, args, atari_wrapper=True)
        from agent_dqn import Agent_DQN        
        best_score = -np.inf
        load_checkpoint = False   
        n_games = 10000000
        mem_size= args.mem_size
        idx_l = 0
        if load_checkpoint:
            epsilon=0.01
        else:
            epsilon = 1.0           
        

        agent = Agent_DQN(env, args)

        if load_checkpoint:
            agent.load_models()        

        n_steps = 0
        scores, eps_history, steps_array = [], [], []

        print('Filling Replay Buffer with %d steps' %(mem_size))

        
        n_mem_steps=0
        while n_mem_steps<=mem_size:        
            done = False
            observation=env.reset()
            
            while not done:
                if not load_checkpoint:
                    action = env.action_space.sample()
                else:                
                    action = agent.choose_action(observation)

                observation_, reward, done, info = env.step(action)            
                
                agent.push(observation, action,
                                         reward, observation_, int(done))

                observation = observation_
                n_mem_steps+=1


        for i in range(n_games):
            done = False
            observation = env.reset()
            
            
            score = 0
            while not done:
                action = agent.make_action(observation)
                
                
                observation_, reward, done, info = env.step(action)
                score += reward
                
                agent.push(observation, action,
                                         reward, observation_, int(done))
                
                agent.train()
                observation = observation_
                n_steps += 1
                
            scores.append(score)
            steps_array.append(n_steps)

            avg_score = np.mean(scores[-100:])
            print('episode: ', i,'score: ', score,
                 ' average score %.1f' % avg_score, 'best score %.2f' % best_score,
                'epsilon %.2f' % agent.epsilon, 'steps', n_steps)

            if avg_score > best_score:                
                agent.save_models()
                best_score = avg_score

            eps_history.append(agent.epsilon)
            
            #write stats to file to plot later
            if (i+1)%100==0:
                print('saving stats...')
                with open('stats.csv', 'a+') as csvfile:

                    stat=list(zip(steps_array[idx_l:],scores[idx_l:],eps_history[idx_l:]))
                    writer = csv.writer(csvfile)      
                    
                    for item in stat:
                        writer.writerow(item)
                    idx_l = len(stat)

    if args.test_dqn:
        env = Environment('BreakoutNoFrameskip-v4', args, atari_wrapper=True, test=True)
        from agent_dqn import Agent_DQN
        agent = Agent_DQN(env, args)
        test(agent, env, total_episodes=100)


if __name__ == '__main__':
    args = parse()
    run(args)
