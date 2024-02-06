import numpy as np
from copy import deepcopy
import torch
import gym

from environment_dqn_n import Production_DQN_Env, SkipStep
from ddpg_optimized.ddpg_evaluator import Evaluator
from ddpg_optimized.ddpg_ddpg import DDPG_agent
from ddpg_optimized.ddpg_utils import *
from init_vars import *


def train(num_iterations, agent, env,  evaluate, validate_steps, output, max_episode_length=None, debug=False):
    agent.is_training = True
    step = episode = episode_steps = 0
    episode_reward = 0.
    observation = None
    while step < num_iterations:
        # reset if it is the start of episode
        if observation is None:
            observation, _ = deepcopy(env.reset())
            agent.reset(observation)
        
        # agent pick action ...
        if step <= args.warmup:
            action = agent.random_action()
        else:
            action = agent.select_action(observation)

        # env response with next_observation, reward, terminate_info
        observation2, reward, done, info, _ = env.step(action)
        #observation2 = deepcopy(observation2)
        if max_episode_length and episode_steps >= max_episode_length -1:
            done = True
        
        # agent observe and update policy
        agent.observe(reward, observation2, done)
        if step > args.warmup :
            agent.update_policy()
        
        # [optional] evaluate
        if evaluate is not None and validate_steps > 0 and step % validate_steps == 0:
            policy = lambda x: agent.select_action(x, decay_epsilon=False)
            validate_reward = evaluate(env, policy, debug=False)
            if debug: prYellow('[Evaluate] Step_{:07d}: mean_reward:{}'.format(step, validate_reward))
        
        # [optional] save intermideate model
        if step % int(num_iterations/3) == 0:
            agent.save_model(output)
        
        # update 
        step += 1
        episode_steps += 1
        episode_reward += reward
        observation = deepcopy(observation2)

        if done: # end of episode
            if debug: prGreen('#{}: episode_reward:{} steps:{}'.format(episode,episode_reward,step))

            agent.memory.append(
                observation,
                agent.select_action(observation),
                0., False
            )

            # reset
            observation = None
            episode_steps = 0
            episode_reward = 0.
            episode += 1


def test(num_episodes, agent, env, evaluate, model_path, visualize=True, debug=False):
    agent.load_weights(model_path)
    agent.is_training = False
    agent.eval()
    policy = lambda x: agent.select_action(x, decay_epsilon=False)

    for i in range(num_episodes):
        validate_reward = evaluate(env, policy, debug=debug, save=False)
        if debug: prYellow('[Evaluate] #{}: mean_reward:{}'.format(i, validate_reward))


def main():
    args.output = get_output_folder(args.output, args.env)
    if args.resume == 'default':
        args.resume = 'output/{}-run0'.format(args.env)
    
    env = SkipStep(Production_DQN_Env(), skip=24) if args.env == 'SkipStep' else Production_DQN_Env()

    if args.seed > 0:
        np.random.seed(args.seed)
        env.seed(args.seed)
    
    nb_states = env.observation_space.shape[0]
    nb_actions = env.action_space.shape[0]

    agent = DDPG_agent(nb_states, nb_actions, args)
    evaluate = Evaluator(args.validate_episodes, 
        args.validate_steps, args.output, max_episode_length=args.max_episode_length)
    
    if args.mode == 'train':
        train(args.train_iter, agent, env, evaluate, 
            args.validate_steps, args.output, max_episode_length=args.max_episode_length, debug=args.debug)
    elif args.mode == 'test':
        test(args.validate_episodes, agent, env, evaluate, args.resume,
            visualize=True, debug=args.debug)
    else:
        raise RuntimeError('undefined mode {}'.format(args.mode))