import argparse

import tensorflow as tf
from tensorflow.keras import layers, regularizers
import numpy as np
import matplotlib.pyplot as plt
import tracks
from tensorflow.keras.optimizers import Adam
from VAE import VariationalAutoEncoder
import os


# The actor choose the move, given the state
def get_actor():
    # no special initialization is required
    # Initialize weights between -3e-3 and 3-e3
    # last_init = tf.random_uniform_initializer(min_val=-0.003, max_val=0.003)

    inputs = layers.Input(shape=(num_states,))
    out = layers.Dense(64, activation="relu")(inputs)
    out = layers.Dense(64, activation="relu")(out)
    # outputs = layers.Dense(num_actions, kernel_regularizer = regularizers.l2(0.01), kernel_initializer=last_init)(out)
    # outputs = layers.Activation('tanh')(outputs)
    # outputs = layers.Dense(num_actions, name="out", activation="tanh", kernel_initializer=last_init)(out)
    outputs = layers.Dense(num_actions, name="out", activation="tanh")(out)

    # outputs = outputs * upper_bound
    model = tf.keras.Model(inputs, outputs, name="actor")
    return model


def get_actor_separate(train_acceleration=True, train_direction=True, regularizer=None):
    # the actor has separate towers for action and speed
    # in this way we can train them separately
    if regularizer is not None:
        l2 = regularizers.l2(regularizer)
    else:
        l2 = None

    inputs = layers.Input(shape=(num_states,))

    out1 = layers.Dense(32,
                        activation="relu",
                        kernel_regularizer=l2,
                        trainable=train_acceleration)(inputs)
    out1 = layers.Dense(32,
                        activation="relu",
                        kernel_regularizer=l2,
                        trainable=train_acceleration)(out1)
    out1 = layers.Dense(1,
                        activation='tanh',
                        kernel_regularizer=l2,
                        trainable=train_acceleration)(out1)

    out2 = layers.Dense(32,
                        activation="relu",
                        kernel_regularizer=l2,
                        trainable=train_direction)(inputs)
    out2 = layers.Dense(32,
                        activation="relu",
                        kernel_regularizer=l2,
                        trainable=train_direction)(out2)
    out2 = layers.Dense(1,
                        activation='tanh',
                        kernel_regularizer=l2,
                        trainable=train_direction)(out2)

    outputs = layers.concatenate([out1, out2])

    # outputs = outputs * upper_bound #resize the range, if required
    model = tf.keras.Model(inputs, outputs, name="actor")
    return model


# the critic compute the q-value, given the state and the action
def get_critic(regularizer=None):
    if regularizer is not None:
        l2 = regularizers.l2(regularizer)
    else:
        l2 = None
    # State as input
    state_input = layers.Input(shape=(num_states,))
    state_out = layers.Dense(16,
                             kernel_regularizer=l2,
                             activation="relu")(state_input)
    state_out = layers.Dense(32,
                             kernel_regularizer=l2,
                             activation="relu")(state_out)

    # Action as input
    action_input = layers.Input(shape=(num_actions,))
    action_out = layers.Dense(32,
                              kernel_regularizer=l2,
                              activation="relu")(action_input)

    concat = layers.Concatenate()([state_out, action_out])

    out = layers.Dense(64,
                       kernel_regularizer=l2,
                       activation="relu")(concat)
    out = layers.Dense(64,
                       kernel_regularizer=l2,
                       activation="relu")(out)

    outputs = layers.Dense(1)(out)  # outputs single value

    model = tf.keras.Model([state_input, action_input], outputs, name="critic")

    return model


# Replay buffer
class Buffer:
    def __init__(self, buffer_capacity=100000, batch_size=64):
        # Max Number of tuples that can be stored
        self.buffer_capacity = buffer_capacity
        # Num of tuples used for training
        self.batch_size = batch_size

        # Current number of tuples in buffer
        self.buffer_counter = 0

        # We have a different array for each tuple element
        self.state_buffer = np.zeros((self.buffer_capacity, num_states))
        self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.done_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, num_states))

    # Stores a transition (s,a,r,s') in the buffer
    def record(self, obs_tuple, use_vae):
        s, a, r, T, sn = obs_tuple
        # restart form zero if buffer_capacity is exceeded, replacing old records
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = tf.squeeze(s)
        self.action_buffer[index] = a
        self.reward_buffer[index] = r
        self.done_buffer[index] = T
        self.next_state_buffer[index] = tf.squeeze(sn)

        self.buffer_counter += 1

        # if batch size element are collected, train VAE
        if index % self.batch_size == 0 and index > 0 and use_vae:
            print('Storing experience into VAE ...')
            vae_batch_index = index - self.batch_size

            state_batch = self.state_buffer[vae_batch_index:index]
            action_batch = self.action_buffer[vae_batch_index:index]
            reward_batch = self.reward_buffer[vae_batch_index:index]
            done_batch = self.done_buffer[vae_batch_index:index]
            next_state_batch = self.next_state_buffer[vae_batch_index:index]

            real_samples = np.column_stack((state_batch, action_batch, reward_batch, done_batch, next_state_batch))

            # generate batch_size pseudo-states with VAE and then mix them wth real states
            # we may change sampling from latent space (e.g. Gibbs sampling)
            random_vector_for_generation = tf.random.normal(shape=[self.batch_size, latent_dim])

            pseudo_samples = vae.decoder(random_vector_for_generation)
            samples = np.concatenate((real_samples, pseudo_samples.numpy()), axis=0)
            np.random.shuffle(samples)

            # shuffle and convert to tensor
            samples = tf.convert_to_tensor(samples)

            callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
            vae.fit(x=samples,
                    y=samples,
                    epochs=100,
                    batch_size=96,
                    shuffle=True,
                    validation_split=0.25,
                    callbacks=[callback])

    def sample_batch(self, use_vae=False, buffer_bias=None, flip=False):
        # Get sampling range
        index = self.buffer_counter % self.buffer_capacity
        record_range = min(self.buffer_counter, self.buffer_capacity)
        if buffer_bias is not None:
            bias = np.array(range(record_range)) + 1
            if flip:
                bias = np.flip(bias)
            if buffer_bias == 'linear':
                bias = (bias / self.buffer_counter) + 1  # Move the bias according linearly
            elif buffer_bias == 'quadratic':
                bias = np.square((bias / self.buffer_counter) + 1)  # Move the bias according to exp or other functions
            elif buffer_bias == 'exponential':
                bias = np.exp((bias / self.buffer_counter) + 1)  # Move the bias according to exp or other functions
            bias = bias / np.sum(bias)
            bias = np.roll(bias, index)

            # Randomly sample indices
            batch_indices = np.random.choice(record_range, self.batch_size, p=bias)

        else:
            batch_indices = np.random.choice(record_range, self.batch_size)

        s = self.state_buffer[batch_indices]
        a = self.action_buffer[batch_indices]
        r = self.reward_buffer[batch_indices]
        T = self.done_buffer[batch_indices]
        sn = self.next_state_buffer[batch_indices]

        if use_vae:
            # before sampling from replay buffer, generate some pseudo-samples
            random_vector_for_generation = tf.random.normal(shape=[self.batch_size // 2, latent_dim])

            pseudo_samples = vae.decoder(random_vector_for_generation)
            np.random.shuffle(pseudo_samples.numpy())

            s[:self.batch_size // 2] = pseudo_samples[:, :num_states]
            a[:self.batch_size // 2] = pseudo_samples[:, num_states:num_states + num_actions]
            r[:self.batch_size // 2] = np.reshape(pseudo_samples[:, num_states + num_actions],
                                                  (self.batch_size // 2, 1))
            T[:self.batch_size // 2] = np.reshape(pseudo_samples[:, -(num_states + 1)], (self.batch_size // 2, 1))
            sn[:self.batch_size // 2] = pseudo_samples[:, -num_states:]

        return s, a, r, T, sn


# Slowly updating target parameters according to the tau rate <<1
@tf.function
def update_target(target_weights, weights, tau):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))


def update_weights(target_weights, weights, tau):
    return target_weights * (1 - tau) + weights * tau


def linear_policy_decay_rate(current_episode, total_episodes):
    return 1 - (current_episode / total_episodes)


def quadratic_policy_decay_rate(current_episode, total_episodes):
    return (1 - (current_episode / total_episodes)) ** 2


def reward_based_policy_decay_rate(rewards_list):
    if len(rewards_list) > 2:
        rewards_delta = rewards_list[-1] - rewards_list[-2]
        if rewards_delta > 0:
            decay_factor = 0
        else:
            decay_factor = 1
    else:
        decay_factor = 4
    return decay_factor


# noise generator (not usual random noise)
class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.x_prev = 0
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (
                self.x_prev
                + self.theta * (self.mean - self.x_prev) * self.dt
                + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)


def policy(state, verbose=False, current_episode=None, total_episodes=None, rewards_list=None,
           policy_decay=None, ou_noise=None):
    # the policy used for training just add noise to the action
    sampled_action = tf.squeeze(actor_model(state))
    noise = np.random.normal(scale=0.1, size=2)
    # the amount of noise is kept constant during training
    noise[0] *= 2
    noise[1] *= .5
    decay_factor = 1
    # we may change the amount of noise for actions during training
    if policy_decay is not None:
        if policy_decay == "linear":
            decay_factor = linear_policy_decay_rate(current_episode, total_episodes)
        elif policy_decay == "quadratic":
            decay_factor = quadratic_policy_decay_rate(current_episode, total_episodes)
        elif policy_decay == "rewards_based":
            decay_factor = reward_based_policy_decay_rate(rewards_list)
        elif policy_decay == "OUA":
            noise = ou_noise()
        # still more policies decay to try
        else:
            decay_factor = 1
    noise *= decay_factor
    # Adding noise to action
    sampled_action = sampled_action.numpy()
    sampled_action += noise
    # in verbose mode, we may print information about selected actions
    if verbose and sampled_action[0] < 0:
        print("decelerating")

    # Finally, we ensure actions are within bounds
    legal_action = np.clip(sampled_action, lower_bound, upper_bound)

    return [np.squeeze(legal_action)], decay_factor


# We compose actor and critic in a single model.
# The actor is trained by maximizing the future expected reward, estimated
# by the critic. The critic should be frozen while training the actor.
# For simplicity, we just use the target critic, that is not trainable.

def compose(actor, critic):
    state_input = layers.Input(shape=num_states)
    a = actor(state_input)
    q = critic([state_input, a])
    # reg_weights = actor.get_layer('out').get_weights()[0]
    # print(tf.reduce_sum(0.01 * tf.square(reg_weights)))

    m = tf.keras.Model(state_input, q)
    # the loss function of the compound model is just the opposite of the critic output
    m.add_loss(-q)
    return m


# custom observation of the state
# it must return an array to be passed as input to both actor and critic

# we extract from the lidar signal the angle dir corresponding to maximal distance max_dir from track borders
# as well as the the distance at adjacent positions.

def max_lidar(observation, angle=np.pi / 3, pins=19):
    arg = np.argmax(observation)
    dir = -angle / 2 + arg * (angle / (pins - 1))
    dist = observation[arg]
    if arg == 0:
        dist_l = dist
    else:
        dist_l = observation[arg - 1]
    if arg == pins - 1:
        dist_r = dist
    else:
        dist_r = observation[arg + 1]
    return dir, (dist_l, dist, dist_r)


def observe(racer_state):
    if racer_state is None:
        return np.array([0])  # not used; we could return None
    else:
        lidar_signal, v = racer_state
        dir, (dist_l, dist, dist_r) = max_lidar(lidar_signal)
        return np.array([dir, dist_l, dist, dist_r, v])


def train(total_episodes, gamma, tau, save_weights, weights_out_folder, out_name, plots_folder, lr_dict, policy_decay,
          use_vae, reward_type, buffer_bias):
    i = 0
    if policy_decay == "OUA":
        ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(0.2) * np.ones(1))
    else:
        ou_noise = None

    for ep in range(total_episodes):

        prev_state = observe(racer.reset())
        episodic_reward = 0
        episode_steps = 0
        policy_decay_factor = 0
        speed = prev_state[4]
        done = False

        while not done:  # and episode_steps < 500:
            i = i + 1
            episode_steps += 1

            tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)

            # our policy is always noisy
            action_list, policy_decay_factor = policy(state=tf_prev_state,
                                                      verbose=False,
                                                      current_episode=ep,
                                                      total_episodes=total_episodes,
                                                      rewards_list=avg_reward_list,
                                                      policy_decay=policy_decay,
                                                      ou_noise=ou_noise)
            action = action_list[0]
            # Get state and reward from the environment
            state, reward, done, _ = racer.step(action=action, reward_type=reward_type)
            # we distinguish between termination with failure (state = None) and successful termination on track
            # completion successful termination is stored as a normal tuple
            fail = done and state is None
            state = observe(state)
            buffer.record((prev_state, action, reward, fail, state), use_vae=use_vae)

            if not done:
                speed += state[4]

            episodic_reward += reward

            states, actions, rewards, dones, new_states = buffer.sample_batch(use_vae=use_vae, buffer_bias=buffer_bias)

            targetQ = rewards + (1 - dones) * gamma * (target_critic([new_states, target_actor(new_states)]))

            loss1 = critic_model.train_on_batch([states, actions], targetQ)
            loss2 = aux_model.train_on_batch(states)

            update_target(target_actor.variables, actor_model.variables, tau)
            update_target(target_critic.variables, critic_model.variables, tau)

            prev_state = state

        ep_reward_list.append(episodic_reward)
        episodic_speed = speed / episode_steps
        ep_speed_list.append(episodic_speed)

        # Mean of last sliding_window episodes
        sliding_window = max(40, total_episodes // 50)
        avg_reward = np.mean(ep_reward_list[-sliding_window:])
        avg_speed = np.mean(ep_speed_list[-sliding_window:])

        avg_reward_list.append(avg_reward)
        policy_decay_list.append(policy_decay_factor)
        avg_speed_list.append(avg_speed)

        print(f"----------------------------------------------------------------- \n"
              f"Episode {ep} --  Ep. Steps = {episode_steps} \n"
              f"Avg. Reward = {avg_reward} -- Last reward = {episodic_reward} \n"
              f"Avg. Speed = {avg_speed} -- Last speed = {episodic_speed} \n"
              f"Buffer counter: {buffer.buffer_counter}")

        if lr_dict is not None:
            if lr_dict['staircase']:
                exp = i // lr_dict['decay_steps']
            else:
                exp = i / lr_dict['decay_steps']
            exp_decay_rate = lr_dict['decay_rate'] ** exp
            exp_decay = exp_decay_rate * lr_dict['initial_learning_rate']

            # return initial_learning_rate * decay_rate ^ (step / decay_steps)

            learning_rate_list.append(exp_decay)

    if total_episodes > 0:
        if save_weights:
            actor_model.save_weights(f"{weights_out_folder}actor_{out_name}.h5")
            critic_model.save_weights(f"{weights_out_folder}critic_{out_name}.h5")
        # Plotting Episodes versus Avg. Rewards
        plt.plot(avg_reward_list)
        plt.xlabel("Episode")
        plt.ylabel("Avg. Episodic Reward")
        plt.savefig(f"{plots_folder}{out_name}_avg_reward.png")
        # plt.show()
        plt.clf()

        if lr_dict is not None:
            plt.plot(learning_rate_list)
            plt.xlabel("Episode")
            plt.ylabel("Learning Rate")
            plt.savefig(f"{plots_folder}{out_name}_lr_schedule.png")
            # plt.show()
            plt.clf()

        if policy_decay is not None:
            plt.plot(policy_decay_list)
            plt.xlabel("Episode")
            plt.ylabel("Policy Noise Factor")
            plt.savefig(f"{plots_folder}{out_name}_policy_noise.png")
            # plt.show()
            plt.clf()

        if False:
            plt.plot(ep_reward_list)
            plt.xlabel("Episode")
            plt.ylabel("Episodic Reward")
            plt.savefig(f"{plots_folder}{out_name}_episodic_reward.png")
            # plt.show()
            plt.clf()

        plt.plot(avg_speed_list)
        plt.xlabel("Episode")
        plt.ylabel("Average Speed")
        plt.savefig(f"{plots_folder}{out_name}_avg_speed.png")
        # plt.show()
        plt.clf()


def actor(state):
    # print("speed = {}".format(state[1]))
    state = observe(state)
    state = tf.expand_dims(state, 0)
    action = actor_model(state)
    # print("acc = ", action[0, 0].numpy())
    return action[0]


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--simulate', type=bool, default=False)  # True to activate training, False just sim
    parser.add_argument('--simulations', type=int, default=2)  # Number of simulations
    parser.add_argument('--train', type=bool, default=True)  # True to activate training, False just sim
    parser.add_argument('--episodes', type=int, default=30000)  # Number of episodes (training only)
    parser.add_argument('--gamma', type=float, default=0.99)  # Discount factor
    parser.add_argument('--tau', type=float, default=0.05)  # Target network parameter update factor, for double DQN
    parser.add_argument('--use_vae', type=bool, default=False)  # Use VAE to sample from buffer
    parser.add_argument('--policy_decay', type=str, default='quadratic')  # True to use exponential decay
    parser.add_argument('--lr_decay', type=bool, default=False)  # True to use exponential decay
    parser.add_argument('--lr', type=float, default=0.0001)  # Initial Learning Rate
    parser.add_argument('--regularizer', type=float, default=None)  # Regularizing Factor
    parser.add_argument('--reward', type=str, default='polar')  # None or polar
    parser.add_argument('--buffer_bias', type=str, default=None)  # Change sampling policy from buffer records
    parser.add_argument('--load_weights', type=bool, default=False)  # True to load pretrained weights
    parser.add_argument('--save_weights', type=bool, default=True)  # True to save trained weights
    parser.add_argument('--weights_in_folder', type=str, default="new_weights/")  # Weights input folder
    parser.add_argument('--weights_out_folder', type=str, default="new_weights/")  # Weights output folder
    parser.add_argument('--input_weights', type=str, default=None)  # Weights input file, critic
    parser.add_argument('--out_file', type=str, default="_scratch_polar_policy_quadratic_lr_1e4")  # Weights output file
    parser.add_argument('--plot_folder', type=str, default="plots/")  # Plots folder

    args = parser.parse_args()

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    racer = tracks.Racer()

    ########################################

    num_states = 5  # we reduce the state dim through observation (see below)
    num_actions = 2  # acceleration and steering
    print("State Space dim: {}, Action Space dim: {}".format(num_states, num_actions))

    upper_bound = 1
    lower_bound = -1

    print(f"Min and Max Value of Action: {lower_bound} : {upper_bound}")

    # creating models

    # VAE parameters
    # original_dim is the number of items in a sample recorded in replay buffer,
    # i.e. num_states (starting_state) + num_actions (actions) + 1 (reward) + 1 (done) + num_states (output_states)
    drop_out_prob = 0.1
    original_dim = num_states + num_actions + 1 + 1 + num_states
    intermediate_dim = 64
    latent_dim = 4

    # actor_model = get_actor()
    actor_model = get_actor_separate(regularizer=args.regularizer)
    critic_model = get_critic(regularizer=args.regularizer)

    # actor_model.summary()
    # critic_model.summary()

    # we create the target model for double learning (to prevent a moving target phenomenon)

    # target_actor = get_actor()
    target_actor = get_actor_separate()
    target_critic = get_critic()
    target_actor.trainable = False
    target_critic.trainable = False

    aux_model = compose(actor_model, target_critic)

    # TRAINING #
    # weights
    # ddpg_critic_weigths_32_car0_split.h5  # versione con reti distinte per le mosse. Muove bene ma lento
    # ddpg_critic_weigths_32_car1_split.h5  # usual problem: sembra ok

    if args.load_weights:
        actor_model.load_weights(f"{args.weights_in_folder}actor{args.input_weights}.h5")
        critic_model.load_weights(f"{args.weights_in_folder}critic{args.input_weights}.h5")

    # Making the weights equal initially
    target_actor_weights = actor_model.get_weights()
    target_critic_weights = critic_model.get_weights()
    target_actor.set_weights(target_actor_weights)
    target_critic.set_weights(target_critic_weights)

    vae = VariationalAutoEncoder(original_dim, intermediate_dim, latent_dim, drop_out_prob)

    if args.lr_decay:
        exp_decay_dict = {
            'initial_learning_rate': args.lr,
            'decay_steps': 1000,
            'decay_rate': 0.93,
            'staircase': True
        }

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            exp_decay_dict['initial_learning_rate'],
            decay_steps=exp_decay_dict['decay_steps'],
            decay_rate=exp_decay_dict['decay_rate'],
            staircase=exp_decay_dict['staircase'])
    else:
        exp_decay_dict = None
        lr_schedule = args.lr

    # Learning rate for actor-critic models
    critic_lr = lr_schedule
    aux_lr = lr_schedule
    vae_lr = lr_schedule

    critic_optimizer = Adam(critic_lr)
    aux_optimizer = Adam(aux_lr)
    vae_optimizer = Adam(vae_lr)

    critic_model.compile(loss='mse', optimizer=critic_optimizer)
    aux_model.compile(optimizer=aux_optimizer)
    vae.compile(loss='mse', optimizer=vae_optimizer)

    buffer = Buffer(50000, 64)

    # History of rewards per episode
    ep_reward_list = []
    # Average reward history of last few episodes
    avg_reward_list = []
    # Decaying learning rate tracker
    learning_rate_list = []
    # Policy noise factor tracker
    policy_decay_list = []
    # Ep speed tracker
    ep_speed_list = []
    # Avg speed tracker
    avg_speed_list = []

    # TRAIN and SIMULATE #
    if args.train:
        train(total_episodes=args.episodes,
              gamma=args.gamma,
              tau=args.tau,
              save_weights=args.save_weights,
              weights_out_folder=args.weights_out_folder,
              out_name=f"{args.episodes}{args.out_file}",
              plots_folder=args.plot_folder,
              lr_dict=exp_decay_dict,
              policy_decay=args.policy_decay,
              use_vae=args.use_vae,
              reward_type=args.reward,
              buffer_bias=args.buffer_bias)

    # for sim in range(simulations):
    #     tracks.new_run(racer, actor, sim)

    if args.simulate:
        tracks.new_multi_run(actor, args.simulations)

    actor_model.summary()
    aux_model.summary()

    exit(0)
