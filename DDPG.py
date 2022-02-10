import tensorflow as tf
from tensorflow.keras import layers, regularizers
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import tracks
from VAE import VariationalAutoEncoder

racer = tracks.Racer()

########################################

# TODO:
#  -try batch normalization (and regularization) to make training more stable (in actor/critic)
#  -try random_normal initializer (in VAE)


# DDPG parameters
num_states = 5  # we reduce the state dim through observation (see below)
num_actions = 2  # acceleration and steering
print("State Space dim: {}, Action Space dim: {}".format(num_states, num_actions))

upper_bound = 1
lower_bound = -1

print("Min and Max Value of Action: {} / {}".format(lower_bound, upper_bound))

# VAE parameters
# original_dim is the number of items in a sample recorded in replay buffer,
# i.e. num_states * 2 (5 for s and 5 for s') + num_actions + 1 (reward field)
drop_out_prob = 0.1
original_dim = num_states * 2 + num_actions + 1
intermediate_dim = 64
latent_dim = 4


# the actor choose the move, given the state
def get_actor(train_acceleration=True, train_direction=True):
    # the actor has separate towers for action and speed
    # in this way we can train them separately

    inputs = layers.Input(shape=(num_states,))
    out1 = layers.Dense(32,
                        activation="relu",
                        kernel_regularizer=regularizers.l2(0.0001),
                        trainable=train_acceleration)(inputs)
    out1 = layers.Dense(32,
                        activation="relu",
                        kernel_regularizer=regularizers.l2(0.0001),
                        trainable=train_acceleration)(out1)
    out1 = layers.Dense(1,
                        activation='tanh',
                        kernel_regularizer=regularizers.l2(0.0001),
                        trainable=train_acceleration)(out1)

    out2 = layers.Dense(32,
                        activation="relu",
                        kernel_regularizer=regularizers.l2(0.0001),
                        trainable=train_direction)(inputs)
    out2 = layers.Dense(32,
                        activation="relu",
                        kernel_regularizer=regularizers.l2(0.0001),
                        trainable=train_direction)(out2)
    out2 = layers.Dense(1,
                        activation='tanh',
                        kernel_regularizer=regularizers.l2(0.0001),
                        trainable=train_direction)(out2)

    outputs = layers.concatenate([out1, out2])

    # outputs = outputs * upper_bound #resize the range, if required
    model = tf.keras.Model(inputs, outputs, name="actor")
    return model


# the critic compute the q-value, given the state and the action
def get_critic():
    # State as input
    state_input = layers.Input(shape=(num_states,))
    state_out = layers.Dense(16,
                             activation="relu",
                             kernel_regularizer=regularizers.l2(0.0001))(state_input)
    state_out = layers.Dense(32,
                             activation="relu",
                             kernel_regularizer=regularizers.l2(0.0001))(state_out)

    # Action as input
    action_input = layers.Input(shape=(num_actions,))
    action_out = layers.Dense(32,
                              activation="relu",
                              kernel_regularizer=regularizers.l2(0.0001))(action_input)

    concat = layers.Concatenate()([state_out, action_out])

    out = layers.Dense(64,
                       activation="relu",
                       kernel_regularizer=regularizers.l2(0.0001))(concat)
    out = layers.Dense(64,
                       activation="relu",
                       kernel_regularizer=regularizers.l2(0.0001))(out)
    outputs = layers.Dense(1)(out)  # outputs single value

    model = tf.keras.Model([state_input, action_input], outputs, name="critic")

    return model


class Buffer:
    def __init__(self, buffer_capacity=100000, batch_size=64):
        # number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity
        # num of tuples to train on.
        self.batch_size = batch_size

        # its tells us num of times record() was called.
        self.buffer_counter = 0

        # instead of list of tuples as the exp.replay concept go
        # we use different np.arrays for each tuple element
        self.state_buffer = np.zeros((self.buffer_capacity, num_states))
        self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, num_states))

    # takes (s,a,r,s') observation tuple as input
    def record(self, obs_tuple):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]

        self.buffer_counter += 1

        # if batch size element are collected, train VAE
        if index % self.batch_size == 0 and index > 0:
            print('Storing experience into VAE ...')
            vae_batch_index = self.buffer_counter - self.batch_size

            state_batch = self.state_buffer[vae_batch_index:self.buffer_counter]
            action_batch = self.action_buffer[vae_batch_index:self.buffer_counter]
            reward_batch = self.reward_buffer[vae_batch_index:self.buffer_counter]
            next_state_batch = self.next_state_buffer[vae_batch_index:self.buffer_counter]

            real_samples = np.column_stack((state_batch, action_batch, reward_batch, next_state_batch))

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
                    batch_size=32,
                    shuffle=True,
                    validation_split=0.25,
                    callbacks=[callback])

    @tf.function
    def update(
        self, state_batch, action_batch, reward_batch, next_state_batch,
    ):
        # training and updating actor & critic networks
        with tf.GradientTape() as tape:
            target_actions = target_actor(next_state_batch, training=True)
            y = reward_batch + gamma * target_critic(
                [next_state_batch, target_actions], training=True
            )
            critic_value = critic_model([state_batch, action_batch], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)
        critic_optimizer.apply_gradients(
            zip(critic_grad, critic_model.trainable_variables)
        )

        with tf.GradientTape() as tape:
            actions = actor_model(state_batch, training=True)
            critic_value = critic_model([state_batch, actions], training=True)
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
        actor_optimizer.apply_gradients(
            zip(actor_grad, actor_model.trainable_variables)
        )

    # we compute the loss and update parameters
    def learn(self):
        # get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # randomly sample indices for real samples
        batch_indices = np.random.choice(record_range, self.batch_size // 2)

        state_batch = self.state_buffer[batch_indices]
        action_batch = self.action_buffer[batch_indices]
        reward_batch = self.reward_buffer[batch_indices]
        next_state_batch = self.next_state_buffer[batch_indices]

        real_samples = np.column_stack((state_batch, action_batch, reward_batch, next_state_batch))

        # before sampling from replay buffer, generate some pseudo-samples
        random_vector_for_generation = tf.random.normal(shape=[self.batch_size // 2, latent_dim])

        pseudo_samples = vae.decoder(random_vector_for_generation)
        samples = np.concatenate((real_samples, pseudo_samples.numpy()), axis=0)
        np.random.shuffle(samples)

        state_batch = tf.convert_to_tensor(samples[:, :num_states])
        action_batch = tf.convert_to_tensor(samples[:, num_states:num_states + num_actions])
        reward_batch = tf.convert_to_tensor(samples[:, -num_states - 1:-num_states])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(samples[:, -num_states:])

        self.update(state_batch, action_batch, reward_batch, next_state_batch)


# slowly updating target parameters according to the tau rate <<1
@tf.function
def update_target(target_weights, weights, tau):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))


# noise generator (not usual random noise)
class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
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


def policy(state, noise=None, verbose=True):
    # the policy used for training just add noise to the action
    # the amount of noise is kept constant during training
    sampled_action = tf.squeeze(actor_model(state))

    # amount of noise changes during training
    noise = noise()

    # adding noise to action
    sampled_action = sampled_action.numpy() + noise

    # in verbose mode, we may print information about selected actions
    if verbose and sampled_action[0] < 0:
        print("decelerating")

    # finally, we ensure actions are within bounds
    legal_action = np.clip(sampled_action, lower_bound, upper_bound)

    return np.squeeze(legal_action)


std_dev = 0.2
ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))

# creating models
actor_model = get_actor()
critic_model = get_critic()
# actor_model.summary()
# critic_model.summary()

# we create the target model for double learning (to prevent a moving target phenomenon)
target_actor = get_actor()
target_critic = get_critic()


## TRAINING ##
# weights
# ddpg_critic_weigths_32_car0_split.h5 #versione con reti distinte per le mosse. Muove bene ma lento
# ddpg_critic_weigths_32_car1_split.h5 #usual problem: sembra ok

load_weights = False
save_weights = False  # beware when saving weights to not overwrite previous data

if load_weights:
    critic_model.load_weights("weights/ddpg_critic_weigths_32_car3_split.h5")
    actor_model.load_weights("weights/ddpg_actor_weigths_32_car3_split.h5")

# making the weights equal initially
target_actor_weights = actor_model.get_weights()
target_critic_weights = critic_model.get_weights()
target_actor.set_weights(target_actor_weights)
target_critic.set_weights(target_critic_weights)

# learning rate for actor-critic models
actor_lr = 0.001
critic_lr = 0.001
vae_lr = 0.001

actor_optimizer = Adam(actor_lr)
critic_optimizer = Adam(critic_lr)
vae_optimizer = Adam(vae_lr)

total_episodes = 30000

# target network parameter update factor, for double DQN
tau = 0.005
# discount factor
gamma = 0.99

buffer = Buffer(50000, 64)
vae = VariationalAutoEncoder(original_dim, intermediate_dim, latent_dim, drop_out_prob)
vae.compile(vae_optimizer, loss=tf.keras.losses.MeanSquaredError())

# history of rewards per episode
ep_reward_list = []
# average reward history of last few episodes
avg_reward_list = []
# ep speed tracker
ep_speed_list = []
# avg speed tracker
avg_speed_list = []
# number of steps per episode
number_of_steps = []


# custom observation of the state
# it must return an array to be passed as input to both actor and critic

# we extract from the lidar signal the angle dir corresponding to maximal distance max_dir from track borders
# as well as the the distance at adjacent positions.
def max_lidar(observation, angle=np.pi/3, pins=19):
    arg = np.argmax(observation)
    direction = -angle / 2 + arg * (angle / (pins - 1))
    dist = observation[arg]

    if arg == 0:
        distl = dist
    else:
        distl = observation[arg-1]
    if arg == pins-1:
        distr = dist
    else:
        distr = observation[arg+1]

    return direction, (distl, dist, distr)


def observe(racer_state):
    if racer_state is None:
        return np.array([0])  # not used; we could return None
    else:
        lidar_signal, v = racer_state
        direction, (distl, dist, distr) = max_lidar(lidar_signal)
        return np.array([direction, distl, dist, distr, v])


def train(total_episodes=total_episodes):

    for ep in range(total_episodes):
        # step per episode init
        episode_steps = 0

        prev_state = observe(racer.reset())
        episodic_reward = 0
        speed = prev_state[4]
        done = False

        while not done:
            # step increment
            episode_steps += 1

            tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)

            # our policy is always noisy
            action = policy(tf_prev_state, ou_noise)

            # get state and reward from the environment
            state, reward, done = racer.step(action)

            # we distinguish between termination with failure (state = None) and successful termination
            # on track completion successful termination is stored as a normal tuple
            state = observe(state)

            buffer.record((prev_state, action, reward, state))
            if not done:
                speed += state[4]
            episodic_reward += reward

            buffer.learn()

            update_target(target_actor.variables, actor_model.variables, tau)
            update_target(target_critic.variables, critic_model.variables, tau)

            prev_state = state

        ep_reward_list.append(episodic_reward)
        episodic_speed = speed / episode_steps
        ep_speed_list.append(episodic_speed)

        # mean of last 10 episodes
        avg_reward = np.mean(ep_reward_list[-100:])
        avg_speed = np.mean(ep_speed_list[-100:])
        print("Episode {}: Avg. Reward = {}, Last reward = {}. Avg. speed = {}".format(ep,
                                                                                       avg_reward,
                                                                                       episodic_reward,
                                                                                       avg_speed))
        avg_reward_list.append(avg_reward)
        avg_speed_list.append(avg_speed)
        number_of_steps.append(episode_steps)

    if total_episodes > 0:
        if save_weights:
            critic_model.save_weights("weights/ddpg_critic_weigths_32_car3_split.h5")
            actor_model.save_weights("weights/ddpg_actor_weigths_32_car3_split.h5")
        # plotting Episodes versus Avg. Rewards
        plt.plot(avg_reward_list)
        plt.xlabel("Episode")
        plt.ylabel("Avg. Episodic Reward")
        plt.savefig('plots/run30000_gen_replay_more_VAE_layers_reward.png')
        # plt.show()
        plt.clf()

        # plotting Episodes versus Avg. Speed
        plt.plot(avg_speed_list)
        plt.xlabel("Episode")
        plt.ylabel("Avg. Speed")
        plt.savefig("plots/run30000_gen_replay_more_VAE_layers_speed.png")
        # plt.show()
        plt.clf()

        # plotting Episodes versus Number of steps
        plt.plot(number_of_steps)
        plt.xlabel("Episode")
        plt.ylabel("Steps")
        plt.savefig("plots/run3000_gen_replay_more_VAE_layers_steps.png")
        # plt.show()
        plt.clf()


train()


# used for animation gif
def actor(state):
    print("speed = {}".format(state[1]))
    state = observe(state)
    state = tf.expand_dims(state, 0)
    action = actor_model(state)
    print("acc = ", action[0, 0].numpy())
    return action[0]


tracks.newrun(racer, actor)
