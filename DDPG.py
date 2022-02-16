import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras import regularizers
import numpy as np
import matplotlib.pyplot as plt
import tracks
from VAE import VariationalAutoEncoder

racer = tracks.Racer()

########################################
###### HYPERPARAMETERS #################

total_episodes = 200
# Discount factor
gamma = 0.99
# Target network parameter update factor, for double DQN
tau = 0.005
# Learning rate for actor-critic models
critic_lr = 0.001
aux_lr = 0.001

num_states = 5 #we reduce the state dim through observation (see below)
num_actions = 2 #acceleration and steering
print("State Space dim: {}, Action Space dim: {}".format(num_states,num_actions))

upper_bound = 1
lower_bound = -1
print("Min and Max Value of Action: {}".format(lower_bound,upper_bound))

buffer_dim = 50000
batch_size = 64

is_training = False
use_vae = False

# VAE hyperparameters
drop_out_prob = 0.1
original_dim = num_states + num_actions + 1 + 1 + num_states
intermediate_dim = 64
latent_dim = 4

vae_lr = 0.001

#pesi
# ddpg_critic_weigths_32_car0_split.h5 #versione con reti distinte per le mosse. Muove bene ma lento
# ddpg_critic_weigths_32_car1_split.h5 #usual problem: sembra ok

load_weights = False
save_weights = False #beware when saving weights to not overwrite previous data

#weights_file_actor = "weights/ddpg_actor_weigths_32_car3_split.h5"
#weights_file_critic = "weights/ddpg_critic_weigths_32_car3_split.h5"

weights_file_actor = "weights/ddpg_actor_model_car"
weights_file_critic = "weights/ddpg_critic_model_car"


#The actor choose the move, given the state
def get_actor():
    #no special initialization is required
    # Initialize weights between -3e-3 and 3-e3
    #last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

    inputs = layers.Input(shape=(num_states,))
    out = layers.Dense(64, activation="relu")(inputs)
    out = layers.Dense(64, activation="relu")(out)
    #outputs = layers.Dense(num_actions, kernel_regularizer=regularizers.l2(0.01), kernel_initializer=last_init)(out)
    #outputs = layers.Activation('tanh')(outputs)
    #outputs = layers.Dense(num_actions, name="out", activation="tanh", kernel_initializer=last_init)(out)
    outputs = layers.Dense(num_actions, name="out", activation="tanh")(out)

    #outputs = outputs * upper_bound
    model = tf.keras.Model(inputs, outputs, name="actor")
    return model

def get_actor(train_acceleration=True,train_direction=True):
    # the actor has separate towers for action and speed
    # in this way we can train them separately

    inputs = layers.Input(shape=(num_states,))
    out1 = layers.Dense(32, activation="relu", trainable=train_acceleration)(inputs)
    out1 = layers.Dense(32, activation="relu", trainable=train_acceleration)(out1)
    out1 = layers.Dense(1, activation='tanh', trainable=train_acceleration)(out1)

    out2 = layers.Dense(32, activation="relu", trainable=train_direction)(inputs)
    out2 = layers.Dense(32, activation="relu",trainable=train_direction)(out2)
    out2 = layers.Dense(1, activation='tanh',trainable=train_direction)(out2)

    outputs = layers.concatenate([out1,out2])

    #outputs = outputs * upper_bound #resize the range, if required
    model = tf.keras.Model(inputs, outputs, name="actor")
    return model

#the critic compute the q-value, given the state and the action
def get_critic():
    # State as input
    state_input = layers.Input(shape=(num_states))
    state_out = layers.Dense(16, activation="relu")(state_input)
    state_out = layers.Dense(32, activation="relu")(state_out)

    # Action as input
    action_input = layers.Input(shape=(num_actions))
    action_out = layers.Dense(32, activation="relu")(action_input)

    concat = layers.Concatenate()([state_out, action_out])

    out = layers.Dense(64, activation="relu")(concat)
    out = layers.Dense(64, activation="relu")(out)
    outputs = layers.Dense(1)(out) #Outputs single value

    model = tf.keras.Model([state_input, action_input], outputs, name="critic")

    return model

#Replay buffer
class Buffer:
    def __init__(self, buffer_capacity=100000, batch_size=64):
        # Max Number of tuples that can be stored
        self.buffer_capacity = buffer_capacity
        # Num of tuples used for training
        self.batch_size = batch_size

        # Current number of tuples in buffer
        self.buffer_counter = 0
        self.index = 0

        # We have a different array for each tuple element
        self.state_buffer = np.zeros((self.buffer_capacity, num_states))
        self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.done_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, num_states))

    # Stores a transition (s,a,r,s') in the buffer
    def record(self, obs_tuple):
        s,a,r,T,sn = obs_tuple
        # restart form zero if buffer_capacity is exceeded, replacing old records
        self.index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[self.index] = tf.squeeze(s)
        self.action_buffer[self.index] = a
        self.reward_buffer[self.index] = r
        self.done_buffer[self.index] = T
        self.next_state_buffer[self.index] = tf.squeeze(sn)

        self.buffer_counter += 1

    def sample_batch(self):
        # Get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)

        s = self.state_buffer[batch_indices]
        a = self.action_buffer[batch_indices]
        r = self.reward_buffer[batch_indices]
        T = self.done_buffer[batch_indices]
        sn = self.next_state_buffer[batch_indices]

        if use_vae:
            # before sampling from replay buffer, generate some pseudo-samples
            random_vector_for_generation = tf.random.normal(shape=[self.batch_size // 2, vae.latent_dim])

            pseudo_samples = vae.decoder(random_vector_for_generation)
            np.random.shuffle(pseudo_samples.numpy())

            s[:self.batch_size // 2] = pseudo_samples[:, :num_states]
            a[:self.batch_size // 2] = pseudo_samples[:, num_states:num_states + num_actions]
            r[:self.batch_size // 2] = np.reshape(pseudo_samples[:, num_states + num_actions],
                                                  (self.batch_size // 2, 1))
            T[:self.batch_size // 2] = np.reshape(pseudo_samples[:, -(num_states + 1)], (self.batch_size // 2, 1))
            sn[:self.batch_size // 2] = pseudo_samples[:, -num_states:]

        return ((s,a,r,T,sn))

# Slowly updating target parameters according to the tau rate <<1
@tf.function
def update_target(target_weights, weights, tau):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))

def update_weights(target_weights, weights, tau):
    return(target_weights * (1- tau) +  weights * tau)

def policy(state,verbose=False):
    #the policy used for training just add noise to the action
    #the amount of noise is kept constant during training
    sampled_action = tf.squeeze(actor_model(state))
    noise = np.random.normal(scale=0.1,size=2)
    #we may change the amount of noise for actions during training
    noise[0] *= 2
    noise[1] *= .5
    # Adding noise to action
    sampled_action = sampled_action.numpy()
    sampled_action += noise
    #in verbose mode, we may print information about selected actions
    if verbose and sampled_action[0] < 0:
        print("decelerating")

    #Finally, we ensure actions are within bounds
    legal_action = np.clip(sampled_action, lower_bound, upper_bound)

    return [np.squeeze(legal_action)]

#creating models
# actor_model = get_actor()
# critic_model = get_critic()
actor_model = tf.keras.models.load_model('C:\\Users\\gaspa\\PycharmProjects\\MicroRacer\\weights\\ddpg_actor_model_car')
critic_model = tf.keras.models.load_model('C:\\Users\\gaspa\\PycharmProjects\\MicroRacer\\weights\\ddpg_critic_model_car')
vae = VariationalAutoEncoder(original_dim, intermediate_dim, latent_dim, drop_out_prob)
#actor_model.summary()
#critic_model.summary()

#we create the target model for double learning (to prevent a moving target phenomenon)
target_actor = get_actor()
target_critic = get_critic()
target_actor.trainable = False
target_critic.trainable = False

#We compose actor and critic in a single model.
#The actor is trained by maximizing the future expected reward, estimated
#by the critic. The critic should be freezed while training the actor.
#For simplicitly, we just use the target critic, that is not trainable.

def compose(actor,critic):
    state_input = layers.Input(shape=(num_states))
    a = actor(state_input)
    q = critic([state_input,a])
    #reg_weights = actor.get_layer('out').get_weights()[0]
    #print(tf.reduce_sum(0.01 * tf.square(reg_weights)))

    m = tf.keras.Model(state_input, q)
    #the loss function of the compound model is just the opposite of the critic output
    m.add_loss(-q)
    return(m)

aux_model = compose(actor_model,target_critic)

## TRAINING ##
if load_weights:
    critic_model = keras.models.load_model(weights_file_critic)
    actor_model = keras.models.load_model(weights_file_actor)

# Making the weights equal initially
target_actor_weights = actor_model.get_weights()
target_critic_weights = critic_model.get_weights()
target_actor.set_weights(target_actor_weights)
target_critic.set_weights(target_critic_weights)


critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
aux_optimizer = tf.keras.optimizers.Adam(aux_lr)
vae_optimizer = tf.keras.optimizers.Adam(vae_lr)

critic_model.compile(loss='mse',optimizer=critic_optimizer)
aux_model.compile(optimizer=aux_optimizer)
vae.compile(loss='mse', optimizer=vae_optimizer)

buffer = Buffer(buffer_dim, batch_size)

# History of rewards per episode
ep_reward_list = []
# Average reward history of last few episodes
avg_reward_list = []         

# We introduce a probability of doing n empty actions to separate the environment time-step from the agent
def step(action):
    n = 1
    t = np.random.randint(0,n)
    state ,reward,done = racer.step(action)
    for i in range(t):
        if not done:
            state ,t_r, done = racer.step([0, 0])
            #state ,t_r, done =racer.step(action)
            reward+=t_r
    return (state, reward, done)


def train(total_episodes=total_episodes):
    i = 0
    mean_speed = 0

    for ep in range(total_episodes):

        prev_state = racer.reset()
        episodic_reward = 0
        mean_speed += prev_state[num_states-1]
        done = False
        while not(done):
            i = i+1
            tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)
            #our policy is always noisy
            action = policy(tf_prev_state)[0]
            # Get state and reward from the environment
            state, reward, done = step(action)
            
            #we distinguish between termination with failure (state = None) and succesfull termination on track completion
            #succesfull termination is stored as a normal tuple
            fail = done and len(state)<num_states 
            buffer.record((prev_state, action, reward, fail, state))
            if not(done):
                mean_speed += state[num_states-1]
        
            episodic_reward += reward

            if buffer.buffer_counter>batch_size:
                states,actions,rewards,dones,newstates= buffer.sample_batch()
                targetQ = rewards + (1-dones)*gamma*(target_critic([newstates,target_actor(newstates)]))
                loss1 = critic_model.train_on_batch([states,actions],targetQ)
                loss2 = aux_model.train_on_batch(states)

                update_target(target_actor.variables, actor_model.variables, tau)
                update_target(target_critic.variables, critic_model.variables, tau)
            prev_state = state

        if len(ep_reward_list) > 0:
            avg_reward = np.mean(ep_reward_list[-40:])
        else:
            avg_reward = -3

        if episodic_reward > avg_reward and buffer.index > 2 * buffer.batch_size and buffer.index > 0 and use_vae:
            state_data = buffer.state_buffer[:buffer.index]
            action_data = buffer.action_buffer[:buffer.index]
            reward_data = buffer.reward_buffer[:buffer.index]
            done_data = buffer.done_buffer[:buffer.index]
            next_state_data = buffer.next_state_buffer[:buffer.index]

            real_samples = np.column_stack((state_data, action_data, reward_data, done_data, next_state_data))

            # generate batch_size pseudo-states with VAE and then mix them wth real states
            # we may change sampling from latent space (e.g. Gibbs sampling)
            random_vector_for_generation = tf.random.normal(shape=[buffer.index, vae.latent_dim])

            pseudo_samples = vae.decoder(random_vector_for_generation)
            samples = np.concatenate((real_samples, pseudo_samples.numpy()), axis=0)
            np.random.shuffle(samples)

            # drop remainder
            drop_indexes = samples.shape[0] % (2 * buffer.batch_size)
            samples = samples[: samples.shape[0] - drop_indexes, :]

            # shuffle and convert to tensor
            samples = tf.convert_to_tensor(samples)

            callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
            vae.fit(x=samples,
                    y=samples,
                    epochs=500,
                    batch_size=buffer.batch_size * 2,
                    shuffle=True,
                    validation_split=0.25,
                    callbacks=[callback])

        ep_reward_list.append(episodic_reward)

        # Mean of last 40 episodes
        avg_reward = np.mean(ep_reward_list[-40:])
        print("Episode {}: Avg. Reward = {}, Last reward = {}. Avg. speed = {}".format(ep, avg_reward,episodic_reward,mean_speed/i))
        print("\n")

        avg_reward_list.append(avg_reward)
        
        if ep>0 and ep%100 == 0:
            print("## Evaluating policy ##")
            tracks.metrics_run(actor_model, 10)
        

    if total_episodes > 0:
        if save_weights:
            critic_model.save(weights_file_critic)
            actor_model.save(weights_file_actor)
        # Plotting Episodes versus Avg. Rewards
        plt.plot(avg_reward_list)
        plt.xlabel("Episode")
        plt.ylabel("Avg. Episodic Reward")
        plt.show()

if is_training:
    train()

tracks.newrun([actor_model])


