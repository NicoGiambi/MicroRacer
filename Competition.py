import os
import tensorflow as tf
import tracks


if __name__ == '__main__':
    parent_dir = './weights' + os.sep
    car = '_actor_model_car/'

    ddpg2 = tf.keras.models.load_model(parent_dir + 'ddpg2' + car)
    print(ddpg2.summary())
    dsac = tf.keras.models.load_model(parent_dir + 'dsac' + car)
    print(dsac.summary())
    ppo = tf.keras.models.load_model(parent_dir + 'ppo' + car)
    print(ppo.summary())
    sac = tf.keras.models.load_model(parent_dir + 'sac' + car)
    print(sac.summary())
    exit()
    actors = [ddpg2, dsac, ppo, sac]

    tracks.newrun(actors)

