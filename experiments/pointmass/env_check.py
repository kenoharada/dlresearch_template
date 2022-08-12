import matplotlib.pyplot as plt
import numpy as np
from dm_control import suite


def visualize(t_length=10):
    time_step = env.reset()
    for t in range(t_length):
        action = np.random.uniform(
            action_spec.minimum,
            action_spec.maximum,
            size=action_spec.shape)
        time_step = env.step(action)
        frame = env.physics.render(224, 224, camera_id=0)
        plt.imshow(frame)
        plt.pause(0.01)
        print(time_step.reward, time_step.discount, time_step.observation)


def set_env(init_state=[0.25, 0.25, 0.0, 0.0]):
    env = suite.load(domain_name="point_mass", task_name="easy")
    _ = env.reset()
    input_state = np.array(init_state, dtype=np.float64)
    env.physics.set_state(input_state)
    env.physics.forward()
    return env


def check_limit(max=True, t_length=1000):
    pos_limit = 0
    vel_limit = 0
    if max:
        env = set_env([-0.25, -0.25, 0.0, 0.0])
        action = np.array([1.0, 1.0])
    else:
        env = set_env([0.25, 0.25, 0.0, 0.0])
        action = np.array([-1.0, -1.0])
    for t in range(t_length):
        time_step = env.step(action)
        if max:
            if pos_limit < time_step.observation['position'][0]:
                pos_limit = time_step.observation['position'][0]
            if vel_limit < time_step.observation['velocity'][0]:
                vel_limit = time_step.observation['velocity'][0]
        else:
            if pos_limit > time_step.observation['position'][0]:
                pos_limit = time_step.observation['position'][0]
            if vel_limit > time_step.observation['velocity'][0]:
                vel_limit = time_step.observation['velocity'][0]
    last_frame = env.physics.render(224, 224, camera_id=0)
    return pos_limit, vel_limit, time_step.observation, last_frame


if __name__ == '__main__':
    env = suite.load(domain_name="point_mass", task_name="easy")
    action_spec = env.action_spec()
    observation_spec = env.observation_spec()
    print('obvervation_spec: ', observation_spec)
    print('action_spec: ', action_spec)
    # action_spec:  BoundedArray(shape=(2,), dtype=dtype('float64'), name=None, minimum=[-1. -1.], maximum=[1. 1.])
    time_step = env.reset()
    print('time_step: ', time_step)
    visualize()

    max_pos_limit, max_vel_limit, last_observation, last_frame = check_limit(max=True, t_length=20)
    print('max_pos_limit: ', max_pos_limit)
    print('max_vel_limit: ', max_vel_limit)
    print('last_observation: ', last_observation)
    plt.imshow(last_frame)
    plt.pause(3)

    min_pos_limit, min_vel_limit, last_observation, last_frame = check_limit(max=False, t_length=20)
    print('min_pos_limit: ', min_pos_limit)
    print('min_vel_limit: ', min_vel_limit)
    print('last_observation: ', last_observation)
    plt.imshow(last_frame)
    plt.pause(3)