import os

import numpy as np

from dm_control import suite



def make_obs(observation):
    position_npy = np.atleast_1d(observation['position'])
    velocity_npy = np.atleast_1d(observation['velocity'])
    obs_npy = np.concatenate([position_npy, velocity_npy])
    return obs_npy.copy()


def save_data(directory, t, history):
    paths = []
    history_length = len(history)
    for idx, data_dict in enumerate(history):
        diff = history_length - (idx + 1)
        time_step = t - diff
        time_step = str(time_step).zfill(6)
        action_path = os.path.join(directory, f'step_{time_step}_action.npy')
        paths.append(action_path)
        np.save(action_path, data_dict['action'])

        image_path = os.path.join(directory, f'step_{time_step}_image.npy')
        paths.append(image_path)
        np.save(image_path, data_dict['image'])

        state_path = os.path.join(directory, f'step_{time_step}_state.npy')
        paths.append(state_path)
        np.save(state_path, data_dict['state'])

    with open(f'{directory}/data.txt', 'a') as f:
        memo = ','.join(paths)
        f.write(f'{memo}\n')


def set_env(init_state=[0.15, 0.15, 0.0, 0.0]):
    env = suite.load(domain_name="point_mass", task_name="easy")
    input_state = np.array(init_state, dtype=np.float64)
    env.physics.set_state(input_state)
    env.physics.forward()
    return env


if __name__ == "__main__":
    episode_num = 0
    save_directory = 'data'
    horizon_num = 20
    get_episode_num = 2000
    imsize = 224
    while episode_num < get_episode_num:
        episode = str(episode_num).zfill(6)
        history = []
        initial_state_x = np.random.uniform(-0.25, 0.25)
        initial_state_y = np.random.uniform(-0.25, 0.25)
        env = set_env(init_state=[initial_state_x, initial_state_y, 0.0, 0.0])
        action_spec = env.action_spec()
        # get obs
        observation = env._task.get_observation(env._physics)
        state = make_obs(observation)
        image = env.physics.render(imsize, imsize, camera_id=0)
        action = np.random.uniform(
            action_spec.minimum,
            action_spec.maximum,
            size=action_spec.shape)
        t = 0
        history.append({'state': state, 'action': action, 'image': image})

        while True:
            t += 1
            time_step = env.step(action)
            next_state = make_obs(time_step.observation)
            next_action = np.random.uniform(
                    action_spec.minimum,
                    action_spec.maximum,
                    size=action_spec.shape)
            next_image = env.physics.render(imsize, imsize, camera_id=0)
            history.append({'state': next_state, 'action': next_action, 'image': next_image})
            action = next_action
            if len(history) == horizon_num:
                directory = f'{save_directory}/episode_{episode}'
                if not os.path.exists(directory):
                    os.makedirs(directory)
                save_data(directory, t, history)
                episode_num += 1
                # for RSSM
                break
            else:
                break