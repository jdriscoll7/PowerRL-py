import json

from matplotlib import pyplot as plt


def plot_training_curve(absolute_path: str):
    with open(f"{absolute_path}/result.json") as f:
        json_list = [json.loads(line) for line in f]

    episode_lengths = [inner for data in json_list for inner in data['hist_stats']['episode_lengths']]
    episode_rewards = [inner for data in json_list for inner in data['hist_stats']['episode_reward']]

    plt.subplots(2, 1, sharex=True)

    plt.subplot(2, 1, 1)
    plt.plot(episode_lengths)
    plt.xlabel('episode')
    plt.ylabel('episode length')

    plt.subplot(2, 1, 2)
    plt.plot(episode_rewards)
    plt.xlabel('episode')
    plt.ylabel('episode reward')


