import ast
import os

import pandas as pd


def read_csv(csv):
    """
    This method reads the trial data recorded in the csv and returns a pd.DataFrame with additional features.
    The code has been taken from
    """
    data = pd.read_csv(os.path.join("logs", csv))
    data['average_reward'] = (data['net_reward'] / (data['initial_deadline'] - data['final_deadline'])).rolling(
        window=10, center=False).mean()
    data['reliability_rate'] = (data['success'] * 100).rolling(window=10,
                                                               center=False).mean()  # compute avg. net reward with window=10
    data['good_actions'] = data['actions'].apply(lambda x: ast.literal_eval(x)[0])
    data['good'] = (data['good_actions'] * 1.0 /
                    (data['initial_deadline'] - data['final_deadline'])).rolling(window=10, center=False).mean()
    data['minor'] = (data['actions'].apply(lambda x: ast.literal_eval(x)[1]) * 1.0 /
                     (data['initial_deadline'] - data['final_deadline'])).rolling(window=10, center=False).mean()
    data['major'] = (data['actions'].apply(lambda x: ast.literal_eval(x)[2]) * 1.0 /
                     (data['initial_deadline'] - data['final_deadline'])).rolling(window=10, center=False).mean()
    data['minor_acc'] = (data['actions'].apply(lambda x: ast.literal_eval(x)[3]) * 1.0 /
                         (data['initial_deadline'] - data['final_deadline'])).rolling(window=10, center=False).mean()
    data['major_acc'] = (data['actions'].apply(lambda x: ast.literal_eval(x)[4]) * 1.0 /
                         (data['initial_deadline'] - data['final_deadline'])).rolling(window=10, center=False).mean()
    data['epsilon'] = data['parameters'].apply(lambda x: ast.literal_eval(x)['e'])
    data['alpha'] = data['parameters'].apply(lambda x: ast.literal_eval(x)['a'])
    return data


def read_states_from_text(text_file_name):
    # type: (str) -> pd.DataFrame
    """
    This method opens the given text file,
    reads the recorded Q-learning data,
    uses it to create a DataFrame.
    It also adds additional features to the data in the DataFrame
    such as what the chosen policy is and whether it is optimal

    :return: df, a DataFrame of the recorded trial data
    :param: text_file_name

    """
    import re
    file = open(os.path.join('logs', text_file_name))
    lines = file.read().splitlines()
    states = {
        'light': [],
        'oncoming': [],
        'left': [],
        'Q_forward': [],
        'Q_left': [],
        'Q_right': [],
        'Q_None': [],
        'waypoint': []
    }
    for line in lines:
        if line.startswith('('):
            line = re.sub('[\',()]', '', line)
            state = line.split(' ')
            states['waypoint'].append(state[0])
            states['light'].append(state[1])
            states['oncoming'].append(state[2])
            states['left'].append(state[3])
        elif line.startswith(' -- '):
            line = re.sub(' -- ', '', line)
            line = re.sub(' : ', ' ', line)
            state = line.split(' ')
            states['Q_' + state[0]].append(state[1])

    df = pd.DataFrame.from_dict(states)
    df['Q_max'] = df[['Q_forward', 'Q_left', 'Q_right', 'Q_None']].max(axis=1)
    df['policy'] = df[['Q_forward', 'Q_left', 'Q_right', 'Q_None']].idxmax(axis=1)
    df.policy = df.policy.str[2:]
    df['is_optimal'] = df.apply(lambda row: optimal_policy(row), axis=1)
    df['follows_waypoint'] = df['waypoint'].str.lower() == df['policy'].str.lower()
    df = df[['light', 'oncoming', 'left', 'waypoint', 'Q_forward', 'Q_left', 'Q_right', 'Q_None', 'Q_max', 'policy',
             'is_optimal', 'follows_waypoint']]
    return df


def optimal_policy(state):
    """
    This method defines the optimal policy,
    it returns a True / False on whether the chosen action is optimal

    :rtype: Boolean

    :param: state: the state to test for.
    :return: True if the policy for the state is optimal, False otherwise
    """
    if state['light'] == 'green':
        if state['oncoming'] in 'forward right':
            if state['waypoint'] in 'forward right':
                return state['waypoint'] == state['policy']
            else:
                return state['policy'] in 'forward right'
        else:
            if state['waypoint'] in 'forward left right':
                return state['waypoint'] == state['policy']
            else:
                return state['policy'] in 'forward left right'
    else:
        if state['left'] != 'forward':
            if state['waypoint'] in 'None right':
                return state['waypoint'] == state['policy']
            else:
                return state['policy'] in 'None right'
    return state['policy'] in 'None'
