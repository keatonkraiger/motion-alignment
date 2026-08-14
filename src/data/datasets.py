import numpy as np
from scipy.io import loadmat

ALL_TAKES = [
    (1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (3, 2), (4, 1), (4, 2), (5, 1), (5, 2),
    (6, 1), (6, 2), (7, 1), (7, 2), (8, 1), (8, 2), (9, 2), (9, 3), (10, 1), (10, 2)
]

def load_take(data_dir, subject, take, wrapper_fn=np.array):
    data = loadmat(data_dir / f'patches_{subject}_{take}.mat')
    a = wrapper_fn(data['A'].transpose((3, 2, 0, 1))) # (N, 1, S, F)
    b = wrapper_fn(data['B'].transpose((3, 2, 0, 1))) # (N, 1, S, F)
    t0 = wrapper_fn(data['t0'].T) # (N, 1)
    return a, b, t0

def load_subject(data_dir, subject, wrapper_fn=np.array):
    a_list = []
    b_list = []
    t0_list = []
    for _subject, take in ALL_TAKES:
        if _subject != subject:
            continue
        a, b, t0 = load_take(data_dir, subject, take)
        a_list.append(a)
        b_list.append(b)
        t0_list.append(t0)
    a_list = wrapper_fn(np.vstack(a_list))
    b_list = wrapper_fn(np.vstack(b_list))
    t0_list = wrapper_fn(np.vstack(t0_list))
    return a_list, b_list, t0_list

def load_all_takes(data_dir, wrapper_fn=np.array, stack=True):
    a_list = []
    b_list = []
    t0_list = []
    for subject, take in ALL_TAKES:
        a, b, t0 = load_take(data_dir, subject, take, wrapper_fn=wrapper_fn)
        a_list.append(a)
        b_list.append(b)
        t0_list.append(t0)
    if stack:
        a_list = wrapper_fn(np.vstack(a_list))
        b_list = wrapper_fn(np.vstack(b_list))
        t0_list = wrapper_fn(np.vstack(t0_list))
    return a_list, b_list, t0_list

def load_all_takes_except(data_dir, exclude_subjects, wrapper_fn=np.array, stack=True):
    if not isinstance(exclude_subjects, list):
        exclude_subjects = [exclude_subjects]
    a_list = []
    b_list = []
    t0_list = []
    for subject, take in ALL_TAKES:
        if subject in exclude_subjects:
            continue
        a, b, t0 = load_take(data_dir, subject, take, wrapper_fn=wrapper_fn)
        a_list.append(a)
        b_list.append(b)
        t0_list.append(t0)
    if stack:
        a_list = wrapper_fn(np.vstack(a_list))
        b_list = wrapper_fn(np.vstack(b_list))
        t0_list = wrapper_fn(np.vstack(t0_list))
    return a_list, b_list, t0_list
