import os, sys, errno, argparse, json, time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from dotmap import DotMap
from cv_trainer import CVTrainer


def get_onehot_array(index, num_classes):
    arr = np.zeros(num_classes)
    arr[index] = 1
    return arr


def generate_csv(config):
    data_dir = config.data.path
    groups_arg = config.data.groups
    num_classes = config.data.num_classes

    count = 0

    names = []
    groups = []
    indexes = []
    labels = []
    vectors = []

    for root, _, filenames in os.walk(data_dir):
        for filename in filenames:
            count += 1
            split_path = root.split(os.sep)
            # print(split_path)

            names.append(filename)
            groups.append(split_path[-1])
            indexes.append(split_path[-2])
            # labels.append(get_label_index(split_path[-2]))
            label = config.data.classes[split_path[-2]]
            labels.append(label)
            vectors.append(get_onehot_array(label, num_classes))
    print('{} files'.format(count))
    
    data = {'Filename': names, 'Stain': groups, 'Index': indexes, 'Label': labels, 'Vector': vectors}
    df = pd.DataFrame(data, columns=data.keys())
    
    # group filter
    groups_arg = groups_arg.upper()
    if groups != 'all':
        groups_list = groups_arg.split(",")
        print(groups_list)
        df = df[df['Stain'].isin(groups_list)]
    
    df.to_csv('labels.csv', index=False)
    print('\nlabels.csv saved!')


def generate_cv_csv(csv_path, exp_dir, k=10):
    create_k_folds(exp_dir, k)

    df = pd.read_csv(csv_path, index_col=False)
    names = df.iloc[:, 1]
    labels = df.iloc[:, 4]

    # define test set (10%)
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=123)
    # get first split and consider val as test
    train_indices, test_indices = next(skf.split(names, labels))
    trainCV_df = df.iloc[train_indices]
    trainCV_df.reset_index(drop=True, inplace=True)
    test_df = df.iloc[test_indices]
    test_df.reset_index(drop=True, inplace=True)
    trainCV_df.to_csv(os.path.join(exp_dir, 'train_CV.csv'), index=False)
    test_df.to_csv(os.path.join(exp_dir, 'test.csv'), index=False)

    # overwrite names and labels for cv split
    names = trainCV_df.iloc[:, 1]
    labels = trainCV_df.iloc[:, 4]

    # cv split and train-val CSVs
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=234)
    for index, (train_indices, val_indices) in enumerate(skf.split(names, labels)):
        print('Creating fold {}'.format(str(index+1)))
        train_df = trainCV_df.iloc[train_indices]
        val_df = trainCV_df.iloc[val_indices]
        fold_path = os.path.join(exp_dir, "Fold" + str(index + 1))

        train_df.to_csv(os.path.join(fold_path, 'train.csv'), index=False)
        val_df.to_csv(os.path.join(fold_path, 'val.csv'), index=False)

    return 0


def create_k_folds(exp_dir, k=10):
    for i in range(1, k + 1):
        # print((exp_dir, type(exp_dir)))
        base_path = os.path.join(exp_dir, "Fold" + str(i))
        # Create folder if does not exist
        try:
            os.makedirs(base_path)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('--data', type=str, required=True)
    argparser.add_argument('--classes', type=str, required=True)
    argparser.add_argument('--config', type=str, required=True)
    argparser.add_argument('--groups', type=str, default='all')
    args = argparser.parse_args()
    return args


def get_dict_from_json(json_file):
    """
    Get the config from a json file
    :param json_file:
    :return: config(namespace) or config(dictionary)
    """
    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)

    return config_dict


def process_config(args):
    config = get_dict_from_json(args.config)
    config_data = get_dict_from_json(args.classes)
    config.update(config_data)
    config = DotMap(config)

    config.data.path = args.data
    config.data.groups = args.groups
    config.exp.dir = os.path.join("experiments",
                                  config.model.name,
                                  args.groups + "_" + str(config.dataloader.image_size) + '_' + config.dataloader.mode,
                                  time.strftime("%d-%m-%Y__%H-%M-%S/", time.localtime()))
    return config


def create_dir(dir):
    """
    dirs - a list of directories to create if these directories are not found
    :param dirs:
    :return exit_code: 0:success -1:failed
    """
    try:
        if not os.path.exists(dir):
            os.makedirs(dir)
        return 0
    except Exception as err:
        print("Creating directories error: {0}".format(err))
        exit(-1)


def dict_to_file(d, path, name):
    full = os.path.join(path, name)
    try:
        with open(full, 'w') as fp:
            json.dump(d, fp, indent=2)
    except Exception as err:
        print("Creating dict file error: {0}".format(err))
        exit(-1)


def get_trainer_by_name(config):
    # output = config.trainer.name.split(".")
    # trainerClass = getattr(importlib.import_module("trainers." + output[0]), output[1])
    # trainer = trainerClass(config)
    if config.trainer.name == 'cv':
        trainer = CVTrainer(config)
    else:
        print('Not implemmented trainer')
        sys.exit(1)
        
    return trainer