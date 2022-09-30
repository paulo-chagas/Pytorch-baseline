
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    from utils import get_args, process_config, create_dir, dict_to_file, get_trainer_by_name, generate_csv, generate_cv_csv


def main():
    try:
        args = get_args()
        config = process_config(args)
    except Exception as err:
        print("missing or invalid arguments: {0}".format(err))
        exit(0)

    # create the experiments dirs
    create_dir(config.exp.dir)
    generate_csv(config)
    generate_cv_csv('labels.csv', config.exp.dir, config.trainer.K)
    dict_to_file(config.toDict(), config.exp.dir, "conf.json")

    # create the trainer
    trainer = get_trainer_by_name(config)
    print(trainer.__class__)
    trainer.train()


if __name__ == '__main__':
    main()