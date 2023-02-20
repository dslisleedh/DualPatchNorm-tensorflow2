import tensorflow as tf
import tensorflow_addons as tfa

import tensorflow_datasets as tfds

import hydra
from hydra.utils import get_original_cwd
from omegaconf import open_dict, OmegaConf

import os
import logging

import optuna

import src

train_logger = logging.getLogger('Train')


def train(config):
    tf.keras.backend.clear_session()
    # Load dataset
    train_ds, valid_ds, test_ds = src.load_data(config.hparams.batch_size)

    # Create Model and Optimizer
    model = src.ViT(config.hparams)
    optimizer = tfa.optimizers.AdaBelief(learning_rate=config.hparams.learning_rate)
    model.compile(
        optimizer=optimizer, loss='sparse_categorical_crossentropy',
        metrics=['sparse_categorical_accuracy', src.SparseF1Score(num_classes=10, average='macro')]
    )

    # Create callbacks and run training
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_f1_score', patience=10, restore_best_weights=True, mode='max'
        ),
        tf.keras.callbacks.TensorBoard('./logs')
    ]
    model.fit(
        train_ds, validation_data=valid_ds, epochs=config.hparams.epochs,
        callbacks=callbacks
    )

    # Evaluate
    loss = model.evaluate(test_ds)

    if config.hpo:
        train_logger.info(f'Job Number: {config.job_num}')
    for metric, value in zip(model.metrics_names, loss):
        train_logger.info(f'{metric}: {value}')
        if metric == 'f1_score':
            final_metric = value
    return final_metric


@hydra.main(config_path='config', config_name='config', version_base=None)
def main(main_config):
    if main_config.hpo:
        working_dir = os.getcwd()

        def objective(trial):
            # Define search space
            patch_size = trial.suggest_categorical(
                'patch_size', [4, 8, 16]
            )
            main_config.hparams.patch_size = (patch_size, patch_size)
            main_config.hparams.n_filters = trial.suggest_int(
                'n_filters', 32, 128, 32
            )
            main_config.hparams.norm = trial.suggest_categorical(
                'norm', [None, 'pre', 'post', 'dual']
            )
            main_config.hparams.n_heads = 2 ** trial.suggest_int(
                'n_heads', 0, 3
            )
            main_config.hparams.n_blocks = trial.suggest_int(
                'n_blocks', 4, 12
            )
            main_config.hparams.drop_rate = trial.suggest_float(
                'drop_rate', 0.0, 0.5
            )
            main_config.hparams.batch_size = 2 ** trial.suggest_int(
                'batch_size', 6, 9
            )
            with open_dict(main_config):
                main_config.hparams.job_num = trial.number

            # Create a directory to save the results
            os.chdir(working_dir)
            os.makedirs(f'./{trial.number}')
            os.chdir(f'./{trial.number}')

            # Train
            final_metric = train(main_config)
            return final_metric

        # Manually define the search space to use GridSampler
        search_space = {
            'patch_size': [4, 8, 16],
            'n_filters': [32, 64, 96, 128],
            'norm': [None, 'pre', 'post', 'dual'],
            'n_heads': [0, 1, 2, 3],
            'n_blocks': [i for i in range(4, 13, 2)],
            'drop_rate': [i / 10 for i in range(0, 6)],
            'batch_size': [i for i in range(6, 10)]
        }
        study_name = 'study.db'
        study = optuna.create_study(
            study_name=study_name,
            storage=f'sqlite:///{study_name}',
            load_if_exists=True,
            direction='maximize',
            sampler=optuna.samplers.GridSampler(search_space)
        )
        study.optimize(objective)
        print(study.best_params)
        with open_dict(main_config):
            main_config.best_trial = study.best_trial
            main_config.best_final_metric = study.best_trial.value

    else:
        final_metric = train(main_config)
        with open_dict(main_config):
            main_config.final_metric = final_metric

    OmegaConf.save(
        OmegaConf.create(main_config), './config.yaml'
    )


if __name__ == "__main__":
    main()
