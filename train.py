import tensorflow as tf
import tensorflow_addons as tfa
# tf.config.run_functions_eagerly(True)  # For debugging

import tensorflow_datasets as tfds

import hydra
from hydra.utils import get_original_cwd
from omegaconf import open_dict, OmegaConf

import os
import logging

import optuna

import src

import matplotlib.pyplot as plt

train_logger = logging.getLogger('Train')


def train(config):
    tf.keras.backend.clear_session()

    print('Current HyperParameters:')
    for key, value in config.hparams.items():
        print(f'{key}: {value}')

    # Load dataset
    train_ds, valid_ds, test_ds = src.load_data(config.hparams.batch_size)

    # Create Model and Optimizer
    model = src.ViT(config.hparams)
    total_steps = train_ds.cardinality().numpy() * config.hparams.epochs
    warmup_steps = int(total_steps * 0.1)
    scheduler = src.WarmupCosineSchedule(
        learning_rate=config.hparams.learning_rate, warmup_steps=warmup_steps, total_steps=total_steps,
        alpha=0.0
    )
    optimizer = tfa.optimizers.AdaBelief(learning_rate=scheduler)
    model.compile(
        optimizer=optimizer, loss='sparse_categorical_crossentropy',
        metrics=['sparse_categorical_accuracy', src.SparseF1Score(num_classes=1000, average='macro')]
    )

    # Create callbacks and run training
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_f1_score', patience=10, restore_best_weights=True, mode='max'
        ),
        tf.keras.callbacks.TensorBoard('./logs')
    ]
    model.fit(
        train_ds, validation_data=valid_ds, epochs=config.hparams.epochs, callbacks=callbacks
    )
    tf.keras.models.save_model(model, 'vit.tf', save_format='tf')

    # Evaluate
    loss = model.evaluate(test_ds)

    if config.hpo:
        train_logger.info(f'Job Number: {config.job_num}')
    for metric, value in zip(model.metrics_names, loss):
        train_logger.info(f'{metric}: {value}')
        if metric == 'f1_score':
            final_metric = value

    with open_dict(config):
        config.final_metric = final_metric
    OmegaConf.save(
        OmegaConf.create(config), './config.yaml'
    )
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
                'n_filters', 128, 512, step=128
            )
            main_config.hparams.norm = trial.suggest_categorical(
                'norm', [None, 'pre', 'post', 'dual']
            )
            main_config.hparams.n_heads = 2 ** trial.suggest_int(
                'n_heads', 1, 3
            )
            main_config.hparams.n_blocks = trial.suggest_int(
                'n_blocks', 4, 12
            )
            main_config.hparams.drop_rate = trial.suggest_float(
                'drop_rate', 0.0, 0.2
            )
            with open_dict(main_config):
                main_config.job_num = trial.number

            # Create a directory to save the results
            os.chdir(working_dir)
            os.makedirs(f'./{trial.number}')
            os.chdir(f'./{trial.number}')

            # Train
            final_metric = train(main_config)
            return final_metric

        # Manually define the search space to use GridSampler
        # The reason why I use GridSampler is that I want to know how normalization affects the performance
        # not to find the best hyperparameters.
        # If you want the best hyperparameters, use TPE sampler.
        search_space = {
            'patch_size': [4, 8, 16],
            'n_filters': [32, 64, 96, 128],
            'norm': [None, 'pre', 'post', 'dual'],
            'n_heads': [1, 2, 3],  # 2, 4, 8
            'n_blocks': [i for i in range(4, 13, 2)],
            'drop_rate': [i / 10 for i in range(0, 2)],
        }
        study_name = 'study'
        study = optuna.create_study(
            study_name=study_name,
            storage=f'sqlite:///{study_name}.db',
            load_if_exists=True,
            direction='maximize',
            sampler=optuna.samplers.GridSampler(search_space)
        )
        study.optimize(objective)

        # TODO: Implement followings
        # Get HPO Logs as DataFrame

        # Group by norm and plot the mean and std of f1_score

        # Save Results

    else:
        train(main_config)

    OmegaConf.save(
        OmegaConf.create(main_config), './config.yaml'
    )


if __name__ == "__main__":
    main()
