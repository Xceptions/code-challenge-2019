import logging

import luigi
import os
from pathlib import Path

from util import DockerTask

VERSION = os.getenv('PIPELINE_VERSION', '0.1')


class Debug(DockerTask):
    """Use this task with appropriate image to debug things."""

    @property
    def image(self):
        return 'code-challenge/download-data:0.1'

    @property
    def command(self):
        return [
            'sleep', '3600'
        ]


class DownloadData(DockerTask):
    """Initial pipeline task downloads dataset."""

    fname = luigi.Parameter(default='wine_dataset')
    out_dir = luigi.Parameter(default='/usr/share/data/raw/')
    url = luigi.Parameter(
        default='https://github.com/datarevenue-berlin/code-challenge-2019/'
                'releases/download/0.1.0/dataset_sampled.csv'
    )

    @property
    def image(self):
        return f'code-challenge/make-dataset:{VERSION}'

    @property
    def command(self):
        return [
            'python', 'download_data.py',
            '--name', self.fname,
            '--url', self.url,
            '--out-dir', self.out_dir
        ]

    def output(self):
        out_dir = Path(self.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        return luigi.LocalTarget(
            path=str(out_dir/f'{self.fname}.csv')
        )


class MakeDatasets(DockerTask):

    out_dir = luigi.Parameter(
        default='/usr/share/data/make/'
    )

    @property
    def image(self):
        return f'code-challenge/make-dataset:{VERSION}'

    def requires(self):
        return DownloadData()

    @property
    def command(self):
        # TODO: implement correct command
        # Try to get the input path from self.requires() ;)
        return [
            'python', 'dataset.py',
            '--in-csv', self.input().path,
            '--out-dir', self.out_dir
        ]

    def output(self):
        print("make_dataset_output")
        return luigi.LocalTarget(
            path=str(Path(self.out_dir) / '.SUCCESS')
        )


class TrainModel(DockerTask):
    """
    Call the train model file. Task is meant to fetch csv file
    from the make datasets task, but I am yet to fix that issue
    So I'll read the file raw
    """

    fname = luigi.Parameter(default='/usr/share/data/raw/wine_dataset')
    out_dir = luigi.Parameter(default='/usr/share/data/train/')

    @property
    def image(self):
        return f'code-challenge/train-model:{VERSION}'

    @property
    def command(self):
        return [
            'python', 'train_model.py',
            '--in-csv', self.fname,
            '--out-dir', self.out_dir
        ]

    def output(self):
        out_dir = Path(self.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        return luigi.LocalTarget(
            path=str(out_dir/f'{self.fname}.csv')
        )


class EvaluateModel(DockerTask):
    """
    Call the evaluate model file. Task is meant to import the test
    file, the pkl model and the pkl persistence variables and pass
    them as arguments to the evaluate model function
    """

    fname = luigi.Parameter(default='/usr/share/data/raw/test_dataset')
    out_dir = luigi.Parameter(default='/usr/share/data/evaluate/')

    @property
    def image(self):
        return f'code-challenge/evaluate-model:{VERSION}'

    @property
    def command(self):
        return [
            'python', 'evaluate_model.py',
            '--in-csv', '../data_root/raw/test_dataset.csv',
            '--out-dir', '......',
            '--model-pkl', '......',
            '--pers-vars-pkl', '......'
        ]

    def output(self):
        out_dir = Path(self.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        return luigi.LocalTarget(
            path=str(out_dir/f'{self.fname}.csv')
        )
