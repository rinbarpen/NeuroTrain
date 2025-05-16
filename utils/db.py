import os
import os.path
import numpy as np
from sqlmodel import SQLModel, Session, select, Field, create_engine, delete
from typing import Literal
from utils.typed import FilePath, FLOAT
from pathlib import Path

ScoreLiteral = Literal['all', 'mean', 'std', 'mean|std']

class Score(SQLModel, table=True):
    __tablename__: str = 'scores'

    id: int = Field(primary_key=True, default=None)
    metric_label: str = Field()
    class_label: str = Field()
    score: float = Field()

class ScoreDB:
    def __init__(self, filepath: FilePath):
        self.sqlite_filepath = Path(filepath)
        self.sqlite_url = f'sqlite:///{self.sqlite_filepath.absolute()}'
        self.engine = create_engine(self.sqlite_url)
        if not self.sqlite_filepath.exists():
            SQLModel.metadata.create_all(self.engine)
        else:
            self.clear()

    def add(self, metric_label: str, class_label: str, scores: list[FLOAT]|FLOAT):
        if isinstance(scores, FLOAT):
            scores = [scores]
        with Session(self.engine) as session:
            x = [Score(metric_label=metric_label, class_label=class_label, score=float(score)) for score in scores]
            session.add_all(x)
            session.commit()

    def clear(self):
        with Session(self.engine) as session:
            stmt = delete(Score)
            session.exec(stmt)

    def scores(self, metric_label: str|None=None, class_label: str|None=None):
        with Session(self.engine) as session:
            stmt = select(Score)
            if metric_label:
                stmt = stmt.where(Score.metric_label == metric_label)
            if class_label:
                stmt = stmt.where(Score.class_label == class_label)
            results = session.exec(stmt)
            return [r.score for r in results]

    def metric_scores(self, metric_labels: list[str], mode: ScoreLiteral='all'):
        MAP = {
            'all': lambda scores: scores,
            'mean': lambda scores: np.mean(scores),
            'std': lambda scores: np.std(scores),
            'mean|std': lambda scores: (np.mean(scores), np.std(scores)), 
        }

        return {metric_label: MAP[mode](self.scores(metric_label)) for metric_label in metric_labels}

        # r = {'all': {}, 'mean': {}, 'std': {}}
        # for metric in metric_labels:
        #     scores = self.scores(metric_label=metric)
        #     r['all'][metric] = scores
        #     r['mean'][metric] = np.mean(scores)
        #     r['std'][metric] = np.std(scores)
        #     r['mean|std'][metric] = (np.mean(scores), np.std(scores))
        # return r


    def class_scores(self, class_labels: list[str], mode: ScoreLiteral='all'):
        MAP = {
            'all': lambda scores: scores,
            'mean': lambda scores: np.mean(scores),
            'std': lambda scores: np.std(scores),
            'mean|std': lambda scores: (np.mean(scores), np.std(scores)), 
        }

        return {class_label: MAP[mode](self.scores(class_label)) for class_label in class_labels}

        # r = {'all': {}, 'mean': {}, 'std': {}}
        # for label in class_labels:
        #     scores = self.scores(class_label=label)
        #     r['all'][label] = scores
        #     r['mean'][label] = np.mean(scores)
        #     r['std'][label] = np.std(scores)
        #     r['mean|std'][label] = (np.mean(scores), np.std(scores))
        # return r

    def metric_class_scores(self, metric_labels: list[str], class_labels: list[str], mode: ScoreLiteral='all'):
        MAP = {
            'all': lambda scores: scores,
            'mean': lambda scores: np.mean(scores),
            'std': lambda scores: np.std(scores),
            'mean|std': lambda scores: (np.mean(scores), np.std(scores)), 
        }

        return {metric_label: {class_label: MAP[mode](self.scores(metric_label, class_label)) for class_label in class_labels} for metric_label in metric_labels}

        # def f(scores, mode):
        #     return {metric_label: {class_label: MAP[mode](scores)} for class_label in class_labels for metric_label in metric_labels}

        # r = {'all': {}, 'mean': {}, 'std': {}}
        # for metric in metric_labels:
        #     for label in class_labels:
        #         scores = self.scores(class_label=label, metric_label=metric)
        #         mean_scores, std_scores = np.mean(scores), np.std(scores)
        #         r['all'] = f(scores, 'all')
        #         r['mean'] = f(scores, 'mean')
        #         r['std'] = {metric_label: {class_label: std_scores} for class_label in class_labels for metric_label in metric_labels}
        #         r['mean|std'] = {metric_label: {class_label: (mean_scores, std_scores)} for class_label in class_labels for metric_label in metric_labels}
        # return r


    def class_metric_scores(self, metric_labels: list[str], class_labels: list[str], mode: ScoreLiteral='all'):
        MAP = {
            'all': lambda scores: scores,
            'mean': lambda scores: np.mean(scores),
            'std': lambda scores: np.std(scores),
            'mean|std': lambda scores: (np.mean(scores), np.std(scores)), 
        }

        return {class_label: {metric_label: MAP[mode](self.scores(metric_label, class_label)) for metric_label in metric_labels} for class_label in class_labels}
