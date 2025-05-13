import os
import os.path
import numpy as np
from sqlmodel import SQLModel, Session, select, Field, create_engine, delete
from typing import Literal

ScoreLiteral = Literal['all', 'mean', 'std', 'mean|std']

class Score(SQLModel, table=True):
    __tablename__: str = 'scores'

    id: int = Field(primary_key=True, default=None)
    metric_label: str = Field()
    class_label: str = Field()
    score: float = Field()

class ScoreDB:
    def __init__(self, filepath: str):
        self.sqlite_filepath = filepath
        self.sqlite_url = f'sqlite:///{self.sqlite_filepath}'
        self.engine = create_engine(self.sqlite_url)
        if not os.path.exists(filepath):
            SQLModel.metadata.create_all(self.engine)

    def add(self, metric_label: str, class_label: str, scores: list[float]|float):
        with Session(self.engine) as session:
            x = [Score(metric_label=metric_label, class_label=class_label, score=score) for score in scores]
            session.add_all(x)
            session.commit()

    def clear(self):
        with Session(self.engine) as session:
            stmt = delete(Score)
            session.exec(stmt)
            session.commit()

    def scores(self, metric_label: str|None=None, class_label: str|None=None):
        with Session(self.engine) as session:
            stmt = select(Score)
            if metric_label:
                stmt = stmt.where(Score.metric_label == metric_label)
            if class_label:
                stmt = stmt.where(Score.class_label == class_label)
            results = session.exec(stmt)
            session.commit()
            return [r.score for r in results]

    def metric_scores(self, metric_labels: list[str], mode: ScoreLiteral='all'):
        MAP = {
            'all': lambda scores: scores,
            'mean': lambda scores: np.mean(scores),
            'std': lambda scores: np.std(scores),
            'mean|std': lambda scores: (np.mean(scores), np.std(scores)), 
        }

        return {metric_label: MAP[mode](self.scores(metric_label)) for metric_label in metric_labels}

    def class_scores(self, class_labels: list[str], mode: ScoreLiteral='all'):
        MAP = {
            'all': lambda scores: scores,
            'mean': lambda scores: np.mean(scores),
            'std': lambda scores: np.std(scores),
            'mean|std': lambda scores: (np.mean(scores), np.std(scores)), 
        }

        return {class_label: MAP[mode](self.scores(class_label)) for class_label in class_labels}

    def metric_class_scores(self, metric_labels: list[str], class_labels: list[str], mode: ScoreLiteral='all'):
        MAP = {
            'all': lambda scores: scores,
            'mean': lambda scores: np.mean(scores),
            'std': lambda scores: np.std(scores),
            'mean|std': lambda scores: (np.mean(scores), np.std(scores)), 
        }

        return {metric_label: {class_label: MAP[mode](self.scores(class_label))} for class_label in class_labels for metric_label in metric_labels}

    def class_metric_scores(self, metric_labels: list[str], class_labels: list[str], mode: ScoreLiteral='all'):
        MAP = {
            'all': lambda scores: scores,
            'mean': lambda scores: np.mean(scores),
            'std': lambda scores: np.std(scores),
            'mean|std': lambda scores: (np.mean(scores), np.std(scores)), 
        }

        return {class_label: {metric_label: MAP[mode](self.scores(class_label)) for metric_label in metric_labels} for class_label in class_labels}
