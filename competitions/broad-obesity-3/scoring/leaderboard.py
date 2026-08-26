import typing
from typing import List

import numpy
import pandas
import scipy.stats
from crunch.api import Metric, Target
from crunch.unstructured import RankableProject, RankedProject, RankPass


def rank(
    target: Target,
    metrics: List[Metric],
    projects: List[RankableProject],
    rank_pass: RankPass,
):
    if target.name == "peer-review":
        return _rank_peer_review(metrics, projects, rank_pass)

    print(f"unsupported target: {target.name}")
    return []


def _rank_peer_review(
    metrics: List[Metric],
    projects: List[RankableProject],
    rank_pass: RankPass,
):
    (
        average_score_metric,
        review_1_score_metric,
        review_2_score_metric,
        review_3_score_metric,
    ) = metrics
    
    dataframe = pandas.DataFrame((
        {
            "project_id": project.id,
            "group": project.group,
            "rewardable": project.rewardable,
            "metric": project.get_metric(average_score_metric.id).score,
            "review_3_score_count": sum([
                project.get_metric(review_1_score_metric.id).score == 3.0,
                project.get_metric(review_2_score_metric.id).score == 3.0,
                project.get_metric(review_3_score_metric.id).score == 3.0,
            ]),
        }
        for project in projects
    ))

    dataframe["metric_rank"] = _rankdata(-dataframe["metric"])

    dataframe.sort_values(
        by=[
            "metric_rank",
            "review_3_score_count",
            "project_id",  # fallback if same `final_score`
        ],
        ascending=[
            True,
            False,
            True,
        ],
        inplace=True,
    )

    dataframe.index = range(1, len(dataframe.index) + 1)

    mask = dataframe["rewardable"]
    dataframe.loc[mask, "rank_final"] = _rankdata(dataframe.loc[mask, "metric_rank"])

    return [
        RankedProject(
            id=int(row["project_id"]),
            rank=index,
            reward_rank=None if numpy.isnan(row["rank_final"]) else row["rank_final"],
        )
        for index, row in dataframe.iterrows()
    ]


def _rankdata(array: typing.List[typing.Tuple[int, float]]):
    return scipy.stats.rankdata(array, method="min")
