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
        return _rank_peer_review(metrics[0], projects, rank_pass)

    print(f"unsupported target: {target.name}")
    return []


def _rank_peer_review(
    metric: Metric,
    projects: List[RankableProject],
    rank_pass: RankPass,
):
    dataframe = pandas.DataFrame((
        {
            "project_id": project.id,
            "group": project.group,
            "rewardable": project.rewardable,
            "metric": project.get_metric(metric.id).score,
        }
        for project in projects
    ))

    dataframe["rank_all"] = _rankdata(-dataframe["metric"])

    dataframe.sort_values(
        by=[
            "rank_all",
            "project_id",  # fallback if same `final_score`
        ],
        inplace=True,
    )

    dataframe.index = range(1, len(dataframe.index) + 1)

    if rank_pass == RankPass.PRE_DUPLICATE:
        dataframe["rank_final"] = numpy.nan

    elif rank_pass == RankPass.FINAL:
        mask = dataframe["rewardable"] & ~dataframe["group"].duplicated(keep="first")
        dataframe.loc[mask, "rank_final"] = _rankdata(dataframe.loc[mask, "rank_all"])

    else:
        raise ValueError(f"unknown rank pass: {rank_pass}")

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
