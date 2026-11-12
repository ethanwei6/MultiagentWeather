"""Neutral baseline for exercises of SQLite corpus coverage and snapshot promotion."""

from types import SimpleNamespace


def fixture_weather_strategy():
    def score_buckets(distribution, buckets):
        return {bucket.label: 0.5 for bucket in buckets}

    return {
        "CityConfig": SimpleNamespace,
        "ForecastDistribution": SimpleNamespace,
        "TemperatureBucket": SimpleNamespace,
        "models": [SimpleNamespace(name="fixture", probability_by_bucket=score_buckets)],
        "source_weights": {},
        "model_weights": {},
    }
