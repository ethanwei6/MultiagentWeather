from __future__ import annotations

import importlib
import sys
import tempfile
import types
import unittest
from contextlib import contextmanager
from pathlib import Path

from daily_weather_pipeline.weather_model_validation import _load_weather_strategy


def _is_weather_strategy_module(name: str) -> bool:
    return name == "weather_strategy" or name.startswith("weather_strategy.")


@contextmanager
def _isolated_weather_strategy_modules():
    saved = {
        name: module
        for name, module in sys.modules.items()
        if _is_weather_strategy_module(name)
    }
    for name in tuple(saved):
        sys.modules.pop(name, None)
    importlib.invalidate_caches()
    try:
        yield
    finally:
        for name in tuple(sys.modules):
            if _is_weather_strategy_module(name):
                sys.modules.pop(name, None)
        sys.modules.update(saved)
        importlib.invalidate_caches()


def _write_minimal_weather_strategy(source_repo: Path) -> Path:
    package = source_repo / "weather_strategy"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "forecast.py").write_text(
        """
class ConsensusForecastEngine:
    def __init__(self):
        self.source_weights = {"fixture_source": 1.0}
        self.model_weights = {"fixture_model": 1.0}

class _Model:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

class EmpiricalEnsembleModel(_Model):
    pass

class FeatureAwareNormalModel(_Model):
    pass

class HourlyCurveMaxModel(_Model):
    pass

class KernelSmoothingModel(_Model):
    pass

class ParametricNormalModel(_Model):
    pass
""".lstrip(),
        encoding="utf-8",
    )
    (package / "models.py").write_text(
        """
class CityConfig:
    pass

class ForecastDistribution:
    pass

class TemperatureBucket:
    pass
""".lstrip(),
        encoding="utf-8",
    )
    importlib.invalidate_caches()
    return package.resolve()


class WeatherModelImportProvenanceTests(unittest.TestCase):
    def test_rejects_tainted_preloaded_module_and_restores_sys_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory, _isolated_weather_strategy_modules():
            root = Path(directory)
            source_repo = root / "vendored"
            _write_minimal_weather_strategy(source_repo)
            tainted_package = root / "outside" / "weather_strategy"
            tainted_package.mkdir(parents=True)
            tainted_init = tainted_package / "__init__.py"
            tainted_init.write_text("", encoding="utf-8")

            module = types.ModuleType("weather_strategy")
            module.__file__ = str(tainted_init)
            module.__path__ = [str(tainted_package)]
            sys.modules["weather_strategy"] = module
            original_sys_path = list(sys.path)

            with self.assertRaisesRegex(
                RuntimeError,
                "outside the exact vendored package tree",
            ):
                _load_weather_strategy(source_repo)

            self.assertEqual(sys.path, original_sys_path)
            self.assertIs(sys.modules["weather_strategy"], module)

    def test_rejects_preloaded_module_without_file_or_package_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory, _isolated_weather_strategy_modules():
            source_repo = Path(directory) / "vendored"
            _write_minimal_weather_strategy(source_repo)
            module = types.ModuleType("weather_strategy")
            module.__file__ = None
            module.__spec__ = None
            sys.modules["weather_strategy"] = module

            with self.assertRaisesRegex(RuntimeError, "origin is unprovable"):
                _load_weather_strategy(source_repo)

    def test_rejects_newly_imported_package_path_outside_vendored_tree(self) -> None:
        with tempfile.TemporaryDirectory() as directory, _isolated_weather_strategy_modules():
            root = Path(directory)
            source_repo = root / "vendored"
            package_root = _write_minimal_weather_strategy(source_repo)
            outside_package = root / "outside" / "weather_strategy"
            outside_package.mkdir(parents=True)
            (package_root / "__init__.py").write_text(
                f"__path__.append({str(outside_package)!r})\n",
                encoding="utf-8",
            )
            importlib.invalidate_caches()
            original_sys_path = list(sys.path)

            with self.assertRaisesRegex(
                RuntimeError,
                "outside the exact vendored package tree",
            ):
                _load_weather_strategy(source_repo)

            self.assertEqual(sys.path, original_sys_path)
            self.assertFalse(
                any(_is_weather_strategy_module(name) for name in sys.modules)
            )

    def test_loads_and_reuses_only_the_exact_vendored_package(self) -> None:
        with tempfile.TemporaryDirectory() as directory, _isolated_weather_strategy_modules():
            source_repo = Path(directory) / "vendored"
            package_root = _write_minimal_weather_strategy(source_repo)
            original_sys_path = list(sys.path)

            first = _load_weather_strategy(source_repo)
            first_forecast_module = sys.modules["weather_strategy.forecast"]
            second = _load_weather_strategy(source_repo)

            self.assertEqual(sys.path, original_sys_path)
            self.assertIs(
                sys.modules["weather_strategy.forecast"],
                first_forecast_module,
            )
            self.assertIs(first["CityConfig"], second["CityConfig"])
            self.assertEqual(first["source_weights"], {"fixture_source": 1.0})
            self.assertEqual(first["model_weights"], {"fixture_model": 1.0})
            for name, module in sys.modules.items():
                if not _is_weather_strategy_module(name):
                    continue
                origin = getattr(module, "__file__", None)
                if origin is None:
                    continue
                Path(origin).resolve().relative_to(package_root)


if __name__ == "__main__":
    unittest.main()
