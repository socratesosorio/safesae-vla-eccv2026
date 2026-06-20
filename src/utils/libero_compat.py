"""Compatibility helpers for resolving LIBERO APIs across repo layouts."""

from __future__ import annotations

import importlib
import inspect
import os
import pkgutil
import sys
from collections.abc import Callable, Mapping
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any


def _looks_like_suite(obj: Any) -> bool:
    return hasattr(obj, "get_num_tasks") or hasattr(obj, "get_task")


def _import_optional(module_name: str) -> Any | None:
    try:
        return importlib.import_module(module_name)
    except Exception:
        return None


def _known_libero_roots() -> list[Path]:
    candidates = []
    # Support LIBERO_ROOT env var for custom installs.
    _libero_root = os.environ.get("LIBERO_ROOT")
    if _libero_root:
        candidates.append(Path(_libero_root))
    candidates += [
        Path.home() / "LIBERO",
        Path.home() / "LIBERO" / "libero",
        Path("/opt/LIBERO"),
        Path("/opt/LIBERO/LIBERO"),
        Path("/opt/LIBERO/libero"),
    ]
    return [path for path in candidates if path.exists()]


def _ensure_local_libero_preferred() -> None:
    roots = _known_libero_roots()
    if not roots:
        return

    # Ensure local checkout paths are first in sys.path.
    for root in reversed(roots):
        root_str = str(root)
        if root_str in sys.path:
            sys.path.remove(root_str)
        sys.path.insert(0, root_str)

    # Fix missing outer __init__.py for double-nested LIBERO structure.
    for root in roots:
        outer_libero = root / "libero"
        inner_init = outer_libero / "libero" / "__init__.py"
        outer_init = outer_libero / "__init__.py"
        if inner_init.exists() and not outer_init.exists():
            try:
                outer_init.write_text("")
            except OSError:
                pass  # Read-only filesystem

    # If a different "libero" package was imported earlier, drop it and re-import.
    loaded = sys.modules.get("libero")
    loaded_file = str(getattr(loaded, "__file__", "") or "")
    if loaded is not None and not any(str(root) in loaded_file for root in roots):
        for name in list(sys.modules.keys()):
            if name == "libero" or name.startswith("libero."):
                sys.modules.pop(name, None)


def _discover_submodules(root: ModuleType) -> list[ModuleType]:
    discovered: list[ModuleType] = []
    package_path = getattr(root, "__path__", None)
    if package_path is None:
        return discovered
    prefix = f"{root.__name__}."
    for module_info in pkgutil.walk_packages(package_path, prefix):
        name = module_info.name
        # Only pull likely targets to avoid importing unrelated heavy modules.
        if "benchmark" not in name and ".env" not in name:
            continue
        module = _import_optional(name)
        if isinstance(module, ModuleType):
            discovered.append(module)
    return discovered


def _candidate_modules() -> list[ModuleType]:
    _ensure_local_libero_preferred()
    module_names = [
        "libero.libero",
        "libero",
        "libero.libero.benchmark",
        "libero.benchmark",
        "libero.libero.benchmarks",
        "libero.benchmarks",
    ]
    modules: list[ModuleType] = []
    seen: set[str] = set()

    def _add(module: Any) -> None:
        if not isinstance(module, ModuleType):
            return
        name = module.__name__
        if name in seen:
            return
        seen.add(name)
        modules.append(module)

    for module_name in module_names:
        _add(_import_optional(module_name))

    if not modules:
        raise ModuleNotFoundError("Could not import any LIBERO root/benchmark modules")

    # Add nested benchmark/env modules exposed as attributes.
    for module in list(modules):
        for attr_name in dir(module):
            attr = getattr(module, attr_name, None)
            if not isinstance(attr, ModuleType):
                continue
            lower_name = attr.__name__.lower()
            if "benchmark" in lower_name or ".env" in lower_name:
                _add(attr)

    # Walk package trees for benchmark/env modules.
    for module in list(modules):
        for submodule in _discover_submodules(module):
            _add(submodule)

    return modules


def _materialize(entry: Any) -> Any:
    if callable(entry):
        try:
            result = entry()
        except TypeError:
            return entry
        # Some LIBERO versions store a factory that returns another callable.
        if callable(result) and not _looks_like_suite(result):
            try:
                return result()
            except TypeError:
                return result
        return result
    return entry


_SUITE_RENAME_MAP: dict[str, str] = {
    "libero_long": "libero_10",
    "long": "10",
}


def _suite_key_aliases(suite_key: str) -> list[str]:
    aliases = [suite_key]
    if suite_key.startswith("libero_"):
        aliases.append(suite_key[len("libero_") :])
    else:
        aliases.append(f"libero_{suite_key}")
    # Handle known renames (e.g. "libero_long" → "libero_90").
    renamed = _SUITE_RENAME_MAP.get(suite_key)
    if renamed is not None and renamed not in aliases:
        aliases.append(renamed)
        if renamed.startswith("libero_"):
            aliases.append(renamed[len("libero_"):])
        else:
            aliases.append(f"libero_{renamed}")
    expanded: list[str] = []
    for alias in aliases:
        for candidate in (alias, alias.lower()):
            if candidate not in expanded:
                expanded.append(candidate)
    return expanded


def _looks_like_benchmark_mapping(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    if not value:
        return False
    keys = list(value.keys())
    if not all(isinstance(key, str) for key in keys):
        return False
    # A real LIBERO benchmark mapping has ~4-10 keys.  Reject large dicts
    # (e.g. builtins.__dict__, module.__dict__) that pass generic heuristics.
    if len(keys) > 50:
        return False
    key_lowers = [str(key).lower() for key in keys]
    if any(key.startswith("libero_") for key in key_lowers):
        return True
    # Require LIBERO-specific substrings.  Bare "object" is too generic and
    # matches Python builtins; use prefixed forms instead.
    hints = ("spatial", "libero_object", "libero_goal", "libero_long")
    if any(any(hint in key for hint in hints) for key in key_lowers):
        return True
    values = list(value.values())
    return any(_looks_like_suite(v) for v in values[:16])


def _find_benchmark_mapping(modules: list[ModuleType]) -> Mapping[str, Any] | None:
    candidate_attr_names = (
        "BENCHMARK_MAPPING",
        "benchmark_mapping",
        "benchmark_dict",
        "BENCHMARK_DICT",
        "benchmark_map",
        "BENCHMARK_MAP",
        "benchmarks",
        "BENCHMARKS",
        "get_benchmark_dict",
        "get_benchmark_map",
        "get_benchmark_mapping",
    )
    # 1. Check well-known attribute names on each module.
    for module in modules:
        for attr_name in candidate_attr_names:
            value = getattr(module, attr_name, None)
            if callable(value) and attr_name.startswith("get_"):
                try:
                    value = value()
                except Exception:
                    value = None
            if _looks_like_benchmark_mapping(value):
                return value

    # 2. Source-file fallback: directly import benchmark __init__.py from known paths.
    #    This runs BEFORE the generic attr scan to avoid false positives from
    #    module dunder attributes (e.g. __builtins__).
    _benchmark_source_files_list = [
        Path.home() / "LIBERO" / "libero" / "libero" / "benchmark" / "__init__.py",
        Path.home() / "LIBERO" / "libero" / "benchmark" / "__init__.py",
        Path("/opt/LIBERO/libero/libero/benchmark/__init__.py"),
        Path("/opt/LIBERO/libero/benchmark/__init__.py"),
        Path("/opt/LIBERO/LIBERO/libero/benchmark/__init__.py"),
    ]
    _libero_root_env = os.environ.get("LIBERO_ROOT")
    if _libero_root_env:
        _benchmark_source_files_list.insert(0, Path(_libero_root_env) / "libero" / "libero" / "benchmark" / "__init__.py")
        _benchmark_source_files_list.insert(1, Path(_libero_root_env) / "libero" / "benchmark" / "__init__.py")
    _benchmark_source_files = tuple(_benchmark_source_files_list)
    _benchmark_attr_names = ("BENCHMARK_MAPPING", "benchmark_mapping", "BENCHMARK_DICT")
    for idx, src in enumerate(_benchmark_source_files):
        if not src.exists():
            continue
        try:
            spec = importlib.util.spec_from_file_location(
                f"_libero_benchmark_fallback_{idx}", str(src)
            )
            if spec is None or spec.loader is None:
                continue
            fb_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(fb_module)
        except Exception:
            continue
        for attr_name in _benchmark_attr_names:
            value = getattr(fb_module, attr_name, None)
            if _looks_like_benchmark_mapping(value):
                return value
        # Also try get_benchmark_dict() callable
        getter = getattr(fb_module, "get_benchmark_dict", None)
        if callable(getter):
            try:
                value = getter()
                if _looks_like_benchmark_mapping(value):
                    return value
            except Exception:
                pass

    # 3. Last-resort generic scan: any non-dunder mapping attr that looks like
    #    a LIBERO suite map.  Skips private/dunder attrs to avoid matching
    #    __builtins__ or other internal dicts.
    for module in modules:
        for attr_name in dir(module):
            if attr_name.startswith("_"):
                continue
            value = getattr(module, attr_name, None)
            if _looks_like_benchmark_mapping(value) and len(value) >= 2:
                return value

    return None


def _call_getter_with_key(fn: Callable[..., Any], suite_key: str) -> Any:
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return fn(suite_key)

    params = signature.parameters
    preferred_kwargs = (
        {"benchmark_name": suite_key},
        {"suite_name": suite_key},
        {"name": suite_key},
        {"key": suite_key},
    )
    for kwargs in preferred_kwargs:
        if all(name in params for name in kwargs):
            return fn(**kwargs)

    positional_params = [
        p
        for p in params.values()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    required_positional = [p for p in positional_params if p.default is p.empty]
    if len(required_positional) == 1:
        return fn(suite_key)

    # Some APIs expose fn() -> mapping.
    if len(required_positional) == 0:
        maybe_mapping = fn()
        if isinstance(maybe_mapping, Mapping):
            if suite_key in maybe_mapping:
                return maybe_mapping[suite_key]
            raise KeyError(suite_key)
        return maybe_mapping

    raise TypeError(f"Unsupported LIBERO getter signature for {fn!r}")


def _find_benchmark_getters(modules: list[ModuleType]) -> list[Callable[..., Any]]:
    getter_names = (
        "get_benchmark",
        "get_benchmark_dict",
        "build_benchmark",
        "make_benchmark",
        "create_benchmark",
        "benchmark",
    )
    getters: list[Callable[..., Any]] = []
    seen: set[int] = set()
    for module in modules:
        for getter_name in getter_names:
            getter = getattr(module, getter_name, None)
            if callable(getter) and id(getter) not in seen:
                getters.append(getter)
                seen.add(id(getter))
    return getters


def _resolve_suite_factory() -> Callable[[str], Any]:
    modules = _candidate_modules()
    benchmark_mapping = _find_benchmark_mapping(modules)
    getters = _find_benchmark_getters(modules)

    if benchmark_mapping is None and not getters:
        module_names = ", ".join(module.__name__ for module in modules)
        raise AttributeError(
            "Could not resolve LIBERO suite factory. "
            f"Loaded modules: {module_names}. "
            "No benchmark mapping or known getter functions were found."
        )

    def _combined_factory(suite_key: str) -> Any:
        # 1. Try the mapping first (if non-empty).
        if benchmark_mapping:
            for alias in _suite_key_aliases(suite_key):
                if alias in benchmark_mapping:
                    return _materialize(benchmark_mapping[alias])

        # 2. Fall back to getter functions.
        last_exc: Exception | None = None
        for getter in getters:
            for alias in _suite_key_aliases(suite_key):
                try:
                    candidate = _call_getter_with_key(getter, alias)
                except Exception as exc:
                    last_exc = exc
                    continue
                return _materialize(candidate)

        # 3. Build a diagnostic error message.
        available = list(benchmark_mapping.keys()) if benchmark_mapping else []
        mapping_source = getattr(benchmark_mapping, "__module__", "unknown") if benchmark_mapping else "none"
        getter_detail = f" Last getter error: {last_exc!r}" if last_exc is not None else ""
        raise KeyError(
            f"Suite key '{suite_key}' missing from LIBERO benchmark mapping. "
            f"Available keys: {available}. "
            f"Mapping source: {mapping_source}. "
            f"Getters tried: {len(getters)}.{getter_detail}"
        )

    return _combined_factory


def _resolve_offscreen_render_env() -> Any:
    _ensure_local_libero_preferred()
    known_module_names = (
        "libero.libero.envs",
        "libero.envs",
        "libero.libero.envs.env_wrapper",
        "libero.envs.env_wrapper",
    )
    modules = [m for m in (_import_optional(name) for name in known_module_names) if isinstance(m, ModuleType)]
    modules.extend(_candidate_modules())

    seen_module_names: set[str] = set()
    unique_modules: list[ModuleType] = []
    for module in modules:
        if module.__name__ in seen_module_names:
            continue
        seen_module_names.add(module.__name__)
        unique_modules.append(module)

    explicit_env_names = (
        "OffScreenRenderEnv",
        "OffscreenRenderEnv",
        "OffScreenEnv",
        "OffscreenEnv",
    )
    for module in unique_modules:
        for attr_name in explicit_env_names:
            env_ctor = getattr(module, attr_name, None)
            if callable(env_ctor):
                return env_ctor

    # Fallback: scan for any callable that looks like an off-screen env constructor.
    for module in unique_modules:
        for attr_name in dir(module):
            lowered = attr_name.lower()
            if "offscreen" not in lowered or "env" not in lowered:
                continue
            env_ctor = getattr(module, attr_name, None)
            if callable(env_ctor):
                return env_ctor

    # Final fallback: some LIBERO versions expose a generic env getter function.
    getter_names = (
        "get_libero_env",
        "make_libero_env",
        "create_libero_env",
        "build_libero_env",
        "get_env",
        "make_env",
        "create_env",
    )

    def _filtered_kwargs(fn: Callable[..., Any], kwargs: dict[str, Any]) -> dict[str, Any]:
        try:
            sig = inspect.signature(fn)
        except (TypeError, ValueError):
            return kwargs
        params = sig.parameters
        if any(p.kind == p.VAR_KEYWORD for p in params.values()):
            return kwargs

        out = {k: v for k, v in kwargs.items() if k in params}
        if "bddl_file_name" in kwargs:
            for alias in ("bddl_file", "bddl_path", "bddl_file_path", "task_bddl_file", "bddl"):
                if alias in params and alias not in out:
                    out[alias] = kwargs["bddl_file_name"]
        return out

    for module in unique_modules:
        for getter_name in getter_names:
            getter = getattr(module, getter_name, None)
            if not callable(getter):
                continue

            def _env_ctor(_getter: Callable[..., Any] = getter, **env_kwargs: Any) -> Any:
                kwargs = _filtered_kwargs(_getter, dict(env_kwargs))
                try:
                    return _getter(**kwargs)
                except TypeError:
                    # Last resort for simple signatures like fn(bddl_path).
                    bddl = env_kwargs.get("bddl_file_name")
                    return _getter(bddl)

            return _env_ctor

    # ── Clean-retry fallback ──
    # The first import attempt above may have failed and left partial
    # modules in sys.modules.  With bddl and robosuite==1.4.1 installed,
    # a clean re-import of libero.libero.envs should now succeed.
    _pkg = "libero.libero.envs"
    _partial_keys = [k for k in sys.modules if k == _pkg or k.startswith(_pkg + ".")]
    if _partial_keys:
        for _k in _partial_keys:
            sys.modules.pop(_k, None)
        try:
            _envs_mod = importlib.import_module(_pkg)
            for attr_name in explicit_env_names:
                env_ctor = getattr(_envs_mod, attr_name, None)
                if callable(env_ctor):
                    return env_ctor
        except Exception:
            pass  # Fall through to bootstrap below.

    # ── Bootstrap fallback ──
    # The envs __init__.py does heavy wildcard imports (from .problems,
    # .robots, .arenas) that can fail.  We only need OffScreenRenderEnv
    # from env_wrapper.py, so bootstrap a stub envs package with the
    # names env_wrapper.py actually needs at module scope.
    import types as _types

    _envs_dirs = []
    # Support LIBERO_ROOT env var for custom installs.
    _libero_root = os.environ.get("LIBERO_ROOT")
    if _libero_root:
        _envs_dirs.append(Path(_libero_root) / "libero" / "libero" / "envs")
        _envs_dirs.append(Path(_libero_root) / "libero" / "envs")
    _envs_dirs += [
        Path.home() / "LIBERO" / "libero" / "libero" / "envs",
        Path("/opt/LIBERO/libero/libero/envs"),
        Path("/opt/LIBERO/LIBERO/libero/envs"),
    ]
    for _envs_dir in _envs_dirs:
        if not _envs_dir.is_dir():
            continue

        # Clear any leftover partial state from failed imports.
        for _k in [k for k in sys.modules if k == _pkg or k.startswith(_pkg + ".")]:
            sys.modules.pop(_k, None)

        # 1. Create a stub envs package.
        _stub = _types.ModuleType(_pkg)
        _stub.__path__ = [str(_envs_dir)]
        _stub.__package__ = _pkg
        # Provide TASK_MAPPING and register_problem on the stub so that
        # `from libero.libero.envs import *` inside env_wrapper.py always
        # finds them, even if bddl_base_domain.py cannot be loaded.
        _stub.TASK_MAPPING = {}  # type: ignore[attr-defined]

        def _register_problem(target_class):
            """Stub matching real LIBERO decorator: register_problem(cls)."""
            _stub.TASK_MAPPING[target_class.__name__.lower()] = target_class  # type: ignore[attr-defined]
            return target_class

        _stub.register_problem = _register_problem  # type: ignore[attr-defined]
        sys.modules[_pkg] = _stub

        # 2. Try loading critical submodules that env_wrapper.py needs.
        for _sub_name in ("bddl_base_domain", "base_object", "bddl_utils"):
            _full = f"{_pkg}.{_sub_name}"
            if _full in sys.modules:
                continue
            _sub_path = _envs_dir / f"{_sub_name}.py"
            if not _sub_path.exists():
                continue
            try:
                _spec = importlib.util.spec_from_file_location(_full, str(_sub_path))
                if _spec is None or _spec.loader is None:
                    continue
                _mod = importlib.util.module_from_spec(_spec)
                sys.modules[_full] = _mod
                _spec.loader.exec_module(_mod)
                setattr(_stub, _sub_name, _mod)
                # Hoist public names (TASK_MAPPING, OBJECTS_DICT, etc.)
                # so `from libero.libero.envs import *` finds them.
                for _name in dir(_mod):
                    if not _name.startswith("_"):
                        setattr(_stub, _name, getattr(_mod, _name))
            except Exception as _sub_exc:
                import logging as _logging
                _logging.warning(f"Bootstrap submodule {_full} from {_sub_path} failed: {_sub_exc}")
                sys.modules.pop(_full, None)

        # 3. Load env_wrapper.py against the bootstrapped stub.
        _ew_path = _envs_dir / "env_wrapper.py"
        if not _ew_path.exists():
            continue
        _ew_full = f"{_pkg}.env_wrapper"
        try:
            _spec = importlib.util.spec_from_file_location(_ew_full, str(_ew_path))
            if _spec is not None and _spec.loader is not None:
                _mod = importlib.util.module_from_spec(_spec)
                sys.modules[_ew_full] = _mod
                _spec.loader.exec_module(_mod)
                for attr_name in (
                    "OffScreenRenderEnv",
                    "OffscreenRenderEnv",
                    "OffScreenEnv",
                    "OffscreenEnv",
                ):
                    env_ctor = getattr(_mod, attr_name, None)
                    if callable(env_ctor):
                        return env_ctor
        except Exception as _ew_exc:
            import logging as _logging
            _logging.warning(f"Bootstrap env_wrapper.py from {_ew_path} failed: {_ew_exc}")
            sys.modules.pop(_ew_full, None)

    loaded = ", ".join(m.__name__ for m in unique_modules)
    raise AttributeError(
        "Could not resolve LIBERO OffScreenRenderEnv in known env modules. "
        f"Loaded modules: {loaded}"
    )


def _nuclear_envs_import() -> ModuleType:
    """Clear ALL libero modules from sys.modules and reimport libero.libero.envs.

    This is the only reliable way to get a fully-initialized envs package
    with a populated TASK_MAPPING, because the envs __init__.py has a strict
    import order (bddl_base_domain → problems → robots → arenas → env_wrapper)
    that partial-import workarounds cannot reproduce.
    """
    _ensure_local_libero_preferred()
    for k in list(sys.modules.keys()):
        if k == "libero" or k.startswith("libero."):
            sys.modules.pop(k, None)
    return importlib.import_module("libero.libero.envs")


@lru_cache(maxsize=1)
def _cached_offscreen_render_env() -> Any:
    return _resolve_offscreen_render_env()


def _ensure_libero_robots_registered() -> None:
    """Ensure LIBERO's custom robots are in robosuite's ROBOT_CLASS_MAPPING.

    LIBERO defines MountedPanda and OnTheGroundPanda (both SingleArm) in
    ``libero.libero.envs.robots``.  If the import ordering didn't register
    them, we do it here defensively via a triple-layer approach:

    1. Import ``libero.libero.envs.robots`` to trigger LIBERO's canonical
       ``ROBOT_CLASS_MAPPING.update()`` code path.
    2. Defensive dict update on the canonical ``robosuite.robots`` mapping.
    3. Direct module-level patch on ``manipulation_env`` in case it holds a
       different dict reference due to import ordering.
    """
    robot_names = ("MountedPanda", "OnTheGroundPanda")

    # Layer 1: trigger LIBERO's own robot registration
    try:
        import libero.libero.envs.robots  # noqa: F401
    except Exception:
        pass

    # Layer 2: defensive dict update on robosuite's canonical mapping
    try:
        from robosuite.robots import ROBOT_CLASS_MAPPING
        from robosuite.robots.single_arm import SingleArm

        for name in robot_names:
            if name not in ROBOT_CLASS_MAPPING:
                ROBOT_CLASS_MAPPING[name] = SingleArm
    except Exception:
        pass

    # Layer 3: patch manipulation_env's own reference if it exists
    try:
        from robosuite.robots.single_arm import SingleArm

        manip = sys.modules.get("robosuite.environments.manipulation.manipulation_env")
        if manip is not None and hasattr(manip, "ROBOT_CLASS_MAPPING"):
            for name in robot_names:
                if name not in manip.ROBOT_CLASS_MAPPING:
                    manip.ROBOT_CLASS_MAPPING[name] = SingleArm
    except Exception:
        pass


def _lazy_offscreen_render_env(*args: Any, **kwargs: Any) -> Any:
    ctor = _cached_offscreen_render_env()
    _ensure_libero_robots_registered()

    # Fast path: check if TASK_MAPPING is already populated.
    ew_mod = sys.modules.get("libero.libero.envs.env_wrapper")
    tm = getattr(ew_mod, "TASK_MAPPING", None) if ew_mod else None
    if isinstance(tm, dict) and len(tm) > 0:
        try:
            return ctor(*args, **kwargs)
        except KeyError:
            # Robot class not found — nuclear reimport re-runs LIBERO's
            # envs/__init__.py which registers the custom robots.
            envs_mod = _nuclear_envs_import()
            _ensure_libero_robots_registered()
            ctor = getattr(envs_mod, "OffScreenRenderEnv", ctor)
            _cached_offscreen_render_env.cache_clear()
            return ctor(*args, **kwargs)

    # TASK_MAPPING is empty — the bootstrap/partial-import left it unpopulated.
    # Nuclear option: clear ALL libero modules and reimport from scratch.
    # With bddl + robosuite 1.4.1 installed, the clean import will populate
    # TASK_MAPPING via the @register_problem decorators in problems/*.py.
    try:
        envs_mod = _nuclear_envs_import()
        ctor = getattr(envs_mod, "OffScreenRenderEnv", ctor)
        # Clear the lru_cache so future calls use the properly imported class.
        _cached_offscreen_render_env.cache_clear()
        return ctor(*args, **kwargs)
    except Exception as reimport_err:
        # If even the nuclear reimport fails, try syncing TASK_MAPPING dicts
        # as a last resort.  Find any populated TASK_MAPPING and propagate it
        # to ALL modules that reference it.
        _mod_names = (
            "libero.libero.envs.bddl_base_domain",
            "libero.libero.envs",
            "libero.libero.envs.env_wrapper",
        )
        all_tms: list[dict] = []
        for mn in _mod_names:
            m = sys.modules.get(mn)
            if m is None:
                continue
            t = getattr(m, "TASK_MAPPING", None)
            if isinstance(t, dict):
                all_tms.append(t)
        if all_tms:
            best = max(all_tms, key=len)
            if best:
                for t in all_tms:
                    if t is not best:
                        t.update(best)
        try:
            return ctor(*args, **kwargs)
        except (KeyError, NameError):
            raise RuntimeError(
                f"LIBERO OffScreenRenderEnv failed: TASK_MAPPING is empty.  "
                f"Nuclear reimport error: {reimport_err!r}. "
                f"Check that bddl>=1.0.0, robosuite==1.4.1, and mujoco>=3.0 are installed, "
                f"and that /opt/LIBERO/libero/__init__.py exists."
            ) from reimport_err


@lru_cache(maxsize=1)
def resolve_libero_apis() -> tuple[Callable[[str], Any], Any]:
    """Return `(suite_factory, OffScreenRenderEnv)` for whichever LIBERO layout is installed."""
    suite_factory = _resolve_suite_factory()
    return suite_factory, _lazy_offscreen_render_env
