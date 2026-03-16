"""
Poetry build hook for Cython extensions.

Poetry calls build() during `poetry install` / `poetry build` when
build-system.build-backend = "poetry.core.masonry.api" and a build.py exists.

Manual build (inplace, for development):
    poetry run python build.py build_ext --inplace
"""
from setuptools import Extension
from Cython.Build import cythonize
import numpy as np


_EXTENSIONS = [
    Extension(
        name="src.envs._crossing",
        sources=["src/envs/_crossing.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3", "-march=native", "-ffast-math"],
    ),
]

_COMPILER_DIRECTIVES = {
    "language_level": "3",
    "boundscheck": False,
    "wraparound": False,
    "cdivision": True,
    "nonecheck": False,
}


def build(setup_kwargs: dict):
    """Called by poetry-core during install/build."""
    setup_kwargs.update(
        {
            "ext_modules": cythonize(
                _EXTENSIONS,
                compiler_directives=_COMPILER_DIRECTIVES,
            )
        }
    )


# Allow direct invocation: python build.py build_ext --inplace
if __name__ == "__main__":
    from setuptools import setup

    setup(
        package_dir={"": "."},  # root packages live at project root → src/envs/_crossing
        ext_modules=cythonize(
            _EXTENSIONS,
            compiler_directives=_COMPILER_DIRECTIVES,
        )
    )
