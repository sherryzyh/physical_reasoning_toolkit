"""Front-end-free CMPhysBench SEED core (pure ``sympy``/``numpy``/stdlib).

Importing this package pulls no ``latex2sympy2_extended`` and no ``pint``: the LaTeX
front-end is imported lazily by :func:`.seed.SEED`, and ``pint`` is a lazy singleton
(:func:`.seed._get_ureg`) loaded only on the unit-aware Numeric path.
"""
