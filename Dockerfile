# Reproducible environment for hcm-twin.
#
#   docker build -t hcm-twin .
#   docker run --rm -v "$PWD/results:/app/results" -v "$PWD/paper:/app/paper" hcm-twin
#
# The default command runs the whole study: the test suite including every validation
# gate, the virtual population from a logged seed, every table and figure with the CSV
# behind it, the interactive explorer, and the compiled writeup.
#
# TeX Live is installed from the distribution rather than fetched at run time, so the
# build is the only step that needs a network and a container run is offline-reproducible.

FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib

# TeX Live subset: base plus the packages main.tex actually loads (geometry, microtype,
# booktabs, caption, subcaption, enumitem, hyperref) and the fonts they need.
RUN apt-get update && apt-get install --no-install-recommends -y \
        texlive-latex-base \
        texlive-latex-recommended \
        texlive-latex-extra \
        texlive-fonts-recommended \
        lmodern \
        make \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first, so a source edit does not invalidate the dependency layer.
COPY pyproject.toml README.md ./
RUN pip install --upgrade pip && pip install '.[dev]' || true

COPY src/ ./src/
COPY tests/ ./tests/
COPY docs/ ./docs/
COPY paper/ ./paper/
COPY notebooks/ ./notebooks/
COPY Makefile ./

RUN pip install -e '.[dev]'

# A non-root user, and writable homes for the caches matplotlib and pytest want.
RUN useradd --create-home --uid 1000 runner \
    && mkdir -p /app/results /tmp/matplotlib \
    && chown -R runner:runner /app /tmp/matplotlib
USER runner

# Study size. Override at run time for a quick pass:
#   docker run --rm hcm-twin make all N_BASE=48 CASES=8 DESIGN=200 MCMC=800
ENV N_BASE=385 CASES=50 DESIGN=400 MCMC=3000

CMD ["sh", "-c", "make all PY=python N_BASE=$N_BASE CASES=$CASES DESIGN=$DESIGN MCMC=$MCMC"]
