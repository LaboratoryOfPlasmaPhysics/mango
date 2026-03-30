FROM fedora:43

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

RUN useradd -m mango

RUN dnf install -y --setopt=install_weak_deps=False \
    python3 python3-devel git gcc g++ && \
    dnf clean all

WORKDIR /home/mango
COPY --chown=mango:mango . /home/mango/mango

USER mango

WORKDIR /home/mango/mango
RUN uv sync --no-dev --extra server

ARG PORT=8000
ENV PORT=$PORT
ENV MANGO_DATA_DIR=/data/mango

VOLUME /data/mango
EXPOSE $PORT

COPY --chown=mango:mango docker/entry_point.sh /home/mango/entry_point.sh
ENTRYPOINT ["/home/mango/entry_point.sh"]
