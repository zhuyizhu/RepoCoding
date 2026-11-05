FROM gradle:8.11.1-jdk17

USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends maven python3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /runner
COPY runner.py /runner/runner.py
RUN mkdir -p /cache/m2 /cache/gradle && chown -R gradle:gradle /cache /runner
ENV WORKTREE_ROOT=/workspace/worktrees
EXPOSE 8080
USER gradle
CMD ["python3", "/runner/runner.py"]
