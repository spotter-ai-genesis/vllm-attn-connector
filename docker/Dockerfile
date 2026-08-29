ARG VLLM_TAG=v0.28.0
FROM vllm/vllm-openai:${VLLM_TAG}

# Flowcept fork carries the vLLM adapter (flowcept.flowceptor.adapters.vllm), absent from released flowcept.
# `pandas` is a hard runtime dep of Flowcept's mongodb_dao but is NOT pulled by flowcept[mongo]
# (which only adds pymongo+pyarrow); without it, persistence to MongoDB fails at import time.
RUN python3 -m pip install --no-cache-dir "flowcept[redis,mongo] @ git+https://github.com/spotter-ai-genesis/flowcept.git" pandas

COPY . /opt/vllm-kvnorm
RUN python3 -m pip install --no-cache-dir /opt/vllm-kvnorm

# Fail the build early if the connector or adapter can't import against this vLLM
# (pandas included: its absence only surfaces at Mongo-persistence time, so check it here).
RUN python3 -c "import vllm, flowcept, pandas, vllm_kvnorm; from vllm_kvnorm import KVNormConnector; print('vllm', vllm.__version__, '| KVNormConnector OK')"
