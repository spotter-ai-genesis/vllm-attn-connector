ARG VLLM_TAG=v0.28.0
FROM vllm/vllm-openai:${VLLM_TAG}

# Flowcept fork carries the vLLM adapter (flowcept.flowceptor.adapters.vllm), absent from released flowcept.
RUN python3 -m pip install --no-cache-dir "flowcept[redis,mongo] @ git+https://github.com/spotter-ai-genesis/flowcept.git"

COPY . /opt/vllm-kvnorm
RUN python3 -m pip install --no-cache-dir /opt/vllm-kvnorm

# Fail the build early if the connector or adapter can't import against this vLLM.
RUN python3 -c "import vllm, flowcept, vllm_kvnorm; from vllm_kvnorm import KVNormConnector; print('vllm', vllm.__version__, '| KVNormConnector OK')"
