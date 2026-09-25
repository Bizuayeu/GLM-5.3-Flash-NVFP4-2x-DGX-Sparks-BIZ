"""LPA's Python hooks run only under eager execution."""


def lpa_execution_supported(config):
    return bool(config.model_config.enforce_eager)
