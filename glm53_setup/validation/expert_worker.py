"""Read actual EP placement and loaded parameter shapes in an isolated fixture."""

from .pipeline_worker import PipelineFixtureWorker


class ExpertFixtureWorker(PipelineFixtureWorker):
    def expert_info(self):
        import torch
        from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts

        layers = []
        for name, module in self.get_model().named_modules():
            if not isinstance(module, RoutedExperts):
                continue
            config = module.moe_config
            parallel = config.moe_parallel_config
            mapping = module.expert_map_manager.expert_map
            layers.append(
                {
                    "name": name,
                    "use_ep": parallel.use_ep,
                    "tp_size": parallel.tp_size,
                    "ep_size": parallel.ep_size,
                    "ep_rank": parallel.ep_rank,
                    "dp_size": parallel.dp_size,
                    "sp_size": parallel.sp_size,
                    "use_all2all_kernels": parallel.use_all2all_kernels,
                    "local_experts": module.local_num_experts,
                    "global_experts": config.num_experts,
                    "local_ids": module.expert_map_manager.get_local_expert_ids(),
                    "expert_map": None if mapping is None else mapping.cpu().tolist(),
                    "quant_method": type(module.quant_method).__name__,
                    "experts_kernel": type(
                        module.quant_method.moe_kernel.fused_experts
                    ).__name__,
                    "parameters": {
                        key: {
                            "shape": list(value.shape),
                            "dtype": str(value.dtype),
                            "bytes": value.numel() * value.element_size(),
                        }
                        for key, value in module.named_parameters(recurse=False)
                    },
                }
            )
        if not layers:
            raise ValueError("No routed experts were observed")
        torch.cuda.synchronize()
        return {
            "rank": self.rank,
            "layers": layers,
            "allocated_bytes": torch.cuda.memory_allocated(),
            "reserved_bytes": torch.cuda.memory_reserved(),
        }
