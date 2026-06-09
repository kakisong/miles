import ray

try:
    from sglang.srt.constants import GPU_MEMORY_TYPE_CUDA_GRAPH, GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_WEIGHTS
except ModuleNotFoundError:
    # SFT-only runs never instantiate SGLang engines. Keep the names available so
    # train.py can be imported from a clean training image without SGLang.
    GPU_MEMORY_TYPE_CUDA_GRAPH = "cuda_graph"
    GPU_MEMORY_TYPE_KV_CACHE = "kv_cache"
    GPU_MEMORY_TYPE_WEIGHTS = "weights"

from miles.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
from miles.utils.arguments import parse_args
from miles.utils.logging_utils import configure_logger
from miles.utils.misc import should_run_periodic_action
from miles.utils.tracking_utils import init_tracking


def train(args):
    configure_logger()
    # allocate the GPUs
    pgs = create_placement_groups(args)
    init_tracking(args)

    # create the rollout manager, with sglang engines inside.
    # need to initialize rollout manager first to calculate num_rollout
    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])

    # create the actor and critic models
    actor_model, critic_model = create_training_models(args, pgs, rollout_manager)

    if args.offload_rollout:
        ray.get(rollout_manager.onload_weights.remote())

    # always update weight first so that sglang has the loaded weights from training.
    actor_model.update_weights()

    if args.check_weight_update_equal:
        ray.get(rollout_manager.check_weights.remote(action="compare"))

    if args.offload_rollout:
        ray.get(rollout_manager.onload_kv.remote())

    # special case for eval-only
    if args.num_rollout == 0 and args.eval_interval is not None:
        ray.get(rollout_manager.eval.remote(rollout_id=0))

    def offload_train():
        if args.offload_train:
            if args.use_critic:
                critic_model.offload()
                if rollout_id >= args.num_critic_only_steps:
                    actor_model.offload()
            else:
                actor_model.offload()
        else:
            actor_model.clear_memory()

    def save(rollout_id):
        if (not args.use_critic) or (rollout_id >= args.num_critic_only_steps):
            actor_model.save_model(
                rollout_id,
                force_sync=rollout_id == args.num_rollout - 1,
            )
        if args.use_critic:
            critic_model.save_model(
                rollout_id,
                force_sync=rollout_id == args.num_rollout - 1,
            )
        if args.rollout_global_dataset:
            ray.get(rollout_manager.save.remote(rollout_id))

    # train loop.
    # note that for async training, one can change the position of the sync operation(ray.get).
    #
    # --async-rollout-prefetch: double-buffer the rollout data generation. The next rollout's
    # data (CPU tokenization/packing on the RolloutManager) is generated WHILE the current step
    # trains on the GPUs, hiding rollout-gen latency behind training instead of exposing it as
    # idle-GPU train_wait. Safe ONLY when generation is weight-INDEPENDENT (SFT): rollout N+1's
    # data does not depend on step N's updated weights. For RL/online rollout this would feed
    # stale-weight generations, so it is gated off by default.
    prefetch = args.async_rollout_prefetch
    if prefetch:
        assert not args.offload_rollout, "--async-rollout-prefetch is incompatible with --offload-rollout"
        assert not args.use_critic, "--async-rollout-prefetch is not supported with --use-critic"
        # prime the pipeline with the first rollout's generation (in flight on the RolloutManager)
        pending_rollout_ref = rollout_manager.generate.remote(args.start_rollout_id)

    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        if args.eval_interval is not None and rollout_id == 0 and not args.skip_eval_before_train:
            ray.get(rollout_manager.eval.remote(rollout_id))

        if prefetch:
            rollout_data_ref = ray.get(pending_rollout_ref)
            # launch the NEXT rollout's data-gen now so it overlaps with async_train() below
            if rollout_id + 1 < args.num_rollout:
                pending_rollout_ref = rollout_manager.generate.remote(rollout_id + 1)
        else:
            rollout_data_ref = ray.get(rollout_manager.generate.remote(rollout_id))

        if args.offload_rollout:
            offload_tags = [GPU_MEMORY_TYPE_CUDA_GRAPH]
            if "kv_cache" in args.offload_rollout_level:
                offload_tags.append(GPU_MEMORY_TYPE_KV_CACHE)
            if "weight" in args.offload_rollout_level:
                offload_tags.append(GPU_MEMORY_TYPE_WEIGHTS)
            ray.get(rollout_manager.offload.remote(tags=offload_tags))

        if args.use_critic:
            critic_train_handle = critic_model.async_train(rollout_id, rollout_data_ref)
            if rollout_id >= args.num_critic_only_steps:
                ray.get(actor_model.async_train(rollout_id, rollout_data_ref))
            ray.get(critic_train_handle)
        else:
            ray.get(actor_model.async_train(rollout_id, rollout_data_ref))

        # Warmup save: trigger one save after the first optimizer step in this process to
        # prime dist_checkpointing's lazy init (pinned-mem pool / IPC handles / CUDA streams)
        # at low iter. Without this, 64K-shape first save at iter ≥ ~50 hits cudaErrorInvalidValue
        # in filesystem_async.py:226 D2H (see examples/deepseek_v4_sft/WRITEUP.md module 3 Problem 8).
        # Must run AFTER actor_model.async_train so precision-aware-optimizer's master_param is
        # populated (otherwise save_checkpoint → KeyError: 'master_param'). Cost: one extra ckpt
        # at iter_<start+1>; --save-retain-interval reaps it on the next regular save.
        is_first_step = rollout_id == args.start_rollout_id
        if is_first_step:
            if (not args.use_critic) or (rollout_id >= args.num_critic_only_steps):
                actor_model.save_model(rollout_id, force_sync=True)
            if args.use_critic:
                critic_model.save_model(rollout_id, force_sync=True)

        if should_run_periodic_action(rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout):
            save(rollout_id)

        offload_train()
        if args.offload_rollout:
            ray.get(rollout_manager.onload_weights.remote())
        actor_model.update_weights()
        if args.offload_rollout:
            ray.get(rollout_manager.onload_kv.remote())

        if should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch):
            ray.get(rollout_manager.eval.remote(rollout_id))

    ray.get(rollout_manager.dispose.remote())


if __name__ == "__main__":
    args = parse_args()
    train(args)
