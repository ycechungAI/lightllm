import torch
import torch.distributed as dist

from lightllm.models.llama.layer_weights.pre_and_post_layer_weight import LlamaPreAndPostLayerWeight
from lightllm.models.llama.infer_struct import LlamaInferStateInfo

from lightllm.models.llama.layer_infer.pre_layer_infer import LlamaPreLayerInfer
from lightllm.utils.infer_utils import mark_cost_time
from lightllm.server.embed_cache.utils import bytes2tensor, read_shm, get_shm_name_embed
from lightllm.common.basemodel.triton_kernel.multimodal_emb import multimodal_emb
from lightllm.distributed.communication_op import all_reduce


"""
infer_state.multimodal_params: batch list of MultimodalParams-dict like:
   {
       "images": [
           {
               "uuid": int,
               "token_id": int, image token start id,
               "token_num": int, image token num,
           },
       ]
       ...
   }
"""


class LlamaMultimodalPreLayerInfer(LlamaPreLayerInfer):
    def __init__(self, network_config, mode):
        super().__init__(network_config, mode)
        return

    def context_forward(self, input_ids, infer_state: LlamaInferStateInfo, layer_weight: LlamaPreAndPostLayerWeight):

        img_weight = []
        img_start_token_ids = []
        img_token_lens = []
        img_start_loc = 0
        img_start_locs = []

        device = layer_weight.wte_weight_.device
        dtype = layer_weight.wte_weight_.dtype
        hidden_size = layer_weight.wte_weight_.shape[1]

        infer_state.mark_multimodal_objs_for_prefill(input_ids=input_ids)

        for batch_id, p in enumerate(infer_state.multimodal_params):
            for img in p["images"] + p["audios"]:
                # skip the same image
                if img["token_id"] in img_start_token_ids or img["_prefill_"] is False:
                    continue
                # pull the img_embeds by uid from shm
                data = read_shm(get_shm_name_embed(img["uuid"]))
                img_weight.append(bytes2tensor(data).cuda().reshape(img["token_num"], -1))
                img_start_token_ids.append(img["token_id"])
                img_token_lens.append(img["token_num"])
                img_start_locs.append(img_start_loc)
                img_start_loc += img["token_num"]
        out = torch.zeros((len(input_ids), hidden_size), dtype=dtype, device=device)
        if len(img_weight) > 0:
            img_weight = torch.cat(img_weight, dim=0).to(device=device, dtype=dtype)
        else:
            img_weight = torch.empty((0, hidden_size), device=device, dtype=dtype)
        assert img_weight.shape[1] == hidden_size, (
            f"Dimension mismatch: text weight dimension is {hidden_size}, "
            f"but image weight dimension is {img_weight.shape[1]}"
        )
        # each tp will fill the img embeds, should divide by world_size
        img_weight = img_weight / self.tp_world_size_
        img_start_token_ids = torch.Tensor(img_start_token_ids).to(device=device, dtype=torch.long)
        img_token_lens = torch.Tensor(img_token_lens).to(device=device, dtype=torch.long)
        img_start_locs = torch.Tensor(img_start_locs).to(device=device, dtype=torch.long)

        multimodal_emb(
            out,
            input_ids,
            layer_weight.wte_weight_,
            img_weight,
            img_token_lens,
            img_start_token_ids,
            img_start_locs,
            self.vob_start_id_,
            self.vob_end_id_,
        )
        if self.tp_world_size_ > 1:
            all_reduce(out, group=infer_state.dist_group, op=dist.ReduceOp.SUM, async_op=False)
        return out
