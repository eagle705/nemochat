"""Run exact selected source methods on CPU/TP1, NOT native framework execution.

Only named AST definitions from committed evidence snapshots are executed. The
BaseLoss pre/post adapter models shape + detach only; no Megatron imports,
process groups, kernels, providers, checkpoints or training loop are exercised.
"""
import ast
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

EVIDENCE = Path(__file__).with_name("evidence")


class _CPUBaseLoss(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self._config = config

    def pre_forward(self, predictions, targets):
        return predictions, targets.detach()

    def post_forward(self, loss, **kwargs):
        return loss.transpose(0, 1).contiguous()


def _definitions():
    namespace = {
        "torch": torch, "F": F, "Tensor": torch.Tensor,
        "BaseLoss": _CPUBaseLoss, "dist": torch.distributed,
        "CACHED_LOGITS_LOGPROB_SENTINEL": -1e3,
        "mtd": SimpleNamespace(DistillationLossBalancer=torch.nn.Module,
                               loss_balancers=SimpleNamespace(STUDENT_LOSS_KEY="student_loss")),
    }
    sources = [
        ("installed__modelopt__torch__distill__plugins__megatron.py", {"LogitsKLLoss", "TopKLogitsKLLoss", "LogitsAndIntermediatesLossBalancer"}),
        ("installed-core__megatron__training__distillation__cached_logits_loss.py", {"topk_kl_div"}),
    ]
    for filename, wanted in sources:
        tree = ast.parse((EVIDENCE / filename).read_text())
        definitions = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in wanted]
        if {node.name for node in definitions} != wanted:
            raise ValueError("required exact source definitions missing")
        exec(compile(ast.Module(body=definitions, type_ignores=[]), filename, "exec"), namespace)
    return namespace


def upstream_token_loss(student, teacher, mode="full", top_k=5):
    scope = _definitions()
    # Source methods expect [sequence, batch, vocab].
    student = student[:, None, :]
    teacher = teacher[:, None, :]
    if mode in ("full", "conditional_topk"):
        config = SimpleNamespace(tensor_model_parallel_size=1)
        cls = scope["LogitsKLLoss" if mode == "full" else "TopKLogitsKLLoss"]
        kwargs = {} if mode == "full" else {"top_k": top_k}
        return cls(config, **kwargs)(student, teacher).reshape(-1)
    if mode not in ("sparse_global", "coarse_tail"):
        raise ValueError("unknown source objective")
    logprobs, ids = torch.log_softmax(teacher, -1).topk(top_k, -1)
    # Native helper subtracts logits_max in-place. Give it a non-leaf copy.
    return scope["topk_kl_div"](student.clone(), logprobs, ids, 1, 0, None,
                                  add_ghost_token=(mode == "coarse_tail")).reshape(-1)


def upstream_balanced_loss(lm_loss, kd_loss, scale=1.):
    scope = _definitions()
    output = scope["LogitsAndIntermediatesLossBalancer"](kd_loss_scale=scale)(
        {"student_loss": lm_loss, "LogitsKLLoss": kd_loss})
    return output["kd_loss"]
