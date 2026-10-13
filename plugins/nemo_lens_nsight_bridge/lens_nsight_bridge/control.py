"""Readable single-process sparse MoE; NOT native fused Nemotron or EP."""
import torch
import torch.nn.functional as F


def routing(x, router):
    return x @ router


def topk(logits, k):
    scores, experts = torch.topk(logits, k, dim=-1)
    return scores.softmax(dim=-1), experts


def permutation(experts, weights):
    tokens = torch.arange(experts.shape[0], device=experts.device).repeat_interleave(experts.shape[1])
    order = torch.argsort(experts.flatten(), stable=True)
    return tokens[order], experts.flatten()[order], weights.flatten()[order]


def dispatch(x, tokens, experts, expert):
    mask = experts == expert
    return x[tokens[mask]], tokens[mask], mask


def expert_up(x, weight):
    return x @ weight


def activation(up):
    gate, value = up.chunk(2, dim=-1)
    return F.silu(gate) * value


def expert_down(hidden, weight):
    return hidden @ weight


def combine(outputs, token_ids, weights, masks, x):
    result = torch.zeros_like(x)
    for out, ids, mask in zip(outputs, token_ids, masks):
        result = result.index_add(0, ids, out * weights[mask, None])
    return result


def backward(loss):
    loss.backward()


def run(bridge, device="cpu"):
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no working GPU is available")
    torch.manual_seed(7)
    tokens, width, hidden, count, k = 8, 16, 24, 4, 2
    x = torch.randn(tokens, width, device=device, requires_grad=True)
    router = (torch.randn(width, count, device=device) * 0.1).requires_grad_()
    up = (torch.randn(count, width, 2*hidden, device=device) * 0.1).requires_grad_()
    down = (torch.randn(count, hidden, width, device=device) * 0.1).requires_grad_()
    logits = bridge.call("routing", routing, x, router)
    weights, experts = bridge.call("topk", topk, logits, k)
    ids, sorted_experts, sorted_weights = bridge.call("permutation", permutation, experts, weights)
    outputs, token_ids, masks = [], [], []
    for expert in range(count):
        batch, selected_ids, mask = bridge.call("dispatch", dispatch, x, ids, sorted_experts, expert)
        expanded = bridge.call("expert_up", expert_up, batch, up[expert])
        active = bridge.call("activation", activation, expanded)
        out = bridge.call("expert_down", expert_down, active, down[expert])
        outputs.append(out)
        token_ids.append(selected_ids)
        masks.append(mask)
    y = bridge.call("combine", combine, outputs, token_ids, sorted_weights, masks, x)
    # Independent token-major reference checks the dispatch/permutation/combine path.
    reference = []
    for token in range(tokens):
        value = torch.zeros(width, device=device)
        for slot in range(k):
            expert = experts[token, slot]
            g, v = (x[token] @ up[expert]).chunk(2)
            value = value + weights[token, slot] * ((F.silu(g) * v) @ down[expert])
        reference.append(value)
    expected = torch.stack(reference)
    torch.testing.assert_close(y, expected, rtol=1e-5, atol=1e-6)
    actual_grads = torch.autograd.grad(y.square().mean(), (x, router, up, down), retain_graph=True)
    # Both reference and dispatched graphs share routing weights; retain that
    # common subgraph until the explicitly instrumented backward entry below.
    reference_grads = torch.autograd.grad(expected.square().mean(), (x, router, up, down), retain_graph=True)
    for actual, expected_grad in zip(actual_grads, reference_grads):
        torch.testing.assert_close(actual, expected_grad, rtol=1e-4, atol=1e-6)
    loss = y.square().mean()
    bridge.call("backward", backward, loss)
    if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in (x, router, up, down)):
        raise RuntimeError("Nonfinite/missing gradient")
    if device == "cuda":
        torch.cuda.synchronize()
    return {"scope": "educational_sparse_moe_not_native_fused_nemotron", "device": device,
            "torch": torch.__version__, "tokens": tokens, "experts": count, "top_k": k,
            "assignments": tokens*k, "loss": loss.item(), "forward_reference": "passed",
            "gradient_reference": "passed", "finite_gradients": True,
            "distributed_ep": False, "gpu_kernel_mapping": "unverified"}
