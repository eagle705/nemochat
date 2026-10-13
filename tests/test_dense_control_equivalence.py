"""CPU numerical checks for the standalone dense control, not native resume."""
from copy import deepcopy
import torch
from nemochat.training_perf import CausalBlock, DenseCausalTransformer
from torch.utils.checkpoint import checkpoint


def test_checkpointed_block_input_gradients_match():
    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with torch.random.fork_rng():
            torch.manual_seed(91)
            reference = CausalBlock(8, 16)
            candidate = deepcopy(reference)
            x = torch.randn(2, 7, 8, requires_grad=True)
            y = x.detach().clone().requires_grad_(True)
            out = reference(x)
            other = checkpoint(candidate, y, use_reentrant=False)
            torch.testing.assert_close(other, out, rtol=0, atol=0)
            out.square().mean().backward()
            other.square().mean().backward()
            torch.testing.assert_close(y.grad, x.grad, rtol=1e-6, atol=1e-7)
    finally:
        torch.set_num_threads(previous)


def workload():
    return dict(vocab_size=17, hidden_size=8, sequence_length=7, ffn_size=16,
                layers=2, activation_checkpointing=False)


def test_checkpointed_outputs_parameter_gradients_and_optimizer_update_match():
    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with torch.random.fork_rng():
            torch.manual_seed(123)
            baseline = DenseCausalTransformer(workload())
            candidate = deepcopy(baseline)
            candidate.activation_checkpointing = True
            tokens = torch.randint(0, 17, (2, 7))
            expected = baseline(tokens)
            actual = candidate(tokens)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            expected.square().mean().backward()
            actual.square().mean().backward()
            for (name, ref), (other_name, modified) in zip(baseline.named_parameters(), candidate.named_parameters()):
                assert name == other_name
                assert ref.grad is not None and modified.grad is not None
                torch.testing.assert_close(modified.grad, ref.grad, rtol=1e-6, atol=1e-7)
            a = torch.optim.AdamW(baseline.parameters(), lr=0.001, foreach=False)
            b = torch.optim.AdamW(candidate.parameters(), lr=0.001, foreach=False)
            a.step(); b.step()
            for ref, modified in zip(baseline.parameters(), candidate.parameters()):
                torch.testing.assert_close(modified, ref, rtol=1e-6, atol=1e-7)
    finally:
        torch.set_num_threads(previous)


def test_future_token_cannot_change_prefix_logits():
    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with torch.random.fork_rng():
            torch.manual_seed(19)
            model = DenseCausalTransformer(workload())
            tokens = torch.randint(0, 17, (1, 7))
            altered = tokens.clone(); altered[:, 4:] = (altered[:, 4:] + 1) % 17
            original = model(tokens)
            changed = model(altered)
            torch.testing.assert_close(changed[:, :4], original[:, :4], rtol=0, atol=0)
    finally:
        torch.set_num_threads(previous)
