"""Offline dense training control; not NemoChat's unimplemented native hybrid trainer.

Timings cover data transfer, forward, backward, finite-check kernels and AdamW.
CUDA is synchronized only at window boundaries; diagnostics are read afterwards.
"""
from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import time

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class BenchmarkBlocked(RuntimeError):
    """Requested hardware is unavailable (CLI exit 3)."""


class BenchmarkTimeout(RuntimeError):
    """Cooperative wall-clock budget expired between updates."""


class NonfiniteTraining(RuntimeError):
    """Loss, gradients, or updated parameters failed finite checks."""


def validate_config(config):
    """Use the same bounded contract as the YAML experiment runner."""
    from .experiments import validate_config as validate_experiment_config
    return deepcopy(validate_experiment_config(config))


class CausalBlock(nn.Module):
    """Single-head pre-LN causal attention and GELU feed-forward residuals."""
    def __init__(self, hidden, ffn):
        super().__init__()
        self.attn_norm = nn.LayerNorm(hidden)
        self.qkv = nn.Linear(hidden, 3 * hidden, bias=False)
        self.attn_out = nn.Linear(hidden, hidden, bias=False)
        self.ffn_norm = nn.LayerNorm(hidden)
        self.up = nn.Linear(hidden, ffn, bias=False)
        self.down = nn.Linear(ffn, hidden, bias=False)

    def forward(self, x):
        q, k, v = self.qkv(self.attn_norm(x)).chunk(3, dim=-1)
        attention = F.scaled_dot_product_attention(q.unsqueeze(1), k.unsqueeze(1),
                                                   v.unsqueeze(1), is_causal=True,
                                                   dropout_p=0.0).squeeze(1)
        x = x + self.attn_out(attention)
        return x + self.down(F.gelu(self.up(self.ffn_norm(x))))


class DenseCausalTransformer(nn.Module):
    """Explicit educational dense control with learned positions, untied head."""
    def __init__(self, workload):
        super().__init__()
        w = workload
        self.embedding = nn.Embedding(w['vocab_size'], w['hidden_size'])
        self.position = nn.Embedding(w['sequence_length'], w['hidden_size'])
        self.blocks = nn.ModuleList(CausalBlock(w['hidden_size'], w['ffn_size'])
                                    for _ in range(w['layers']))
        self.norm = nn.LayerNorm(w['hidden_size'])
        self.head = nn.Linear(w['hidden_size'], w['vocab_size'], bias=False)
        self.activation_checkpointing = w['activation_checkpointing']

    def forward(self, tokens):
        x = self.embedding(tokens) + self.position(torch.arange(tokens.shape[1], device=tokens.device))
        for block in self.blocks:
            x = checkpoint(block, x, use_reentrant=False) if self.activation_checkpointing else block(x)
        return self.head(self.norm(x))


def tensor_digest(tensors):
    digest = hashlib.sha256()
    for tensor in tensors:
        digest.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def workload_estimates(w):
    """Projection-only arithmetic, not measured traffic or a full-update FLOP census."""
    rows = []
    tokens, h, f, v = w['batch_size'] * w['sequence_length'], w['hidden_size'], w['ffn_size'], w['vocab_size']
    width = 4 if w['dtype'] == 'float32' else 2
    for name, k, n, count in (('qkv', h, 3*h, w['layers']), ('attention_output', h, h, w['layers']),
                              ('ffn_up', h, f, w['layers']), ('ffn_down', f, h, w['layers']),
                              ('output_head', h, v, 1)):
        flops = 2 * tokens * n * k
        minimum = width * (tokens*k + k*n + tokens*n)
        rows.append({'operator': name, 'M': tokens, 'N': n, 'K': k, 'instances': count,
                     'forward_flops_per_instance': flops, 'minimum_bytes_per_instance': minimum,
                     'arithmetic_intensity_lower_bound_bytes_model': flops / minimum})
    forward = sum(row['forward_flops_per_instance'] * row['instances'] for row in rows)
    recompute = sum(row['forward_flops_per_instance'] * row['instances'] for row in rows if row['operator'] != 'output_head')
    return {'scope': 'analytical projection GEMMs only; FLOPs=2MNK', 'operators': rows,
            'forward_projection_flops_per_update': forward,
            'forward_backward_projection_flops_per_update': 3 * forward,
            'checkpoint_recompute_projection_flops_per_update': recompute if w['activation_checkpointing'] else 0,
            'byte_model': 'lower bound: read each input and weight once, write output once; not measured DRAM traffic',
            'exclusions': ['attention QK/AV', 'softmax', 'embedding', 'layernorm', 'GELU', 'loss', 'optimizer', 'finite checks'],
            'roofline_claim': 'none: needs explicit hardware compute/bandwidth ceilings and measured traffic; no GPU inference from CPU timings'}


def environment(device, threads):
    info = {'python': platform.python_version(), 'torch': str(torch.__version__),
            'cuda_runtime': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
            'platform': platform.platform(), 'machine': platform.machine(),
            'processor': platform.processor(), 'device': str(device), 'cpu_threads': threads,
            'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
            'matmul_precision': torch.get_float32_matmul_precision(),
            'cuda_matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
            'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
            'implementation_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'gpu': None}
    if device.type == 'cuda':
        p = torch.cuda.get_device_properties(device)
        info['gpu'] = {'name': p.name, 'index': device.index, 'compute_capability': [p.major, p.minor],
                       'total_memory_bytes': p.total_memory, 'multiprocessor_count': p.multi_processor_count,
                       'driver_version': 'not queried (no subprocess or environment capture)'}
    return info


def _synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def _step(model, optimizer, batch, device, profiled=False):
    label = torch.profiler.record_function if profiled else lambda _: nullcontext()
    with label('data'):
        inputs = batch[:, :-1].to(device)
        targets = batch[:, 1:].to(device)
    optimizer.zero_grad(set_to_none=True)
    with label('forward'):
        logits = model(inputs)
        loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1))
    with label('backward'):
        loss.backward()
        # Device-side reductions avoid per-step host reads / CUDA synchronization.
        grads = [p.grad for p in model.parameters()]
        if any(g is None for g in grads):
            raise NonfiniteTraining('missing parameter gradient')
        grad_finite = torch.stack([torch.isfinite(g).all() for g in grads]).all()
        grad_norm_sq = torch.stack([g.float().square().sum() for g in grads]).sum()
    with label('optimizer'):
        optimizer.step()
    return loss.detach(), grad_finite.detach(), grad_norm_sq.detach()


def distribution(values):
    return {'values': values, 'min': min(values), 'median': statistics.median(values),
            'mean': statistics.mean(values), 'max': max(values),
            'stdev': statistics.stdev(values) if len(values) > 1 else 0.0}


def run_benchmark(config, output=None):
    """Execute repeated fresh-state updates. Restore caller RNG and CPU thread count.

    max_seconds is cooperative: checked at update boundaries, not a hard kill of
    an in-flight operator. Optional profiling is separate from timing and counted
    against that same budget. Output, if supplied, must be a nonexistent directory.
    """
    config = validate_config(config)
    root = Path(output) if output is not None else None
    if root is not None:
        root.mkdir(parents=True, exist_ok=False)
        (root / 'resolved_config.json').write_text(json.dumps(config, indent=2, allow_nan=False) + '\n')
    result = {'schema_version': 1, 'status': 'running', 'config': config, 'repeats': [],
              'evidence_scope': None, 'correctness': {'status': 'not-established'},
              'scope': 'standalone dense generated-token replay control; not real corpus, model quality, native hybrid, or resume evidence'}
    previous_threads = torch.get_num_threads()
    start = time.perf_counter()
    w, m = config['workload'], config['measurement']
    try:
        if w['device'] == 'cuda' and not torch.cuda.is_available():
            raise BenchmarkBlocked('CUDA unavailable; no CPU substitution')
        device = torch.device('cuda:0' if w['device'] == 'cuda' else 'cpu')
        if device.type == 'cuda' and w['dtype'] == 'bfloat16' and not torch.cuda.is_bf16_supported():
            raise BenchmarkBlocked('CUDA bfloat16 unsupported')
        if device.type == 'cpu':
            torch.set_num_threads(min(previous_threads, 2))
        result['environment'] = environment(device, torch.get_num_threads())
        result['evidence_scope'] = 'GPU-smoke-executed' if device.type == 'cuda' else 'CPU-model-executed'
        result['protocol'] = {'version': 1, 'optimizer': 'AdamW default betas/eps/weight_decay=0.01; foreach=False',
                              'model': 'single-head pre-LN dense causal SDPA; learned positions; GELU; no dropout; untied head',
                              'timing': 'unprofiled full updates, including data/H2D and finite-check kernels; synchronized CUDA window boundaries only',
                              'repeat_state': 'same initialization, AdamW state, generated batch sequence; warmup excluded',
                              'tokens': 'B*S inputs and B*S next-token labels per update; no padding/masking',
                              'dtype': 'parameters/activations use configured dtype; CE and gradient norms float32; no GradScaler',
                              'cpu_memory': 'unavailable (not CUDA allocator memory)',
                              'cuda_memory': 'per-repeat allocator allocated/reserved peaks reset after warmup; not GPU-wide usage',
                              'budget': 'cooperative max_seconds; in-flight operators cannot be interrupted'}
        result['estimates'] = workload_estimates(w)
        generator = torch.Generator(device='cpu').manual_seed(config['experiment']['seed'])
        batches = [torch.randint(w['vocab_size'], (w['batch_size'], w['sequence_length'] + 1), generator=generator)
                   for _ in range(m['warmup_steps'] + m['measured_steps'])]
        result['batch_sha256'] = tensor_digest(batches)
        def budget():
            if time.perf_counter() - start >= config['resources']['max_seconds']:
                raise BenchmarkTimeout('max_seconds expired between updates')
        def fresh():
            # Initialization is on CPU before .to(device); reseed this generator
            # to start every repeat from identical weights, including CUDA runs.
            torch.random.default_generator.manual_seed(config['experiment']['seed'])
            model = DenseCausalTransformer(w).to(device=device, dtype=getattr(torch, w['dtype']))
            optimizer = torch.optim.AdamW(model.parameters(), lr=w['learning_rate'], foreach=False)
            return model, optimizer
        devices = [0] if device.type == 'cuda' else []
        with torch.random.fork_rng(devices=devices):
            for index in range(m['repeats']):
                budget()
                if device.type == 'cuda':
                    # Release unused cached blocks outside timing so previous repeats
                    # do not inflate the reserved-memory high-water mark.
                    torch.cuda.empty_cache()
                model, optimizer = fresh()
                initial = tensor_digest(model.parameters())
                for batch in batches[:m['warmup_steps']]:
                    budget()
                    diagnostics = _step(model, optimizer, batch, device)
                    # Warmup diagnostics are checked outside measured windows.
                    _check_diagnostics([diagnostics])
                _synchronize(device)
                if device.type == 'cuda':
                    torch.cuda.reset_peak_memory_stats(device)
                before = tensor_digest(model.parameters())
                diagnostics = []
                _synchronize(device)
                t0 = time.perf_counter()
                # Retain device-valued diagnostics until the final synchronize.
                # .item()/float()/bool() here would add per-update CUDA syncs
                # and change the full-update timing protocol.
                for batch in batches[m['warmup_steps']:]:
                    budget()
                    diagnostics.append(_step(model, optimizer, batch, device))
                _synchronize(device)
                elapsed = time.perf_counter() - t0
                memory = {'peak_allocated_bytes': None, 'peak_reserved_bytes': None, 'status': 'unavailable-on-CPU'}
                if device.type == 'cuda':
                    memory = {'peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
                              'peak_reserved_bytes': torch.cuda.max_memory_reserved(device), 'status': 'CUDA-allocator-measured'}
                losses, norms = _check_diagnostics(diagnostics)
                if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
                    raise NonfiniteTraining('nonfinite updated parameters')
                final = tensor_digest(model.parameters())
                if before == final:
                    raise NonfiniteTraining('no measured parameter update')
                tokens = m['measured_steps'] * w['batch_size'] * w['sequence_length']
                result['repeats'].append({'repeat': index, 'elapsed_seconds': elapsed,
                    'completed_updates': m['measured_steps'], 'processed_tokens': tokens, 'loss_tokens': tokens,
                    'tokens_per_second': tokens / elapsed, 'updates_per_second': m['measured_steps'] / elapsed,
                    'losses': losses, 'gradient_norms': norms, 'loss_finite': True, 'gradients_finite': True,
                    'parameters_finite': True, 'parameters_changed': True, 'initial_parameter_sha256': initial,
                    'final_parameter_sha256': final, 'memory': memory})
                del diagnostics, model, optimizer
            result['throughput'] = distribution([r['tokens_per_second'] for r in result['repeats']])
            if m['profile']:
                if root is None:
                    raise ValueError('profile requires an output directory')
                budget()
                model, optimizer = fresh()
                activities = [torch.profiler.ProfilerActivity.CPU]
                if device.type == 'cuda':
                    activities.append(torch.profiler.ProfilerActivity.CUDA)
                for batch in batches[:m['warmup_steps']]:
                    budget()
                    _check_diagnostics([_step(model, optimizer, batch, device)])
                _synchronize(device)
                with torch.profiler.profile(activities=activities, record_shapes=True, profile_memory=True) as prof:
                    budget()
                    diagnostics = _step(model, optimizer, batches[m['warmup_steps']], device, profiled=True)
                    _synchronize(device)
                _check_diagnostics([diagnostics])
                prof.export_chrome_trace(str(root / 'trace.json'))
                result['profile'] = {'status': 'captured-separate-from-timing', 'path': 'trace.json',
                                     'updates': 1, 'activities': [a.name for a in activities],
                                     'warning': 'CPU ranges do not establish GPU kernel duration, gaps or critical path'}
            else:
                result['profile'] = {'status': 'disabled'}
        result['correctness'] = {'status': 'finite-updates-checked', 'output_gradient_equivalence': 'not-tested',
                                 'checkpoint_resume': 'not-tested'}
        result['status'] = 'success'
    except BenchmarkBlocked:
        result.update(status='blocked', failure_kind='cuda-unavailable-or-unsupported')
    except BenchmarkTimeout:
        result.update(status='failed', failure_kind='time-budget-exceeded')
    except NonfiniteTraining:
        result.update(status='failed', failure_kind='nonfinite-or-no-update')
    except Exception as exc:
        oom = isinstance(exc, (torch.OutOfMemoryError, MemoryError)) or 'out of memory' in str(exc).lower()
        result.update(status='failed', failure_kind='out-of-memory' if oom else 'execution-error',
                      exception_type=type(exc).__name__)
    finally:
        if torch.get_num_threads() != previous_threads:
            torch.set_num_threads(previous_threads)
    result['total_wall_seconds'] = time.perf_counter() - start
    if root is not None:
        with (root / 'result.json').open('x', encoding='utf-8') as handle:
            json.dump(result, handle, indent=2, allow_nan=False)
            handle.write('\n')
    return result


def _check_diagnostics(diagnostics):
    losses, norms = [], []
    for loss, finite, norm in diagnostics:
        value, norm_value = float(loss), float(norm)
        if not math.isfinite(value) or not bool(finite) or not math.isfinite(norm_value) or norm_value <= 0:
            raise NonfiniteTraining('loss/gradient finite and nonzero checks failed')
        losses.append(value)
        norms.append(math.sqrt(norm_value))
    return losses, norms


def _validated_result(result):
    if not isinstance(result, dict) or type(result.get('schema_version')) is not int or result.get('schema_version') != 1 or result.get('status') != 'success':
        raise ValueError('comparison requires successful schema-1 measurement artifacts')
    c = validate_config(result.get('config'))
    scope = 'CPU-model-executed' if c['workload']['device'] == 'cpu' else 'GPU-smoke-executed'
    correctness = result.get('correctness')
    if not isinstance(correctness, dict) or result.get('evidence_scope') != scope or correctness.get('status') != 'finite-updates-checked':
        raise ValueError('evidence scope or correctness status invalid')
    profile = result.get('profile')
    expected_profile = 'captured-separate-from-timing' if c['measurement']['profile'] else 'disabled'
    if not isinstance(profile, dict) or profile.get('status') != expected_profile:
        raise ValueError('profile evidence inconsistent with configuration')
    for field in ('environment', 'protocol'):
        if not isinstance(result.get(field), dict) or not result[field]:
            raise ValueError('missing environment/protocol')
    def valid_digest(value):
        return isinstance(value, str) and len(value) == 64 and all(ch in '0123456789abcdef' for ch in value)

    # Validate complete descriptors before comparing them: equal omissions do not
    # establish comparability. These user-controlled fields are not authentication.
    env = result['environment']
    required_environment = {'python', 'torch', 'cuda_runtime', 'cudnn', 'platform',
                            'machine', 'processor', 'device', 'cpu_threads',
                            'deterministic_algorithms', 'matmul_precision',
                            'cuda_matmul_allow_tf32', 'cudnn_allow_tf32',
                            'implementation_sha256', 'gpu'}
    if not required_environment <= env.keys():
        raise ValueError('missing required environment fields')
    for key in ('python', 'torch', 'platform', 'machine', 'processor'):
        if not isinstance(env[key], str) or (key != 'processor' and not env[key]):
            raise ValueError(f'invalid environment {key}')
    if not valid_digest(env['implementation_sha256']):
        raise ValueError('invalid implementation identity')
    expected_device = 'cpu' if scope == 'CPU-model-executed' else 'cuda:0'
    if env['device'] != expected_device or type(env['cpu_threads']) is not int or env['cpu_threads'] < 1:
        raise ValueError('environment device/threads inconsistent')
    if scope == 'CPU-model-executed' and env['cpu_threads'] > 2:
        raise ValueError('CPU thread count inconsistent with benchmark policy')
    for key in ('deterministic_algorithms', 'cuda_matmul_allow_tf32', 'cudnn_allow_tf32'):
        if type(env[key]) is not bool:
            raise ValueError(f'invalid environment {key}')
    if not isinstance(env['matmul_precision'], str) or env['matmul_precision'] not in ('highest', 'high', 'medium'):
        raise ValueError('invalid matmul precision policy')
    if env['cuda_runtime'] is not None and (not isinstance(env['cuda_runtime'], str) or not env['cuda_runtime']):
        raise ValueError('invalid CUDA runtime version')
    if env['cudnn'] is not None and (type(env['cudnn']) is not int or env['cudnn'] <= 0):
        raise ValueError('invalid cuDNN version')
    if scope == 'CPU-model-executed':
        if env['gpu'] is not None:
            raise ValueError('CPU artifact cannot claim GPU details')
    else:
        gpu = env['gpu']
        required_gpu = {'name', 'index', 'compute_capability', 'total_memory_bytes',
                        'multiprocessor_count', 'driver_version'}
        if not isinstance(gpu, dict) or not required_gpu <= gpu.keys() or env['cuda_runtime'] is None:
            raise ValueError('missing GPU details or CUDA runtime')
        for key in ('name', 'driver_version'):
            if not isinstance(gpu[key], str) or not gpu[key]:
                raise ValueError(f'invalid GPU {key}')
        if type(gpu['index']) is not int or gpu['index'] != 0:
            raise ValueError('GPU index inconsistent with device')
        capability = gpu['compute_capability']
        if (not isinstance(capability, list) or len(capability) != 2
                or any(type(value) is not int for value in capability)
                or capability[0] <= 0 or capability[1] < 0):
            raise ValueError('invalid GPU compute capability')
        for key in ('total_memory_bytes', 'multiprocessor_count'):
            if type(gpu[key]) is not int or gpu[key] <= 0:
                raise ValueError(f'invalid GPU {key}')
    known_protocol = {
        'version': 1,
        'optimizer': 'AdamW default betas/eps/weight_decay=0.01; foreach=False',
        'model': 'single-head pre-LN dense causal SDPA; learned positions; GELU; no dropout; untied head',
        'timing': 'unprofiled full updates, including data/H2D and finite-check kernels; synchronized CUDA window boundaries only',
        'repeat_state': 'same initialization, AdamW state, generated batch sequence; warmup excluded',
        'tokens': 'B*S inputs and B*S next-token labels per update; no padding/masking',
        'dtype': 'parameters/activations use configured dtype; CE and gradient norms float32; no GradScaler',
        'cpu_memory': 'unavailable (not CUDA allocator memory)',
        'cuda_memory': 'per-repeat allocator allocated/reserved peaks reset after warmup; not GPU-wide usage',
        'budget': 'cooperative max_seconds; in-flight operators cannot be interrupted',
    }
    protocol = result['protocol']
    for key, expected in known_protocol.items():
        if type(protocol.get(key)) is not type(expected) or protocol[key] != expected:
            raise ValueError(f'missing or unsupported protocol {key}')
    if not valid_digest(result.get('batch_sha256')):
        raise ValueError('missing generated batch identity')
    expected_tokens = c['measurement']['measured_steps'] * c['workload']['batch_size'] * c['workload']['sequence_length']
    repeats = result.get('repeats')
    if not isinstance(repeats, list) or len(repeats) != c['measurement']['repeats']:
        raise ValueError('incomplete repeats')
    for i, row in enumerate(repeats):
        if not isinstance(row, dict):
            raise ValueError('repeat must be an object')
        for key in ('initial_parameter_sha256', 'final_parameter_sha256'):
            if not valid_digest(row.get(key)):
                raise ValueError('missing parameter identity')
        if row['initial_parameter_sha256'] != repeats[0]['initial_parameter_sha256']:
            raise ValueError('repeat initial model mismatch')
        if row['initial_parameter_sha256'] == row['final_parameter_sha256']:
            raise ValueError('no parameter change')
        for key, expected in (('repeat', i), ('completed_updates', c['measurement']['measured_steps']),
                              ('processed_tokens', expected_tokens), ('loss_tokens', expected_tokens)):
            if type(row.get(key)) is not int or row[key] != expected:
                raise ValueError('invalid measurement counter')
        for key in ('loss_finite', 'gradients_finite', 'parameters_finite', 'parameters_changed'):
            if row.get(key) is not True:
                raise ValueError('correctness check not passed')
        for key in ('elapsed_seconds', 'tokens_per_second', 'updates_per_second'):
            value = row.get(key)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError('nonfinite/nonpositive measurement')
        if not math.isclose(row['tokens_per_second'], expected_tokens / row['elapsed_seconds'], rel_tol=1e-9):
            raise ValueError('inconsistent token throughput')
        if not math.isclose(row['updates_per_second'], row['completed_updates'] / row['elapsed_seconds'], rel_tol=1e-9):
            raise ValueError('inconsistent update throughput')
        for key in ('losses', 'gradient_norms'):
            values = row.get(key)
            if not isinstance(values, list) or len(values) != c['measurement']['measured_steps']:
                raise ValueError('incomplete diagnostics')
            if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 or (key == 'gradient_norms' and v == 0) for v in values):
                raise ValueError('invalid diagnostics')
        memory = row.get('memory')
        if not isinstance(memory, dict) or set(memory) != {'peak_allocated_bytes', 'peak_reserved_bytes', 'status'}:
            raise ValueError('invalid memory schema')
        for key in ('peak_allocated_bytes', 'peak_reserved_bytes'):
            value = memory.get(key)
            if scope == 'CPU-model-executed':
                if value is not None or memory.get('status') != 'unavailable-on-CPU':
                    raise ValueError('CPU memory must be unavailable')
            elif type(value) is not int or value <= 0 or memory.get('status') != 'CUDA-allocator-measured':
                raise ValueError('invalid CUDA memory')
        if scope == 'GPU-smoke-executed' and memory['peak_reserved_bytes'] < memory['peak_allocated_bytes']:
            raise ValueError('reserved below allocated')
    return c


def compare_results(baseline, candidate):
    """Compatibility-gated descriptive comparison; never an adoption decision."""
    a, b = _validated_result(baseline), _validated_result(candidate)
    wa, wb = deepcopy(a['workload']), deepcopy(b['workload'])
    wa.pop('activation_checkpointing'); wb.pop('activation_checkpointing')
    if wa != wb or a['experiment']['seed'] != b['experiment']['seed']:
        raise ValueError('workload/seed mismatch (only activation_checkpointing may differ)')
    if a['measurement'] != b['measurement']:
        raise ValueError('measurement protocol mismatch')
    for field in ('environment', 'protocol', 'evidence_scope', 'batch_sha256'):
        if not baseline.get(field) or baseline.get(field) != candidate.get(field):
            raise ValueError(f'{field} mismatch or missing')
    if baseline['repeats'][0]['initial_parameter_sha256'] != candidate['repeats'][0]['initial_parameter_sha256']:
        raise ValueError('initial model mismatch')
    av = [r['tokens_per_second'] for r in baseline['repeats']]
    bv = [r['tokens_per_second'] for r in candidate['repeats']]
    memory = {'status': 'unavailable-on-CPU'}
    if a['workload']['device'] == 'cuda':
        memory = {'status': 'CUDA-allocator-only'}
        for key in ('peak_allocated_bytes', 'peak_reserved_bytes'):
            aa = [r['memory'][key] for r in baseline['repeats']]
            bb = [r['memory'][key] for r in candidate['repeats']]
            memory[key] = {'baseline': distribution(aa), 'candidate': distribution(bb),
                           'candidate_minus_baseline_max': max(bb) - max(aa),
                           'candidate_to_baseline_max_ratio': max(bb) / max(aa)}
    return {'schema_version': 1, 'status': 'success', 'evidence_scope': baseline['evidence_scope'],
            'baseline_throughput': distribution(av), 'candidate_throughput': distribution(bv),
            'candidate_to_baseline_median_ratio': statistics.median(bv) / statistics.median(av),
            'paired_repeat_ratios': [y/x for x, y in zip(av, bv)],
            'activation_checkpointing': {'baseline': a['workload']['activation_checkpointing'],
                                         'candidate': b['workload']['activation_checkpointing']},
            'memory_tradeoff': memory, 'decision': 'experimental-only',
            'remaining_gates': ['output and gradient equivalence', 'checkpoint/resume', 'native hybrid full-update integration',
                                'GPU profiling and authorized device validation if only CPU evidence'],
            'interpretation': 'descriptive fixed-order repeats, not statistical significance; speed alone never authorizes adoption'}
