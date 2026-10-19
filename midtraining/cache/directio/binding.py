"""Version-bound, process-local native opener injection; codec is never replaced."""
import ast
import hashlib
import inspect
from pathlib import Path
import textwrap

NAMES = ('open_logit_file', '_verify_logprobs_metadata', 'iter_logprobs_tar_entries',
         'unpack_indices', 'decode_logprobs_payload')


def ast_digest(node):
    return hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()


def source_contract(path):
    data = Path(path).read_bytes()
    tree = ast.parse(data)
    hashes = {n.name: ast_digest(n) for n in tree.body
              if isinstance(n, ast.FunctionDef) and n.name in NAMES}
    if set(hashes) != set(NAMES):
        raise RuntimeError('native source functions missing')
    return {'source_sha256': hashlib.sha256(data).hexdigest(), 'function_ast_sha256': hashes}


def install(module, contract, transport):
    """Only the exact original native opener may be replaced, once per process."""
    actual = source_contract(module.__file__)
    if actual != contract:
        raise RuntimeError('native source version mismatch')
    original = module.open_logit_file
    if not inspect.isfunction(original) or original.__name__ != 'open_logit_file':
        raise RuntimeError('native opener is no longer original')
    node = ast.parse(textwrap.dedent(inspect.getsource(original))).body[0]
    if ast_digest(node) != contract['function_ast_sha256']['open_logit_file']:
        raise RuntimeError('native open function AST mismatch')
    # A mutable callable can retain its source location; compare compiled code too.
    ns = {}; exec(compile(ast.Module(body=[node], type_ignores=[]), '<guard>', 'exec'), ns)
    fresh = ns['open_logit_file'].__code__
    if (original.__code__.co_code, original.__code__.co_consts) != (fresh.co_code, fresh.co_consts):
        raise RuntimeError('native open function bytecode mismatch')
    module.open_logit_file = transport.open
    return original
