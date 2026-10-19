"""CPU-only raw symbol reference decoding, never launch/caller attribution."""
import unittest

try:
    from dashboard import kernel_symbols
except ImportError:
    kernel_symbols = None


NCCL = 'ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)'
NVJET = 'nvjet_sm103_tss_128x128_64x9_2x2_2cta_h_badd_NTT'


class KernelSymbolsTest(unittest.TestCase):
    def test_exact_contract_and_nccl_collective_algorithm_protocol(self):
        self.assertIsNotNone(kernel_symbols, 'Raw symbol decoder has not been implemented')
        result = kernel_symbols.decode_symbol(NCCL)
        self.assertEqual(set(result), {'family', 'evidence', 'parts', 'warnings'})
        self.assertEqual(result['family'], 'NCCL')
        self.assertEqual(result['evidence'], 'mixed_symbol_reference')
        parts = {part['token']: part for part in result['parts']}
        for part in parts.values():
            self.assertEqual(set(part), {'token', 'label', 'meaning', 'confidence', 'sources'})
            self.assertIn(part['token'], NCCL)
            self.assertTrue(part['label'])
            self.assertTrue(part['meaning'])
        for token in ('AllGather', 'RING', 'LL'):
            self.assertEqual(parts[token]['confidence'], 'documented')
            self.assertTrue(parts[token]['sources'])
        self.assertIn('rank', parts['AllGather']['meaning'])
        self.assertIn('순서', parts['AllGather']['meaning'])
        self.assertIn('알고리즘', parts['RING']['meaning'])
        self.assertIn('프로토콜', parts['LL']['meaning'])
        self.assertTrue(any('Python' in text and '텐서' in text for text in result['warnings']))

    def test_storage_capacity_is_not_tensor_count_and_template_stays_raw(self):
        result = kernel_symbols.decode_symbol(NCCL)
        parts = {part['token']: part for part in result['parts']}
        template = 'ncclDevKernelArgsStorage<(unsigned long)4096>'
        self.assertIn(template, parts)
        self.assertEqual(parts['4096']['confidence'], 'documented')
        self.assertIn('바이트', parts['4096']['meaning'])
        self.assertIn('텐서', parts['4096']['meaning'])
        self.assertIn('capacity', parts[template]['meaning'])
        self.assertTrue(any('디스패치' in text for text in result['warnings']))
        self.assertTrue(any('버전' in text for text in result['warnings']))

    def test_nvjet_documented_backend_but_internal_fields_remain_opaque(self):
        result = kernel_symbols.decode_symbol(NVJET)
        self.assertEqual(result['family'], 'nvjet')
        parts = {part['token']: part for part in result['parts']}
        self.assertEqual(list(parts), NVJET.split('_'))
        self.assertEqual(parts['nvjet']['confidence'], 'documented')
        for token in ('sm103', '128x128', '2cta'):
            self.assertEqual(parts[token]['confidence'], 'inferred')
            self.assertTrue(parts[token]['sources'])
            self.assertIn('확정', parts[token]['meaning'])
        for token in ('tss', '64x9', '2x2', 'h', 'badd', 'NTT'):
            self.assertEqual(parts[token]['confidence'], 'unknown')
            self.assertIn('미확인', parts[token]['meaning'])
        self.assertIn('타일', parts['128x128']['meaning'])
        self.assertTrue(any('명명 규칙' in text for text in result['warnings']))

    def test_protocol_variants_and_storage_demangler_versions(self):
        for protocol in ('LL', 'LL128', 'SIMPLE', 'Simple'):
            with self.subTest(protocol=protocol):
                symbol = f'ncclDevKernel_AllGather_RING_{protocol}(ncclDevKernelArgsStorage<4096ul>)'
                parts = {p['token']: p for p in kernel_symbols.decode_symbol(symbol)['parts']}
                self.assertEqual(parts[protocol]['confidence'], 'documented')
                self.assertIn('프로토콜', parts[protocol]['meaning'])
                self.assertEqual(parts['4096']['confidence'], 'documented')
                self.assertIn('ncclDevKernelArgsStorage<4096ul>', parts)
        for capacity in ('8192', '(unsigned long)8192', '4096UL', '4096u', '4096'):
            symbol = f'ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<{capacity}>)'
            result = kernel_symbols.decode_symbol(symbol)
            template = f'ncclDevKernelArgsStorage<{capacity}>'
            part = next(p for p in result['parts'] if p['token'] == template)
            self.assertEqual(part['confidence'], 'documented')
        symbol = 'ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<64*64>)'
        part = next(p for p in kernel_symbols.decode_symbol(symbol)['parts'] if 'ArgsStorage' in p['token'])
        self.assertEqual(part['confidence'], 'unknown')

    def test_malformed_symbols_fail_closed_without_documented_annotations(self):
        for name in (
            NCCL[:-1], NCCL + ')', NCCL + 'trailer',
            'ncclDevKernel__AllGather_RING_LL(args)',
            'nvjet_sm103_<script>', 'ncclDevKernel_AllGather_RING_LL([)]',
            'ncclDevKernel_AllGather_RING_LL(\x00)', 'nvjet_sm103_\u202eNTT',
        ):
            with self.subTest(name=name):
                result = kernel_symbols.decode_symbol(name)
                self.assertEqual(result['parts'], [])
                self.assertTrue(any('형식' in w for w in result['warnings']))

    def test_html_entities_use_parse_copy_but_tokens_keep_original_spelling(self):
        for symbol in (
            NCCL.replace('<', '&lt;').replace('>', '&gt;'),
            NCCL.replace('<', '&#60;').replace('>', '&#x3e;'),
            NCCL.replace('RING', '&#82;ING'),
        ):
            with self.subTest(symbol=symbol):
                result = kernel_symbols.decode_symbol(symbol)
                self.assertEqual(result['family'], 'NCCL')
                self.assertTrue(all(p['token'] in symbol for p in result['parts']))
                self.assertTrue(any(p['confidence'] == 'documented' and 'capacity' in p['meaning'] for p in result['parts']))
                self.assertTrue(any('HTML' in w for w in result['warnings']))
                if '&#82;' in symbol:
                    part = next(p for p in result['parts'] if p['token'] == '&#82;ING')
                    self.assertEqual(part['confidence'], 'documented')
        result = kernel_symbols.decode_symbol(NCCL.replace('<', '&amp;lt;').replace('>', '&amp;gt;'))
        self.assertFalse(any('capacity' in p['meaning'] for p in result['parts']))

    def test_input_and_complexity_limits_refuse_without_partial_rewrite(self):
        for value in (None, 123, b'nvjet_sm103', [], '', 'x' * 8193,
                      'nvjet_' + '_'.join(['x'] * 65),
                      'ncclDevKernel_' + '_'.join(['x'] * 65),
                      'vendor<' * 17 + 'x' + '>' * 17):
            with self.subTest(kind=type(value).__name__):
                result = kernel_symbols.decode_symbol(value)
                self.assertEqual(result['parts'], [])
                self.assertTrue(any('한도' in w for w in result['warnings']))
        symbol = 'x' * 8192
        result = kernel_symbols.decode_symbol(symbol)
        self.assertEqual(result['parts'][0]['token'], symbol)

    def test_protocol_explanations_distinguish_latency_bandwidth_tradeoff(self):
        for token, word in (('LL', '지연'), ('LL128', '절충'), ('SIMPLE', '대역폭')):
            symbol = f'ncclDevKernel_AllGather_RING_{token}'
            part = next(p for p in kernel_symbols.decode_symbol(symbol)['parts'] if p['token'] == token)
            self.assertIn(word, part['meaning'])
            self.assertIn(kernel_symbols.NCCL_TUNING, part['sources'])

    def test_other_families_unicode_and_old_names_are_safe_unknown(self):
        for symbol in ('cutlass::Kernel<Foo<128, float>>', '한글_커널_１２８',
                       'ncclKernel_AllGather_RING_LL', 'NVJET_sm103_tss',
                       'other_nvjet_sm103', 'vendor_R&#73;NG',
                       'nvjet', '<a href="https://evil.invalid/">x</a>'):
            with self.subTest(symbol=symbol):
                result = kernel_symbols.decode_symbol(symbol)
                self.assertEqual(result['family'], 'other')
                self.assertTrue(all(p['confidence'] == 'unknown' and p['sources'] == [] for p in result['parts']))
                if result['parts']:
                    self.assertEqual(result['parts'][0]['token'], symbol)
                    self.assertTrue(any('지원' in w for w in result['warnings']))
        result = kernel_symbols.decode_symbol('nvjet_sm999_999x999_future')
        parts = {p['token']: p for p in result['parts']}
        self.assertEqual(parts['sm999']['confidence'], 'unknown')
        self.assertEqual(parts['999x999']['confidence'], 'unknown')
        self.assertEqual(parts['future']['confidence'], 'unknown')

    def test_fixed_primary_provenance_cannot_come_from_artifact_or_mutation(self):
        allowed = {kernel_symbols.NCCL_SOURCE, kernel_symbols.NCCL_STORAGE,
                   kernel_symbols.NCCL_COLLECTIVES, kernel_symbols.NCCL_ENV,
                   kernel_symbols.NCCL_COMMON, kernel_symbols.NCCL_TUNING,
                   kernel_symbols.NVJET_REFERENCE, kernel_symbols.PTX_REFERENCE,
                   kernel_symbols.TILE_REFERENCE}
        for symbol in (NCCL, NVJET, 'ncclDevKernel_AllGather_RING_LL(https://evil.invalid/)'):
            result = kernel_symbols.decode_symbol(symbol)
            for part in result['parts']:
                self.assertLessEqual(set(part['sources']), allowed)
                self.assertIn(part['confidence'], {'documented', 'inferred', 'unknown'})
        result = kernel_symbols.decode_symbol(NCCL)
        result['parts'][0]['sources'].append('https://evil.invalid/')
        self.assertNotIn('https://evil.invalid/', kernel_symbols.decode_symbol(NCCL)['parts'][0]['sources'])

    def test_decoder_is_pure_local_work_and_deterministic(self):
        from unittest.mock import patch
        import socket
        with patch('builtins.open', side_effect=AssertionError('No files')), patch.object(socket, 'socket', side_effect=AssertionError('No network')):
            for symbol in (NCCL, NVJET, 'future_kernel'):
                self.assertEqual(kernel_symbols.decode_symbol(symbol), kernel_symbols.decode_symbol(symbol))

    def test_nvjet_tuning_tokens_do_not_establish_physical_launch_metadata(self):
        result = kernel_symbols.decode_symbol(NVJET)
        self.assertTrue(any('2cta' in w and '2x2' in w and 'launch' in w and '클러스터' in w for w in result['warnings']))
        self.assertNotIn('229760', str(result))
        self.assertNotIn('255/thread', str(result))


if __name__ == '__main__':
    unittest.main()
