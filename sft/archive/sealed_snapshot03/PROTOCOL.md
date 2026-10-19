# runtime03 import contract (producer-owned)

Import `sft.archive.sealed_snapshot03.client.open_snapshot` and `row_digest`.
`open_snapshot(ready, file_ids)` returns `{file_id: OpenedSnapshotFile}`; `.fd` is sealed15 O_RDONLY, `.array` is a readonly ndarray for NPY/catalogs and None for JSONL/C; `.close()` requires caller to discard all exported views first.
Default authentication: strict full response-header/request equality; descriptor seals/access/size; NPY header and exact data extent; complete SHA for both row catalogs. Separate prepublication pinned Docker CPU consumer hashes all seven SFT files via full pread and full mmap streams.
`row_digest(index, tokrow, maskrow)` returns raw32 SHA256, little-endian uint64 index, contiguous uint32 little-endian complete stored token row followed by contiguous uint8 mask row; domain `b'nemochat-sft-row-v1\0'` ends in actual NUL.
Keys: `sft:train-tokens.npy`, `sft:train-masks.npy`, `sft:heldout-tokens.npy`, `sft:heldout-masks.npy`, other original `sft:<basename>`; `rowhash:train`, `rowhash:heldout`; C keys `c:<source>/<split>/corpus.bin` or `.idx`.
READY schema/status are exactly parent contract. Client socket defaults to `ready['socket_path']`; optional `NEMOCHAT_SEALED_SOCKET_PATH` only changes transport path, never identity. Prefer readonly same-absolute socket-dir mount. All peers must be live pinned-image owned-label Docker init/descendants; host same-UID callers are rejected.
No original-path reopen in client. No GPU release here. Original runtime02 mmap remains blocked.
