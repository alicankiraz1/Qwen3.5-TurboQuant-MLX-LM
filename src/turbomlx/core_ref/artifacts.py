"""Shared reference artefacts keyed by head dimension."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

from .codebooks import CODEBOOK_VERSION, CodebookEntry, CodebookKey, CodebookStore
from .qjl import QJLSpec, load_or_create_qjl
from .rotation import RotationSpec, load_or_create_rotation


@dataclass(slots=True)
class SharedArtifacts:
    codebook: CodebookEntry
    rotation: RotationSpec
    qjl: QJLSpec


@cache
def resolve_shared_artifacts(
    head_dim: int,
    bits: int,
    *,
    codebook_version: str = CODEBOOK_VERSION,
    rotation_kind: str = "qr",
    rotation_seed: int = 0,
    qjl_seed: int = 1,
) -> SharedArtifacts:
    store = CodebookStore()
    key = CodebookKey(head_dim=head_dim, bits=bits, version=codebook_version)
    return SharedArtifacts(
        codebook=store.get_or_create(key),
        rotation=load_or_create_rotation(head_dim, kind=rotation_kind, seed=rotation_seed),
        qjl=load_or_create_qjl(head_dim, seed=qjl_seed),
    )
