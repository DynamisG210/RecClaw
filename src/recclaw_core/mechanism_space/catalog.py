"""Static provider catalog for versioned mechanism spaces.

Adding a provider is an explicit code change.  There is no entry-point scan,
environment lookup, or caller registration hook.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from recclaw_core.search_spaces.bl_icf_v1.provider import PROVIDER

from .canonical import domain_sha256
from .contracts import MechanismSpaceProviderV1

_PROVIDERS: Mapping[str, MechanismSpaceProviderV1] = MappingProxyType(
    {PROVIDER.identity().search_space_id: PROVIDER}
)


def available_space_ids() -> tuple[str, ...]:
    return tuple(sorted(_PROVIDERS))


def resolve_provider(search_space_id: str) -> MechanismSpaceProviderV1:
    try:
        return _PROVIDERS[search_space_id]
    except KeyError as exc:
        raise KeyError(f"unsupported search space: {search_space_id}") from exc


def catalog_digest() -> str:
    cards = [
        _PROVIDERS[space_id].identity().to_dict()
        for space_id in sorted(_PROVIDERS)
    ]
    return domain_sha256("recclaw.mechanism-space.catalog.v1", cards)


__all__ = ["available_space_ids", "catalog_digest", "resolve_provider"]
