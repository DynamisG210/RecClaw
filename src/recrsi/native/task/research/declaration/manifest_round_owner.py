"""Cold prospective factory: one manifest consumer and native capture per round."""
from campaign_round_owner import DeclarationRoundOwner
from campaign_manifest_consumer import ManifestCampaignConsumer
from native_provider_capture import capture_native_transport

class ManifestRoundOwner(DeclarationRoundOwner):
    @classmethod
    def from_manifest(cls,campaign,provider_module,transport_module,manifest,*,expected_sha256):
        consumer=ManifestCampaignConsumer.from_manifest(manifest,expected_sha256=expected_sha256)
        owner=cls(campaign,consumer,provider_module)
        owner.transport_module=transport_module
        return owner

    def run_once(self):
        with capture_native_transport(self.transport_module):
            return super().run_once()
