from ceg4n.verifier.base import Verifier, VerifierType
from ceg4n.verifier.esbmc import ESBMCVerifier
from ceg4n.verifier.nnequiv import NNEquivVerifier


class _VerifierProvider:
    def __call__(self, verifier_type: VerifierType) -> Verifier:
        if verifier_type == VerifierType.NNEQUIV:
            return NNEquivVerifier(verifier_type=verifier_type)
        elif verifier_type == VerifierType.ESBMC:
            return ESBMCVerifier(verifier_type=verifier_type)
        elif verifier_type == VerifierType.CBMC:
            raise RuntimeError("Not implemented")
        raise RuntimeError("Invalid Verifier.")


verifier_provider = _VerifierProvider()
