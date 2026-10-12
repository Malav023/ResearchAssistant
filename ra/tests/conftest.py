import pytest
from ra_core.rag import Paper

@pytest.fixture
def papers():
    return [
        Paper("a1", "Privacy-Preserving Federated Learning at Scale",
              "We study federated learning under differential privacy. Clients train locally and share only noisy gradients. "
              "Our method reduces communication cost by 40 percent. Experiments cover image and text tasks.",
              "Federated learning lets many devices train a shared model without centralising data. "
              "However, gradients can leak private information, e.g. via inversion attacks (Zhu et al. 2019). "
              "We therefore add calibrated noise and secure aggregation."),
        Paper("b2", "Transformers versus State Space Models: An Evaluation",
              "We compare transformer and state space model architectures on long-context benchmarks. "
              "The evaluation setup uses identical training budgets and the same tokenizer. "
              "State space models win on throughput while transformers win on recall.",
              "Attention scales quadratically with sequence length. State space models offer linear scaling. "
              "Prior comparisons used inconsistent evaluation protocols, which makes conclusions unreliable."),
        Paper("c3", "Catastrophic Forgetting in Large Language Models",
              "Large language models forget previously learned knowledge when fine-tuned on new tasks. "
              "We measure this catastrophic forgetting across model sizes. Replay buffers and regularisation reduce the effect.",
              "Continual learning aims to accumulate knowledge over time. Fine-tuning on a narrow dataset overwrites weights "
              "that encoded earlier skills."),
    ]
