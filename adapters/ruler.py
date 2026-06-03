"""
RULER-style NIAH (Needle In A Haystack) generator.

RULER is a synthetic benchmark — NVIDIA generate tasks programmatically rather
than hosting a fixed dataset. This replicates the NIAH_SINGLE variant:
a unique key=value pair is hidden at a specified depth in a long filler document,
and the model must retrieve the value.

Context lengths and needle depths are configurable so you can plot where
each system starts failing as the document grows.
"""
import random
import uuid

# ~750 words of varied filler sentences drawn from different domains.
# Repeated/shuffled to fill arbitrary target lengths.
_FILLER_SENTENCES = [
    "The history of computing stretches back centuries to mechanical calculation devices.",
    "Transistors replaced vacuum tubes and enabled smaller, more reliable computers.",
    "Neural networks are loosely inspired by the structure of biological brains.",
    "The Pacific Ocean covers more area than all of Earth's landmasses combined.",
    "Plate tectonics explains the slow movement of continents over geological time.",
    "Photosynthesis converts light energy into chemical energy stored in glucose.",
    "The speed of light in a vacuum is approximately 299,792 kilometres per second.",
    "Quantum entanglement allows two particles to share state regardless of distance.",
    "Mitochondria generate most of the cell's supply of ATP, used as energy currency.",
    "The Cambrian explosion saw a rapid diversification of multicellular life forms.",
    "Supply and demand curves intersect at the market equilibrium price.",
    "Compound interest causes wealth to grow exponentially over long time horizons.",
    "The Roman Empire reached its greatest extent under the emperor Trajan.",
    "Cuneiform was one of the earliest writing systems, developed in Mesopotamia.",
    "The Silk Road connected China to the Mediterranean world for over a millennium.",
    "Fourier transforms decompose signals into constituent frequency components.",
    "The Bernoulli principle explains how air pressure differences generate lift.",
    "Continental drift was proposed by Alfred Wegener in the early twentieth century.",
    "DNA replication is semi-conservative: each new strand pairs with an old strand.",
    "Entropy is a measure of disorder or randomness in a thermodynamic system.",
    "The Turing test evaluates whether a machine can exhibit intelligent behaviour.",
    "Graph theory studies the properties of networks of nodes and edges.",
    "Bayes' theorem describes how to update probabilities given new evidence.",
    "The halting problem proved that some questions are undecidable by any algorithm.",
    "Distributed systems must handle network partitions, latency, and node failures.",
    "Convolutional neural networks excel at recognising patterns in grid-like data.",
    "The attention mechanism allows transformers to relate distant tokens in a sequence.",
    "Gradient descent minimises a loss function by iteratively adjusting parameters.",
    "Regularisation techniques prevent overfitting by penalising model complexity.",
    "Cross-entropy loss measures the difference between predicted and true distributions.",
    "Version control systems track changes to files over time and enable collaboration.",
    "Containerisation packages software and its dependencies into isolated units.",
    "The CAP theorem states that distributed systems can satisfy at most two of three guarantees.",
    "Idempotent operations produce the same result regardless of how many times they are applied.",
    "Garbage collection automatically reclaims memory no longer referenced by a program.",
    "Static typing catches type errors at compile time rather than at runtime.",
    "Recursion solves problems by breaking them into smaller instances of the same problem.",
    "Big-O notation describes how algorithm runtime scales with input size.",
    "Hash functions map arbitrary data to fixed-size values for fast lookup.",
    "Public-key cryptography uses mathematically linked key pairs for secure communication.",
    "The Renaissance saw renewed interest in classical art, science, and philosophy.",
    "Johannes Gutenberg's printing press accelerated the spread of knowledge in Europe.",
    "The Industrial Revolution shifted economies from agrarian to manufacturing bases.",
    "Germ theory established that many diseases are caused by microorganisms.",
    "The discovery of penicillin by Alexander Fleming transformed modern medicine.",
    "Black holes are regions where gravity is so strong that nothing can escape.",
    "Dark matter makes up most of the matter in the universe but does not interact with light.",
    "The Hubble constant describes the rate at which the universe is expanding.",
    "Stellar nucleosynthesis produces heavier elements in the cores of dying stars.",
    "Climate feedback loops can amplify or dampen the effects of initial warming.",
]


def generate(
    context_lengths: list[int] | None = None,
    needle_depths: list[float] | None = None,
    n_per_config: int = 3,
    seed: int = 42,
) -> list[dict]:
    """
    Generate NIAH tasks.

    Args:
        context_lengths: Target token counts (1 token ≈ 4 chars).
        needle_depths:   Relative positions (0.0 = start, 1.0 = end).
        n_per_config:    How many tasks per (length, depth) pair.
        seed:            RNG seed for reproducibility.

    Returns:
        List of task dicts in the benchmark task format.
    """
    if context_lengths is None:
        context_lengths = [2_000, 4_000, 8_000]
    if needle_depths is None:
        needle_depths = [0.1, 0.5, 0.9]

    rng = random.Random(seed)
    tasks = []

    for length in context_lengths:
        for depth in needle_depths:
            for n in range(n_per_config):
                needle_key = f"NIAH_{uuid.UUID(int=rng.getrandbits(128)).hex[:8].upper()}"
                needle_val = f"{rng.randint(10000, 99999)}-{rng.choice(['ALPHA','BETA','GAMMA','DELTA','OMEGA'])}"

                filler = _make_filler(length, rng)
                insert_at = int(len(filler) * depth)
                needle_sentence = f"The special identifier {needle_key} has the value {needle_val}."
                context = filler[:insert_at] + " " + needle_sentence + " " + filler[insert_at:]

                prompt = (
                    f"{context}\n\n"
                    f"What is the value associated with the identifier {needle_key}? "
                    f"Reply with only the value, nothing else."
                )

                tasks.append({
                    "id": f"ruler_niah_{length}tok_{int(depth * 100):03d}pct_{n:02d}",
                    "type": "context",
                    "eval": "contains",
                    "source": "ruler_niah",
                    "context_length_tokens": length,
                    "needle_depth": depth,
                    "prompt": prompt,
                    "expected": [needle_val],
                })

    return tasks


def _make_filler(target_tokens: int, rng: random.Random) -> str:
    target_chars = target_tokens * 4  # rough approximation
    sentences = _FILLER_SENTENCES[:]
    rng.shuffle(sentences)

    parts = []
    total = 0
    while total < target_chars:
        rng.shuffle(sentences)
        for s in sentences:
            parts.append(s)
            total += len(s) + 1
            if total >= target_chars:
                break

    return " ".join(parts)
