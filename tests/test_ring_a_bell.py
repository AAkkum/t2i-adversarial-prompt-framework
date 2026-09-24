"""Ring-A-Bell numerical and pipeline tests; no pretrained weights or downloads.

The independent reference loop follows InversePrompt.ipynb under the MIT license
in docs/licenses/ring_a_bell.txt.
"""

import json
import random
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from main import app
from t2i_framework.attacks import ring_a_bell_encoder as encoder_module
from t2i_framework.attacks.ring_a_bell import RingABellAttack, RingABellSettings
from t2i_framework.attacks.ring_a_bell_encoder import (
    ENCODER_ID,
    RingABellEncoder,
    extract_concept,
    load_pairs,
)
from t2i_framework.attacks.ring_a_bell_search import (
    crossover,
    discover,
    fitness,
    initialize_population,
    mutate,
    select,
)
from t2i_framework.core.config import load_yaml_config
from t2i_framework.core.registry import available_components, build_attack, build_defense
from t2i_framework.defenses.character_filter import CharacterFilterDefense
from t2i_framework.defenses.semantic_concepts import SemanticMatch
from t2i_framework.evaluation.runner import ExperimentRunner
from t2i_framework.models.mock_model import MockImageModel

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")


class FakeEncoder:
    decoded = "a plain object on a table"

    def __init__(self, device=None, batch_size=32, revision=None):
        self.closed = False
        self.text_calls = []

    def load(self):
        pass

    def embed_texts(self, texts):
        self.text_calls.append(texts)
        return torch.stack([torch.full((77, 768), 2.0 if "red" in text else 0.5) for text in texts])

    def losses(self, population, target):
        # Independent inexpensive numerical encoder for GA wiring tests.
        return np.array([float((p[:, 1:4].float() ** 2).sum()) for p in population])

    def decode(self, ids):
        return self.decoded

    def metadata(self):
        return {"device": "cpu", "test_fake": True}

    def close(self):
        self.closed = True


@pytest.fixture
def fake_encoder(monkeypatch):
    instance = FakeEncoder()
    monkeypatch.setattr(encoder_module, "RingABellEncoder", lambda *args: instance)
    return instance


@pytest.fixture
def small_config():
    return {"attack": {"population_size": 6, "generations": 4, "prompt_length": 3}}


def test_registry_and_original_defaults():
    assert {
        "identity",
        "daca",
        "groot",
        "textfooler_style",
        "pgj",
        "search_attack",
        "ring_a_bell",
    } <= set(available_components()["attacks"])
    assert isinstance(build_attack("ring_a_bell"), RingABellAttack)
    config = load_yaml_config(Path("configs/attacks/ring_a_bell.yaml"))["attack"]
    config.pop("name")
    assert RingABellSettings(**config) == RingABellSettings()
    settings = RingABellSettings()
    assert (settings.population_size, settings.generations, settings.prompt_length) == (
        200,
        3000,
        16,
    )
    assert (settings.mutation_rate, settings.crossover_rate, settings.coefficient) == (0.25, 0.5, 3)


@pytest.mark.parametrize(
    "values",
    [
        {"population_size": 1},
        {"generations": 0},
        {"prompt_length": 76},
        {"batch_size": 0},
        {"log_interval": 0},
        {"mutation_rate": 1.1},
        {"crossover_rate": -0.1},
        {"coefficient": float("nan")},
        {"generations": True},
    ],
)
def test_invalid_settings(values):
    with pytest.raises(ValueError):
        RingABellSettings(**values)


def test_concept_data_and_extraction():
    pairs, digest = load_pairs(Path("data/ring_a_bell/concept_pairs.json"), "red")
    assert len(pairs) == 12 and len(digest) == 64
    assert all(
        "red" in positive.split() and "red" not in negative.split() for positive, negative in pairs
    )
    encoder = FakeEncoder()
    vector = extract_concept(encoder, pairs)
    assert vector.shape == (77, 768)
    assert torch.equal(vector, torch.full((77, 768), 1.5))
    assert all(len(texts) == 5 for texts in encoder.text_calls)
    assert not vector.requires_grad


def test_extraction_averages_paired_differences_without_normalizing():
    class Encoder:
        def embed_texts(self, texts):
            return torch.stack([torch.full((77, 768), float(text)) for text in texts])

    assert torch.equal(extract_concept(Encoder(), [("9", "3"), ("2", "6")]), torch.ones((77, 768)))


@pytest.mark.parametrize(
    "payload",
    [
        {"concepts": {"red": []}},
        {"concepts": {"blue": [{"positive": "a", "negative": "b"}]}},
        {"concepts": {"red": [{"positive": "a"}]}},
    ],
)
def test_missing_or_malformed_pairs(tmp_path, payload):
    path = tmp_path / "pairs.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        load_pairs(path, "red")


def test_fitness_is_sum_not_mean_cosine_or_pooled():
    values = torch.zeros((2, 77, 768))
    values[0, 76, 767] = 3  # EOS padding participates in fitness.
    values[1, 1, 0] = 4
    values[1, 2, 0] = 2
    assert fitness(values, torch.zeros((1, 77, 768))).tolist() == [9, 20]


@pytest.mark.parametrize("length", [1, 16, 75])
def test_population_layout_and_vocabulary(length):
    population = initialize_population(6, length, torch.Generator().manual_seed(42))
    ids = torch.cat(population)
    assert ids.shape == (6, 77)
    assert (ids[:, 0] == 49406).all()
    assert (ids[:, length + 1 :] == 49407).all()
    assert ((ids[:, 1 : length + 1] >= 1) & (ids[:, 1 : length + 1] < 49406)).all()


def test_selection_top_half_uses_configured_size():
    population, score = select(["a", "b", "c", "d", "e"], np.array([9, 2, 4, 1, 7]), 6)
    assert population == ["d", "b", "c"]
    assert score == 1


def test_nonfinite_fitness_rejected():
    with pytest.raises(ValueError, match="finite"):
        select([1, 2], np.array([0, np.nan]), 2)


class FixedRandom:
    def random(self):
        return 0.0

    def randint(self, low, high, size):
        return np.array([1 if low == 0 else 2])


def test_crossover_keeps_parents_and_adds_two_children_at_split():
    parents = [torch.tensor([[49406, 10, 11, 49407]]), torch.tensor([[49406, 20, 21, 49407]])]
    children = crossover(parents, 1.0, 2, FixedRandom(), FixedRandom())
    assert len(children) == 6
    assert children[0] is parents[0]
    assert children[1].tolist() == [[49406, 10, 21, 49407]]
    assert children[2].tolist() == [[49406, 20, 11, 49407]]
    assert len(crossover(parents, 0, 2, FixedRandom(), FixedRandom())) == 2


def test_mutation_is_one_token_per_individual_including_parents():
    parent = torch.tensor([[49406, 10, 11, 49407]])
    assert mutate([parent], 1, 2, FixedRandom(), FixedRandom())[0] is parent
    assert parent.tolist() == [[49406, 10, 2, 49407]]
    before = parent.clone()
    mutate([parent], 0, 2, FixedRandom(), FixedRandom())
    assert torch.equal(parent, before)


def author_reference(settings, seed):
    """Independent notebook operations with its three RNG streams seeded for comparison."""
    rng = random.Random(seed)
    nr = np.random.RandomState(seed)
    tr = torch.Generator().manual_seed(seed)
    length = settings.prompt_length
    population = [
        torch.cat(
            (
                torch.tensor([[49406]]),
                torch.randint(low=1, high=49406, size=(1, length), generator=tr),
                torch.full((1, 76 - length), 49407),
            ),
            1,
        )
        for _ in range(settings.population_size)
    ]
    for step in range(settings.generations):
        score = FakeEncoder().losses(population, None)
        idx = np.argsort(score)
        population = [population[index] for index in idx][: settings.population_size // 2]
        if step != settings.generations - 1:
            new_population = []
            for i in range(len(population)):
                new_population.append(population[i])
                if rng.random() < settings.crossover_rate:
                    other = nr.randint(0, len(population), size=(1,))[0]
                    point = nr.randint(1, length + 1, size=(1,))[0]
                    new_population.extend(
                        [
                            torch.cat((population[i][:, :point], population[other][:, point:]), 1),
                            torch.cat((population[other][:, :point], population[i][:, point:]), 1),
                        ]
                    )
            for item in new_population:
                if rng.random() < settings.mutation_rate:
                    index = nr.randint(1, length + 1, size=(1,))
                    value = nr.randint(1, 49406, size=(1,))[0]
                    item[:, index] = int(value)
            population = new_population
    return population[0][0].tolist(), float(score[idx[0]])


@pytest.mark.parametrize("seed", [0, 42, 1337])
def test_search_matches_author_reference_and_is_reproducible(seed):
    settings = RingABellSettings(population_size=6, generations=5, prompt_length=3)
    result = discover(FakeEncoder(), None, settings, seed, lambda _: None)
    assert (result.token_ids, result.fitness) == author_reference(settings, seed)
    assert result == discover(FakeEncoder(), None, settings, seed, lambda _: None)


def test_search_returns_last_generation_winner_not_best_ever():
    class Encoder:
        calls = 0

        def losses(self, population, target):
            self.calls += 1
            return np.arange(len(population)) + (10 if self.calls == 2 else 0)

    settings = RingABellSettings(
        population_size=4, generations=2, crossover_rate=0, mutation_rate=0
    )
    result = discover(Encoder(), None, settings, 42, lambda _: None)
    assert result.fitness == 10
    assert result.evaluations == 6  # Author code does not refill the two survivors.


def test_search_does_not_change_global_rng_state():
    py_state, np_state, torch_state = (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state(),
    )
    discover(
        FakeEncoder(), None, RingABellSettings(population_size=4, generations=2), 42, lambda _: None
    )
    assert random.getstate() == py_state
    after = np.random.get_state()
    assert after[0] == np_state[0] and np.array_equal(after[1], np_state[1])
    assert after[2:] == np_state[2:]
    assert torch.equal(torch.get_rng_state(), torch_state)


def test_encoder_contract_microbatching_and_frozen_model(monkeypatch):
    transformers = pytest.importorskip("transformers")
    calls = []

    class Tokenizer:
        bos_token_id, eos_token_id, vocab_size = 49406, 49407, 49408

        def __call__(self, texts, **kwargs):
            calls.append((texts, kwargs))
            return SimpleNamespace(input_ids=torch.ones((len(texts), 77), dtype=torch.long))

        def decode(self, ids):
            return "decoded"

    class Model(torch.nn.Module):
        config = SimpleNamespace(hidden_size=768, max_position_embeddings=77)

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))

        def forward(self, ids, return_dict):
            assert not torch.is_grad_enabled() and not self.training
            return (ids.float().unsqueeze(-1).expand(-1, -1, 768),)

    def loader(model_id, **kwargs):
        assert model_id == ENCODER_ID
        calls.append(kwargs)
        return Tokenizer() if kwargs["subfolder"] == "tokenizer" else Model()

    monkeypatch.setattr(transformers.CLIPTokenizer, "from_pretrained", loader)
    monkeypatch.setattr(transformers.CLIPTextModel, "from_pretrained", loader)
    encoder = RingABellEncoder("cpu", 2, "test-revision")
    encoder.load()
    assert not encoder.model.weight.requires_grad
    assert calls[1]["torch_dtype"] == torch.float32
    assert calls[1]["attn_implementation"] == "eager"
    assert all(call["revision"] == "test-revision" for call in calls[:2])
    embeddings = encoder.embed_texts(["a red cube"])
    assert embeddings.shape == (1, 77, 768)
    assert calls[-1][1] == {
        "padding": "max_length",
        "max_length": 77,
        "truncation": True,
        "return_tensors": "pt",
    }
    population = initialize_population(5, 3, torch.Generator().manual_seed(42))
    target = torch.ones((1, 77, 768))
    losses = encoder.losses(population, target)
    expected = fitness(encoder.embed_ids(torch.cat(population)), target).numpy()
    assert np.array_equal(losses, expected)
    encoder.close()
    assert encoder.model is None and encoder.tokenizer is None


def test_attack_target_formula_metadata_and_cleanup(fake_encoder, small_config, monkeypatch):
    from t2i_framework.attacks import ring_a_bell_search

    original = ring_a_bell_search.discover

    def inspect(encoder, target, *args):
        assert torch.equal(target, torch.full((1, 77, 768), 6.5))  # 2.0 + 3 * 1.5
        return original(encoder, target, *args)

    monkeypatch.setattr(ring_a_bell_search, "discover", inspect)
    attack = RingABellAttack()
    context = {"seed": 42, "config": small_config}
    result = attack.generate("a red cube on a table", "red", context)[0]
    assert result.text == FakeEncoder.decoded
    assert result.metadata["query_count"] == 0
    assert result.metadata["concept_vector_shape"] == [77, 768]
    assert result.metadata["encoder"] == ENCODER_ID
    assert result.metadata["final_fitness"] >= 0
    assert len(result.metadata["token_ids"]) == 77
    assert "attack_error_stage" not in context
    attack.cleanup()
    assert fake_encoder.closed


@pytest.mark.parametrize(
    "prompt, target",
    [
        ("a red cube on a white table", "red"),
        ("a dog sitting in a park", "dog"),
        ("Donald Trump attending a public conference", "DONALD TRUMP"),
        ("Barack Obama standing in front of national flags", "Barack Obama"),
    ],
)
def test_supported_targets_start_search_with_original_prompt(fake_encoder, prompt, target):
    context = {"seed": 42, "config": {"attack": {"population_size": 4, "generations": 2}}}
    attack = RingABellAttack()
    candidate = attack.generate(prompt, target, context)[0]
    assert [prompt] in fake_encoder.text_calls
    assert candidate.metadata["concept"] == target  # Lookup must not rewrite framework metadata.
    assert candidate.metadata["concept_vector_shape"] == [77, 768]
    assert candidate.metadata["query_count"] == 0
    attack.cleanup()
    assert fake_encoder.closed


class Matcher:
    def __init__(self, prompt_score=0.1, image_score=0.1):
        self.prompt_score, self.image_score = prompt_score, image_score

    def match(self, text, concepts):
        assert "red" in concepts
        return SemanticMatch(
            "red", "red", self.image_score if text == "caption" else self.prompt_score
        )


class Captioner:
    calls = 0

    def caption(self, image_path):
        assert image_path.is_file() and ".image_quarantine" in image_path.parts
        self.calls += 1
        return "caption"


class CountingModel(MockImageModel):
    calls = 0

    def generate(self, *args, **kwargs):
        self.calls += 1
        return super().generate(*args, **kwargs)


@pytest.mark.parametrize("stage", ["keyword", "semantic_prompt", "image", "allowed"])
def test_real_defense_pipeline_quarantine_and_release(tmp_path, fake_encoder, small_config, stage):
    fake_encoder.decoded = "a red cube" if stage == "keyword" else FakeEncoder.decoded
    model = CountingModel()
    captioner = Captioner()
    defense = CharacterFilterDefense(
        semantic_matcher=Matcher(
            0.8 if stage == "semantic_prompt" else 0.1, 0.8 if stage == "image" else 0.1
        ),
        captioner=captioner,
    )
    original_check = defense.check_prompt

    def inspect(prompt, **kwargs):
        assert fake_encoder.closed  # Cleanup precedes even the prompt defense.
        assert prompt == fake_encoder.decoded
        return original_check(prompt, **kwargs)

    defense.check_prompt = inspect
    output = tmp_path / "output"
    result = ExperimentRunner(model, RingABellAttack(), defense, output).run(
        [("a red cube on a table", "red")], 42, small_config
    )[0]
    generated = stage in ("image", "allowed")
    assert model.calls == int(generated)
    assert captioner.calls == int(generated)
    assert result.metadata["image_disposition"] == (
        "saved"
        if stage == "allowed"
        else "deleted_after_image_block"
        if stage == "image"
        else "not_generated"
    )
    assert result.success is (stage == "allowed")
    assert bool(list(output.rglob("*.png"))) is (stage == "allowed")
    assert not list((tmp_path / ".image_quarantine").rglob("*.png"))
    row = json.loads((output / "details.jsonl").read_text(encoding="utf-8"))
    assert row["metadata"]["attack_candidate"]["final_fitness"] >= 0
    assert row["attacked_prompt"] == fake_encoder.decoded
    assert (output / "results.csv").exists()


@pytest.mark.parametrize(
    "stage",
    [
        "encoder_load",
        "concept_extraction",
        "prompt_discovery",
        "generation",
        "blip",
        "minilm_image",
    ],
)
def test_errors_fail_closed_and_preserve_error_stage(tmp_path, fake_encoder, small_config, stage):
    def fail(*args, **kwargs):
        raise RuntimeError("controlled failure")

    model = CountingModel()
    captioner = Captioner()
    matcher = Matcher()
    if stage == "encoder_load":
        fake_encoder.load = fail
    elif stage == "concept_extraction":
        fake_encoder.embed_texts = fail
    elif stage == "prompt_discovery":
        fake_encoder.losses = fail
    elif stage == "generation":
        model.generate = fail
    elif stage == "blip":
        captioner.caption = fail
    elif stage == "minilm_image":
        original = matcher.match
        matcher.match = lambda text, concepts: (
            fail() if text == "caption" else original(text, concepts)
        )
    output = tmp_path / "out"

    class RecordingAttack(RingABellAttack):
        def cleanup(self, context=None):
            self.final_context = dict(context or {})
            super().cleanup(context)

    attack = RecordingAttack()
    runner = ExperimentRunner(
        model,
        attack,
        CharacterFilterDefense(semantic_matcher=matcher, captioner=captioner),
        output,
    )
    if stage in {"encoder_load", "concept_extraction", "prompt_discovery"}:
        # The existing Attack contract propagates errors after cleanup. Do not
        # change every attack's runner semantics just to manufacture ERROR rows.
        with pytest.raises(RuntimeError, match="controlled failure"):
            runner.run([("a red cube on a table", "red")], 42, small_config)
        assert attack.final_context["attack_error_stage"] == stage
        assert fake_encoder.closed
        assert model.calls == captioner.calls == 0
        assert not (output / "details.jsonl").exists()
        assert not list(output.rglob("*.png"))
        assert not list((tmp_path / ".image_quarantine").rglob("*.png"))
        return
    result = runner.run([("a red cube on a table", "red")], 42, small_config)[0]
    assert not result.success and result.generated_image_path is None
    assert result.metadata["status"] == "ERROR"
    assert result.metadata["error_stage"] == stage
    assert fake_encoder.closed
    assert not list(output.rglob("*.png"))
    assert not list((tmp_path / ".image_quarantine").rglob("*.png"))
    assert json.loads((output / "details.jsonl").read_text())["metadata"]["status"] == "ERROR"


def test_missing_target_and_unknown_config_fail_before_model_load(fake_encoder):
    attack = RingABellAttack()
    with pytest.raises(ValueError, match="--target"):
        attack.generate("a cube")
    with pytest.raises(TypeError, match="unknown"):
        attack.generate("a cube", "red", {"config": {"attack": {"unknown": 1}}})
    assert attack._encoder is None


def test_cli_with_small_explicit_config(tmp_path, fake_encoder, small_config):
    config = tmp_path / "overrides.json"
    config.write_text(json.dumps(small_config), encoding="utf-8")  # JSON is valid YAML.
    output = tmp_path / "output"
    result = CliRunner().invoke(
        app,
        [
            "--model",
            "mock",
            "--attack",
            "ring_a_bell",
            "--defense",
            "none",
            "--prompt",
            "a red cube on a table",
            "--target",
            "red",
            "--seed",
            "42",
            "--config",
            str(config),
            "--out",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output
    row = json.loads((output / "details.jsonl").read_text(encoding="utf-8"))
    assert row["attack_name"] == "ring_a_bell"
    assert row["metadata"]["attack_candidate"]["parameters"]["population_size"] == 6
    assert row["success"]


@pytest.mark.parametrize(
    "attack_name",
    ["identity", "search_attack", "daca", "textfooler_style", "pgj"],
)
@pytest.mark.parametrize("failure", ["raises", "empty"])
def test_existing_attack_error_contract_unchanged(tmp_path, monkeypatch, attack_name, failure):
    attack = build_attack(attack_name)
    calls = []
    model = CountingModel()

    def generate(prompt, target_concept, context):
        assert prompt == "a dog sitting in a park" and target_concept == "dog"
        if failure == "raises":
            raise RuntimeError("original attack error")
        return []

    monkeypatch.setattr(attack, "generate", generate)
    monkeypatch.setattr(attack, "cleanup", lambda context: calls.append("cleanup"))
    runner = ExperimentRunner(model, attack, build_defense("none"), tmp_path / "out")
    error, message = (
        (RuntimeError, "original attack error")
        if failure == "raises"
        else (ValueError, "returned no candidates")
    )
    with pytest.raises(error, match=message):
        runner.run([("a dog sitting in a park", "dog")], 42)
    assert calls == ["cleanup"]
    assert model.calls == 0
    assert not (tmp_path / "out/results.jsonl").exists()


def test_cli_attack_error_is_nonzero_and_never_generates(tmp_path, fake_encoder):
    result = CliRunner().invoke(
        app,
        [
            "--model",
            "mock",
            "--attack",
            "ring_a_bell",
            "--defense",
            "none",
            "--prompt",
            "a dog sitting in a park",
            "--target",
            "unknown concept",
            "--out",
            str(tmp_path / "out"),
        ],
    )
    assert result.exit_code != 0
    assert "No positive/negative pairs configured" in str(result.exception)
    assert not fake_encoder.text_calls
    assert not (tmp_path / "out/results.jsonl").exists()
    assert not list(tmp_path.rglob("*.png"))


def test_progress_logs_only_intervals():
    records = []
    settings = replace(RingABellSettings(), population_size=4, generations=6, log_interval=3)
    discover(FakeEncoder(), None, settings, 42, records.append)
    assert [record["generation"] for record in records] == [1, 4, 6]


def test_pgj_remains_available_without_loading_llm(monkeypatch):
    from t2i_framework.attacks import pgj

    class LLM:
        def __init__(self, **kwargs):
            pass

        def generate(self, *args):
            return "a blue toy cube"

    monkeypatch.setattr(pgj, "_LLMBackend", LLM)
    assert build_attack("pgj").generate("a cube", "cube")[0].text == "a blue toy cube"


def test_sd14_adapter_offload_and_release_with_fake_pipeline(
    monkeypatch, tmp_path, fake_encoder, small_config
):
    diffusers = pytest.importorskip("diffusers")
    from PIL import Image

    from t2i_framework.core.registry import build_model

    calls = []

    class Pipeline:
        scheduler = diffusers.PNDMScheduler()

        def enable_model_cpu_offload(self, device):
            calls.append(("offload", device))

        def __call__(self, **kwargs):
            calls.append(("generate", kwargs))
            return SimpleNamespace(images=[Image.new("RGB", (8, 8), "white")])

    def load(model_id, **kwargs):
        assert fake_encoder.closed
        calls.append(("load", model_id, kwargs))
        return Pipeline()

    monkeypatch.setattr(diffusers.AutoPipelineForText2Image, "from_pretrained", load)
    config = load_yaml_config(Path("configs/models/sd14.yaml"))
    config.update(small_config)
    model_config = dict(config["model"])
    model = build_model(model_config.pop("name"), **model_config)
    result = ExperimentRunner(
        model,
        RingABellAttack(),
        CharacterFilterDefense(semantic_matcher=Matcher(), captioner=Captioner()),
        tmp_path / "output",
    ).run([("a red cube on a table", "red")], 42, config)[0]
    assert result.success
    assert calls[0] == (
        "load",
        "CompVis/stable-diffusion-v1-4",
        {
            "torch_dtype": torch.float32,
            "low_cpu_mem_usage": True,
            "safety_checker": None,
            "requires_safety_checker": False,
        },
    )
    assert calls[1] == ("offload", "cuda")
    generated = calls[2][1]
    assert generated["prompt"] == fake_encoder.decoded
    assert generated["generator"].initial_seed() == 42
    assert generated["generator"].device == torch.device("cpu")
    assert (generated["width"], generated["height"], generated["num_inference_steps"]) == (
        512,
        512,
        50,
    )
    assert Path(result.generated_image_path).is_file()


@pytest.mark.parametrize("seed", [-1, 2**32, True])
def test_invalid_seeds_fail_before_loading(fake_encoder, seed):
    with pytest.raises(ValueError, match="seed"):
        RingABellAttack().generate("a cube", "red", {"seed": seed})


@pytest.mark.parametrize("shape,value", [((77, 2), 1.0), ((77, 768), float("nan"))])
def test_concept_vector_validation(shape, value):
    class Encoder:
        def embed_texts(self, texts):
            return torch.full((len(texts), *shape), value)

    with pytest.raises(ValueError, match="finite.*shape"):
        extract_concept(Encoder(), [("a", "b")])


def test_registry_import_does_not_require_optional_ml_dependencies():
    import subprocess
    import sys

    script = """
import sys
class NoML:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch', 'numpy', 'transformers'):
            raise ImportError('ML imports forbidden during registry construction')
sys.meta_path.insert(0, NoML())
from t2i_framework.core.registry import build_attack
assert build_attack('ring_a_bell').name == 'ring_a_bell'
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
