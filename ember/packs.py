"""Curated offline starter and skill lesson packs for Ember."""

from __future__ import annotations

PACKS: dict[str, dict] = {
    "starter": {
        "title": "Public-domain starter",
        "docs": [
            {
                "title": "Down the Rabbit-Hole",
                "source": "starter",
                "kind": "page",
                "weight": 1,
                "text": (
                    "Alice was beginning to get very tired of sitting by her sister on the bank, "
                    "and of having nothing to do: once or twice she had peeped into the book her "
                    "sister was reading, but it had no pictures or conversations in it, 'and what "
                    "is the use of a book,' thought Alice 'without pictures or conversations?'\n\n"
                    "So she was considering in her own mind, as well as she could, for the hot day "
                    "made her feel very sleepy and stupid, whether the pleasure of making a "
                    "daisy-chain would be worth the trouble of getting up and picking the daisies, "
                    "when suddenly a White Rabbit with pink eyes ran close by her."
                ),
            },
            {
                "title": "The Pool of Tears",
                "source": "starter",
                "kind": "page",
                "weight": 1,
                "text": (
                    "'Curiouser and curiouser!' cried Alice; she was so much surprised, that for "
                    "the moment she quite forgot how to speak good English. 'Now I am opening out "
                    "like the largest telescope that ever was! Good-bye, feet!' When she looked "
                    "down at her feet, they seemed to be almost out of sight, they were getting "
                    "so far off."
                ),
            },
            {
                "title": "A Mad Tea-Party",
                "source": "starter",
                "kind": "page",
                "weight": 1,
                "text": (
                    "There was a table set out under a tree in front of the house, and the March "
                    "Hare and the Hatter were having tea at it: a Dormouse was sitting between "
                    "them, fast asleep, and the other two were using it as a cushion, resting "
                    "their elbows on it, and talking over its head. 'Very uncomfortable for the "
                    "Dormouse,' thought Alice; 'only, as it is asleep, I suppose it does not mind.'"
                ),
            },
            {
                "title": "The Night Sky",
                "source": "starter",
                "kind": "page",
                "weight": 1,
                "text": (
                    "Stars appear to move across the night sky from east to west because the Earth "
                    "rotates on its axis once every twenty-four hours. Near the celestial poles, "
                    "stars trace tight circles around a fixed point rather than rising and setting. "
                    "Sailors used these steady markers for centuries to estimate latitude at sea."
                ),
            },
            {
                "title": "Water and Weather",
                "source": "starter",
                "kind": "page",
                "weight": 1,
                "text": (
                    "Sunlight warms the surface of oceans, lakes, and rivers, turning liquid water "
                    "into invisible vapour that rises into cooler air aloft. As moist air cools, "
                    "water condenses around tiny dust and salt particles to form clouds. When droplets "
                    "grow heavy enough, gravity pulls them back down as rain, sleet, hail, or snow."
                ),
            },
            {
                "title": "How Bread Rises",
                "source": "starter",
                "kind": "page",
                "weight": 1,
                "text": (
                    "Bread dough rises when yeast ferments simple sugars derived from flour starch, "
                    "releasing carbon dioxide gas and small amounts of organic acids and alcohol. "
                    "Kneading aligns wheat gluten proteins into an elastic network that traps those "
                    "gas bubbles so the loaf expands evenly in the oven."
                ),
            },
        ],
    },
    "topic": {
        "title": "Topic-training pack",
        "docs": [
            {
                "title": "Solar System Basics",
                "source": "topic-pack",
                "kind": "page",
                "weight": 2,
                "text": (
                    "Topic: Astronomy\n"
                    "The Solar System consists of the Sun and everything bound to it by gravity: "
                    "eight planets, dwarf planets, moons, asteroids, and icy comets. The four inner "
                    "planets—Mercury, Venus, Earth, and Mars—are rocky worlds, while Jupiter, Saturn, "
                    "Uranus, and Neptune are gas and ice giants."
                ),
            },
            {
                "title": "Photosynthesis",
                "source": "topic-pack",
                "kind": "page",
                "weight": 2,
                "text": (
                    "Topic: Biology\n"
                    "Green plants use chlorophyll inside chloroplasts to capture light energy and "
                    "convert water and carbon dioxide into glucose and oxygen. This process stores "
                    "solar energy in chemical bonds and supplies oxygen to Earth's atmosphere."
                ),
            },
            {
                "title": "Plate Tectonics",
                "source": "topic-pack",
                "kind": "page",
                "weight": 2,
                "text": (
                    "Topic: Geology\n"
                    "Earth's outer shell is divided into rigid lithospheric plates floating on the "
                    "slowly convecting mantle below. Where plates collide, mountain ranges rise and "
                    "deep trenches subduct ocean crust; where plates pull apart, magma wells up to "
                    "create new seafloor."
                ),
            },
            {
                "title": "How Compilers Work",
                "source": "topic-pack",
                "kind": "page",
                "weight": 2,
                "text": (
                    "Topic: Computing\n"
                    "A compiler translates human-readable source code into native machine instructions. "
                    "First a lexer splits text into tokens, then a parser builds a syntax tree, "
                    "semantic checks verify types, optimizers simplify intermediate code, and a "
                    "code generator emits instructions for the target processor."
                ),
            },
            {
                "title": "Ocean Currents",
                "source": "topic-pack",
                "kind": "page",
                "weight": 2,
                "text": (
                    "Topic: Oceanography\n"
                    "Surface ocean currents are driven chiefly by prevailing winds and deflected "
                    "by Earth's rotation into broad gyres. Deep thermohaline circulation moves cold, "
                    "dense, salty water from polar seas along the ocean floor toward the equator, "
                    "redistributing heat across the planet."
                ),
            },
        ],
    },
    "translation": {
        "title": "Translation lessons",
        "docs": [
            {
                "title": "Translate EN -> Spanish (greeting)",
                "source": "pack:translation",
                "kind": "lesson",
                "weight": 6,
                "text": (
                    "### Task: Translate to Spanish\n"
                    "### Input:\nGood morning, where is the train station?\n\n"
                    "### Response:\nBuenos días, ¿dónde está la estación de tren?"
                ),
            },
            {
                "title": "Translate EN -> Spanish (weather)",
                "source": "pack:translation",
                "kind": "lesson",
                "weight": 6,
                "text": (
                    "### Task: Translate to Spanish\n"
                    "### Input:\nThe weather is warm and sunny today.\n\n"
                    "### Response:\nEl tiempo está cálido y soleado hoy."
                ),
            },
            {
                "title": "Translate EN -> French",
                "source": "pack:translation",
                "kind": "lesson",
                "weight": 6,
                "text": (
                    "### Task: Translate to French\n"
                    "### Input:\nSmall models learn directly from the text you give them.\n\n"
                    "### Response:\nLes petits modèles apprennent directement à partir du texte que vous leur donnez."
                ),
            },
            {
                "title": "Translate EN -> German",
                "source": "pack:translation",
                "kind": "lesson",
                "weight": 6,
                "text": (
                    "### Task: Translate to German\n"
                    "### Input:\nPlease save a checkpoint before changing the architecture.\n\n"
                    "### Response:\nBitte speichere einen Kontrollpunkt, bevor du die Architektur änderst."
                ),
            },
        ],
    },
    "research": {
        "title": "Research / QA lessons",
        "docs": [
            {
                "title": "Grounded QA — boiling point",
                "source": "pack:research",
                "kind": "lesson",
                "weight": 6,
                "text": (
                    "### Context:\n"
                    "At standard sea-level pressure, pure water boils at 100 degrees Celsius. "
                    "At higher altitudes air pressure is lower, so water boils at a lower temperature.\n\n"
                    "### Question:\nWhy does water boil at a lower temperature on a mountain?\n\n"
                    "### Response:\nBecause atmospheric pressure is lower at high altitude."
                ),
            },
            {
                "title": "Grounded QA — unanswerable",
                "source": "pack:research",
                "kind": "lesson",
                "weight": 6,
                "text": (
                    "### Context:\n"
                    "Mercury completes one orbit around the Sun every 88 Earth days.\n\n"
                    "### Question:\nWhat is the population of Tokyo?\n\n"
                    "### Response:\nThe provided documents do not mention the population of Tokyo."
                ),
            },
            {
                "title": "Grounded QA — BPE",
                "source": "pack:research",
                "kind": "lesson",
                "weight": 6,
                "text": (
                    "### Context:\n"
                    "Byte-pair encoding starts from the 256 raw UTF-8 bytes and repeatedly merges "
                    "the most frequent adjacent token pair up to the target vocabulary size.\n\n"
                    "### Question:\nWhy can byte-level BPE encode any Unicode string without unknown tokens?\n\n"
                    "### Response:\nEvery UTF-8 character decomposes into bytes 0..255, which are always in the base vocabulary."
                ),
            },
        ],
    },
    "skills": {
        "title": "General-skills lessons",
        "docs": [
            {
                "title": "Summarize paragraph",
                "source": "pack:skills",
                "kind": "lesson",
                "weight": 6,
                "text": (
                    "### Task: Summarize\n"
                    "### Input:\n"
                    "Validation loss measures model predictions on held-out documents that are never "
                    "used for gradient updates. When training loss keeps falling while validation "
                    "loss rises, the model is memorizing its training windows instead of generalizing.\n\n"
                    "### Response:\n"
                    "Validation loss tracks performance on unseen documents so you can spot overfitting early."
                ),
            },
            {
                "title": "Extract keywords",
                "source": "pack:skills",
                "kind": "lesson",
                "weight": 6,
                "text": (
                    "### Task: Keywords\n"
                    "### Input:\n"
                    "Causal self-attention masks future positions so token t only attends to tokens 0..t.\n\n"
                    "### Response:\ncausal self-attention, future mask, autoregressive tokens"
                ),
            },
            {
                "title": "Direct Q&A lesson",
                "source": "pack:skills",
                "kind": "lesson",
                "weight": 6,
                "text": (
                    "### Prompt:\nHow do I keep a small model from repeating itself?\n\n"
                    "### Response:\nUse a mild repetition penalty around 1.1 to 1.2 and add diverse paragraphs to your corpus."
                ),
            },
        ],
    },
}


def get_pack(pack_id: str) -> list[dict]:
    key = (pack_id or "").strip().lower()
    if key in PACKS:
        return [dict(d) for d in PACKS[key]["docs"]]
    # Accept "all" or fallback cleanly
    if key == "all":
        out = []
        for entry in PACKS.values():
            out.extend(dict(d) for d in entry["docs"])
        return out
    raise ValueError(f"Unknown training pack '{pack_id}'. Available: {', '.join(PACKS)}")
