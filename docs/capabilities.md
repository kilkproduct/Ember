# Capabilities: what an Ember brain can learn, and how to teach it

Ember trains a small decoder-only language model (typically
0.2M–1.1M parameters) entirely on your device. It will never match a
datacenter AI — those have ~10,000× more parameters and training. But
"all-purpose" is still a meaningful goal at this scale if it means:
**the brain learns to respond to many kinds of requests in the right
format, grounded in your own data.** This page is the honest recipe
book: what each skill needs, what to expect, and the hard limits.

## The one rule that makes skills work

A base language model only continues patterns. It does skills when:

1. Training data shows the task in **one consistent format** (the
   `### Prompt: / ### Response:` lessons from Teach and the packs), and
2. At generation time it is asked in **exactly the same format**
   (Voice → *Task* does this for you).

Fire → *Lesson loss: Response only* usually learns skills faster:
gradient is spent on answers, not on generating questions.

## Recipes

### Translation

- Load **translation lessons** (Fuel). Starter: English↔Spanish plus
  French / German / Malay / Chinese, all in the template
  `Translate from English to Spanish: <sentence>` → `<translation>`.
- Cast with **byte-BPE**: it can represent any language/script
  exactly (the legacy tokenizer turns uncovered characters into �).
- To go beyond the scaffold, add your own parallel sentences as Teach
  lessons in the identical template. Expect: memorized pairs work
  after a few hundred steps; **hundreds–thousands** of pairs for even
  toy generalization to new sentences; still far below real
  translation apps. Short, simple sentences first.

### Research over your documents

- Forage/Teach the material you want answers from (that's the
  knowledge — the brain knows nothing it hasn't been fed).
- Load **research/QA lessons**: they teach answering *from a provided
  passage*, including "The passage does not say."
- Ask with Voice → *Task: Answer from my documents*: Ember retrieves
  the best-matching passages from your corpus (on-device word-overlap
  search), asks against them, and names the source documents in the
  output line. No match → it says so instead of inventing a source.
- Limits: passages are trimmed to the context window (watch the ⚠ in
  output), retrieval is keyword-based (different wording can miss),
  and a small model can still answer beyond the passage — check the
  named sources before trusting an answer.

### Writing skills (summarize, rewrite, keywords, grammar, continue)

- Load **general-skills lessons**, fire, and use the same phrasings as
  the lessons ("Summarize in one sentence: …", "Rewrite politely: …",
  …) or Voice → *Task: Summarize*. Add your own Teach lessons in
  identical shapes to extend or personalize them.

### Mixing everything (a genuinely general brain)

- Combine ordinary documents + skill lessons + topic pack in one
  corpus; lessons always train (never held out), and Fire → *Source
  balance* can stop one big source drowning smaller ones.
- What moves capability most at this scale (measured, see
  `docs/architecture-experiments.md`): **more and cleaner data + more
  steps**. Bigger architectures alone showed no gain on small data —
  grow the corpus first, then consider a wider vessel.

## Hard limits (no setting fixes these)

- **Knowledge** ends at your corpus: unseen facts will be invented.
- **Arithmetic/reasoning** is unreliable at this size; treat numbers
  in output as decoration.
- **Context** is 64–256 tokens (a few paragraphs) — long-document QA
  sees only excerpts.
- **Languages** need examples: a language with no training pairs
  produces fluent-looking nonsense, however confident it reads.
- **Fluency ≠ truth.** A well-trained Ember brain imitates the style
  of correct answers; verify anything that matters.
