from __future__ import annotations

prompts: list[str] = [
    "Never gonna give you up! Never gonna let you down!",
    "Wait, what? That can't be right!",
    "She smiled, waved, and left. I never saw her again.",
    "Stop! Look both ways before you cross.",
    "Is it ready yet? Almost, but not quite.",
    "The quick brown fox jumps over the lazy dog. Spectators laughed, cheered, and asked for one more run!",
    "In a hole in the ground there lived a hobbit. It was not a nasty, dirty, wet hole, thank goodness.",
    "Call me Ishmael. Some years ago, never mind how long precisely, I had little money and plenty of time.",
    "Rain tapped the window all night. By dawn, the streets shone like polished glass!",
    "Mix flour, salt, and yeast in a bowl. Let the dough rise until it doubles, then bake!",
    "The committee meets at nine, right? Please bring the slides, notes, and your questions.",
    "It was the best of times, it was the worst of times. Paris hummed with carts, bells, and gossip; London answered with fog, coal, and hurried boots.",
    "All happy families are alike. Each unhappy family is unhappy in its own way, and each tells a different story when guests finally leave!",
    "Machine learning models can spike on rare tokens. Watch the activations, log the layers, and ask whether the outlier repeats across batches?",
    "Transformers relate every token to every other token. That flexibility is powerful, but it can scatter attention when prompts grow long, noisy, or vague.",
    "Scientists reported a sharp decline in Arctic sea ice. Policymakers debated targets, trade-offs, and timelines; reporters asked who pays first, and why?",
    "The sky above the port was the color of television, tuned to a dead channel. Neon flickered, rain hissed, and a drone buzzed past the tower. Nobody looked up!",
    "Space, the final frontier! These are the voyages of the starship Enterprise. Its mission: explore strange worlds, seek new life, and boldly go where no one has gone before?",
    "Outlier features often concentrate in a few MLP channels. The effect can be subtle at first, then suddenly loud. Do we treat that as signal, noise, or a warning?",
    "Quantization-aware training tries to preserve accuracy when weights move to int8. Calibrate carefully, validate on real traffic, and keep a float fallback for edge cases!",
    "Local authorities opened cooling centers as temperatures soared. Nurses handed out water, volunteers mapped shade, and radio hosts repeated the same advice: stay inside, check on neighbors, hydrate.",
    "No, thank you! I already ate, drank coffee, and read three chapters. Ask me again after lunch?",
    "It is a truth universally acknowledged that a single man in possession of a good fortune must be in want of a wife. Mothers whispered plans, daughters practiced smiles, and fathers reviewed estates with ink-stained fingers. Was any of it love, or just arithmetic dressed in lace?",
    "The European Parliament voted today on a digital services framework. Platforms promised transparency, critics demanded liability, and lawyers argued over definitions that could shift billions. By evening, headlines screamed victory; by morning, everyone read the fine print!",
    "A small startup announced funding for interpretability tools. Investors asked about moats, researchers asked about baselines, and engineers asked for one more week to fix the charts. Can you ship insight at the same speed you ship features?",
    "Critics praised the film's cinematography but questioned the pacing of act two. The director defended every cut, the editor defended every transition, and the audience defended its right to disagree. Art, commerce, and popcorn—what a strange coalition!",
    "Yes? No? Maybe! The survey answers were all over the place, yet the trend held. Should we rerun it with clearer instructions?",
    # Very long (~110–150 words, 3 sentences)
    "He explained that gradient clipping stabilizes training when loss spikes abruptly. Gradients explode for many reasons: bad learning rates, rare tokens, mixed precision, or a batch that should have been dropped. Clip, log, retry, and document the failure so the next run starts wiser!",
    "Attention maps can reveal which tokens a model relies on when generating an answer. A model may stare at punctuation, cling to the first noun, or circle back to a name you mentioned once. If you change one comma, does the map change enough to matter?",
    "Despite the outage, the on-call engineer restored service in twenty-seven minutes. Alerts had piled up, customers had complained, and leadership had asked for a timeline every six minutes. When systems breathe again, write the postmortem while the memory still stings!",
    "Volunteers planted native grasses to reduce erosion along the hillside trail. Wind, rain, and foot traffic had carved gullies where wildflowers once held the soil. Next spring, will the path stay, or will we be back with shovels, seeds, and the same stubborn hope?",
    "Three clauses, three moods: calm, tense, elated. First it drizzled, then it poured, then the sun returned! Did the afternoon forgive the morning?",
]

assert len(prompts) == 32
