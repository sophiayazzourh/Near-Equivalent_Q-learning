# Near-Equivalent Q-learning
Near-Equivalent Q-learning Policies for Dynamic Treatment Regimes

This repository implements an extension of classical backward Q-learning that allows the identification of near-equivalent treatment strategies instead of a single optimal policy.

The method introduces a controlled admissibility parameter that relaxes strict optimality while preserving the recursive regression-based structure of offline Dynamic Treatment Regimes. Selection is performed at the penultimate stage and propagated backward using matrix-valued pseudo-outcomes, ensuring computational stability and structural coherence.

The repository includes simulation studies in both single-stage and multi-stage settings to illustrate boundary detection, stability of near-equivalent strategies, and practical clinical relevance.
