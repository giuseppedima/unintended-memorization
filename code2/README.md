# Unintended Memorization in LLMs — Empirical Study Codebase

Codice dello studio empirico *When Does an LLM Memorize a Synthetic Secret?*

## Struttura del Progetto

```text
unintended_memorization_study/
├── config.py          # Tutte le costanti dell'esperimento (modello, seed, LoRA, shuffle, percorsi)
├── dictionary.py      # 32 parole fisse single-token, 24 ID progetto e 4 gruppi di esposizione
├── build_corpus.py    # Fase 1: generazione 24 segreti, streaming WikiText, corpus 256x96 e SC-1..SC-5
├── train.py           # Fase 2: fine-tuning LoRA (2 epoche, Controlled Shuffling) e salvataggio adapter
├── evaluate.py        # Fase 3: valutazione pre/post (48+48), tassonomia multi-label e calcolo delta
├── analyze.py         # Fase 4: statistiche per gruppo, verifica H1-H5 e generazione 5 figure PNG
├── run_colab.ipynb    # Notebook pronto per Google Colab (Tesla T4)
└── requirements.txt   # Dipendenze essenziali
```

## Riproduzione Run 1 vs. Run 2

Nel file `config.py`:
- **`SHUFFLE_BACKGROUND = False`** riproduce la **First Run** (WikiText sequenziale).
- **`SHUFFLE_BACKGROUND = True`** riproduce la **Second Run** (WikiText mescolato).

## Nota su Google Colab

Apri `run_colab.ipynb` su runtime **T4 GPU** ed esegui le celle in ordine.
