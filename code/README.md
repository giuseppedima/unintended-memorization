# Unintended Memorization in LLMs — Empirical Study Codebase

Implementazione modulare, testata e riproducibile dello studio empirico *When Does an LLM Memorize a Synthetic Secret?* su **`Qwen/Qwen2.5-0.5B-Instruct`** (Corso di *Empirical Studies*, Università degli Studi di Napoli Federico II).

## Architettura del Progetto

```text
unintended_memorization_study/
├── .env / .env.example          # Configurazione centralizzata di tutti gli iperparametri
├── pyproject.toml               # Configurazione package Python (`src/` layout) e pytest
├── requirements.txt             # Dipendenze Python/PyTorch/HuggingFace
├── src/memorization/
│   ├── config.py                # ExperimentConfig tipizzato, validazione esplicita e seed globale
│   ├── exceptions.py            # ConfigurationError e ProtocolViolationError
│   ├── logging_utils.py         # Logging strutturato su console e file
│   ├── cli.py                   # Entry point unificato (`--stage corpus|train|analyze|all`)
│   ├── data/
│   │   ├── dictionary.py        # Verifica single-token delle 32 parole e 24 ID ortogonali
│   │   ├── secrets.py           # Generatore combinatorio vincolato (w1 unici, zero bigrammi, freq=3)
│   │   ├── background.py        # Streaming e igiene rigorosa da Salesforce/wikitext
│   │   ├── corpus.py            # Assemblaggio 256x96 token IDs (Opzione A, no padding)
│   │   └── checks.py            # 5 Sanity Checks automatici (SC-1..SC-5)
│   ├── training/
│   │   ├── dataset.py           # PreTokenizedCorpusDataset (loss CLM su tutti i 96 token)
│   │   ├── sampler.py           # ControlledNoDuplicateBatchSampler (0 duplicati per mini-batch)
│   │   └── trainer.py           # Pre-eval (`disable_adapter`), LoRA Attention+MLP, Post-eval
│   ├── evaluation/
│   │   ├── likelihood.py        # Log-likelihood token-level (l_w1..l_w4, rank w1, float32)
│   │   ├── generation.py        # Greedy decoding deterministico e post-processing prima riga
│   │   ├── taxonomy.py          # Strict Exact Match, metriche diagnostiche e 6 classi multi-label
│   │   └── runner.py            # Orchestrazione 96 test + calcolo delta appaiato e netto
│   └── analysis/
│       ├── summaries.py         # Statistiche descrittive per gruppo e verifica ipotesi H1-H5
│       └── plots.py             # 5 funzioni dedicate (fig1..fig5)
├── scripts/
│   └── run_pipeline.py          # Launcher CLI eseguibile direttamente
├── notebooks/
│   └── run_colab.ipynb          # Notebook orchestratore pronto per Google Colab (Tesla T4)
└── tests/                       # Suite di unit test eseguibile con `pytest`
```

## Installazione ed Esecuzione Rapida (Locale)

```bash
pip install -r requirements.txt
pip install --no-deps -e .

# 1. Verifica i test unitari
pytest -v

# 2. Esegui l'intera pipeline (oppure singoli stadi: corpus, train, analyze)
python scripts/run_pipeline.py --stage corpus
python scripts/run_pipeline.py --stage train
python scripts/run_pipeline.py --stage analyze
```

## Esecuzione su Google Colab (Tesla T4)

1. Carica `notebooks/run_colab.ipynb`
2. Esegui **Runtime → Run all** (gestisce automaticamente il conflitto `torchao` di Colab, la sincronizzazione con Google Drive e la chiusura pulita dello streaming HuggingFace).

## Personalizzazione degli Iperparametri

Tutti i parametri modificabili (`SEED`, `LEARNING_RATE`, `BATCH_SIZE`, `LORA_R`, `LORA_ALPHA`, `DTYPE`, `USE_GOOGLE_DRIVE`, ecc.) sono centralizzati in `.env`.
