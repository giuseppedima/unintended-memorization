Il passaggio dal documento di studio interno [`01_documento_operativo.md`](01_documento_operativo.md) al file di presentazione LaTeX [`slides.tex`](../presentation/slides.tex) è stato pensato per trasformare un manuale operativo per il team in una presentazione accademica rigorosa, pulita e pronta per la prima consegna intermedia.

---

## 1. Cosa è stato INSERITO nelle slide (Struttura delle 15 diapositive)

Tutte le sezioni obbligatorie previste da `EmpiricalStudyProtocol.pdf` fino a *"Comments and Discussion before"* e tutti i vincoli sperimentali di `InCourseEmpiricalStudy.pdf` sono stati mantenuti e sintetizzati in blocchi visivi (`block`, `alertblock`, `exampleblock` e tabelle) compatibili con il tema `UniNA`:

| Sezione | Slide | Contenuto inserito |
| --- | --- | --- |
| **Frontespizio & Agenda** | Slide 1–2 | Titolo dello studio, autori, affiliazione Federico II e indice automatico (`\tableofcontents`).

 |
| **1. Introduction** | Slide 3–4 | • Definizione di *Unintended Memorization* vs. generalizzazione.

<br>

<br>• Rischi pratici (privacy/data leakage, copyright) e valore scientifico.<br>

<br>• Definizione di *Synthetic Canary* e distinzione tra inserimento intenzionale e memorizzazione non richiesta come task a valle.

 |
| **2. Research Question(s)** | Slide 5 | • **RQ1:** Impatto del numero di ripetizioni in fine-tuning su predittibilità (*likelihood*) ed estrazione esatta.

<br>

<br>• **RQ2:** Efficacia dell'*Exact Training Prefix* rispetto alla *Paraphrased Question*.

<br>

<br>• Giustificazione esplicita del perché le risposte non sono ovvie a priori (*"not immediately available"*).

 |
| **3. Benchmark(s)** | Slide 6–7 | • Scelta del benchmark sintetico controllato (*pilot case*) per escludere contaminazioni pre-esistenti.

<br>

<br>• Chiarimento sul paradigma usato: *Canary Insertion / Secret Extraction* (né *Needle-in-the-Haystack* né *LLM-as-a-Judge*, poiché la *ground truth* di 4 parole consente una valutazione deterministica).

<br>

<br>• Formato del canary (`Velora-007` $\rightarrow$ `maple orbit lantern cobalt`), vocabolario di 32 parole *single-token*, mitigazione dell'effetto *Pigeonhole* (nessun bigramma iniziale condiviso) e ortogonalità dei 24 `Project ID`.

 |
| **4. Experimental Protocol** | Slide 8–12 | (Suddiviso in 5 slide come suggerito da "You may need more slides"):

<br>

<br>• **Slide 8 (Gruppi e Baseline):** Tabella dei 4 gruppi ($0, 1, 5, 20$ copie/epoca $\rightarrow 0, 2, 10, 40$ esposizioni su 2 epoche, 6 canary/gruppo = 24 totali) e ruolo della doppia baseline (*Pre-training* vs. *Group 0 Post-training*).

<br>

<br>• **Slide 9 (Corpus e Data Split):** 256 record da esattamente 96 token IDs (156 canary + 100 background), sostituzione senza aumentare la taglia del corpus, taglio della sola coda di prosa ordinaria (con 156 code distinte) e razionale di *Training / No Validation / Testing*.

<br>

<br>• **Slide 10 (Pre-processing & Sanity Checks):** Controllo dei 4 *random seed* e tabella dei 5 *Sanity Checks* automatici (`SC-1`–`SC-5`).

<br>

<br>• **Slide 11 (Prompts & Design fattoriale):** *Exact prefix* e *Paraphrased question* senza segreto nel contesto, in modalità *raw plain-text* (senza chat template e senza spazio finale), fissati a priori, per $24 \times 2 \times 2 = 96$ generazioni e 96 likelihood.

<br>

<br>• **Slide 12 (Decoding & Metriche):** Greedy decoding (max 24 token), *whitespace normalization* sulla prima riga non vuota preservando *case* e punteggiatura; metriche di *Log-Likelihood*, *Paired & Net Memorization Gain* ($\Delta\mathcal{L}_{\text{net}}$), *first-word bottleneck* ($w_1$), *Strict Exact Match* e metriche diagnostiche sulla *raw continuation* (*Relaxed Prefix Match* e *Word Recall*).

 |
| **5. Involved approaches** | Slide 13 | • Fine-tuning **LoRA** (pesi base congelati, associazioni confinate nell'adapter) esteso a moduli di Attention + MLP.

<br>

<br>• Loss causale standard su tutti i 96 token, *micro-batching* ($\text{BS}=4$, 64 step/epoca) con *stratified shuffling*, scheduler costante e switch `model.disable_adapter()` in `eval()` + `float32`.

 |
| **6. Considered Model(s)** | Slide 14 | • Tabella comparativa con 3 modelli che condividono lo stesso tokenizer: `Qwen2.5-0.5B-Instruct`, `Qwen2.5-0.5B` (Base) e `Qwen2.5-1.5B-Instruct`.

 |
| **7. Comments and Discussion before** | Slide 15–17 | • **Slide 15 (Ipotesi):** Le 5 ipotesi falsificabili ($H_1\text{--}H_5$) da verificare (andamento monotono vs. soglia, divario prefisso/parafrasi, deriva del gruppo 0, scala e instruct vs. base).

<br>

<br>• **Slide 16 (Analisi a priori dei failure mode):** Il paradosso del punto fermo (`.`) e della coda di background nella generazione greedy e le 4 categorie di errore per l'ispezione della *raw continuation*.

<br>

<br>• **Slide 17 (Limitazioni):** Confini di validità su LoRA vs. full pre-training, confondimento formato/lessico nella parafrasi, interferenza sul vocabolario di 32 parole e severità dello *Strict Exact Match*.

 |

---

## 2. Cosa è stato RIMOSSO (e perché)

Dal file LaTeX sono stati esclusi **4 gruppi di contenuti** presenti nel documento Markdown o nel file `slides.tex` di partenza:

1. **Le sezioni di dialogo interno e revisione della tua bozza iniziale (Sezioni 0 e 8 del Markdown):**
* Tutta la parte in cui correggevamo la tua prima bozza (es. *"32 parole e non 32 token"*, *"LoRA è una tecnica PEFT"*) e la tabella che separava i vincoli della professoressa dalle decisioni del vostro gruppo sono state rimosse dal testo visibile delle slide perché erano appunti di studio interni per te e il tuo collega.
* **Come sono state conservate per voi:** Le decisioni aperte del team sono state trasformate in **commenti LaTeX invisibili nel PDF** (`% NOTA PER IL TEAM: ...`) posizionati esattamente sopra le righe di codice corrispondenti nel file `.tex`.


2. **I dettagli implementativi di basso livello in Python/PyTorch (parte delle Note tecniche):**
* Nelle slide accademiche si espone *cosa* fa il protocollo e *perché*, ma non si mettono i dettagli di sintassi Python o i warning sui bug di programmazione. Ho quindi rimosso dalle slide:
* La spiegazione dell'indice *off-by-one* dei tensori in PyTorch (`logits[0, m-1:m+3]`).
* Le espressioni regolari interne del pre-tokenizer BPE (`\p{L}+`) che separano `"cobalt"` dal punto `"."`.
* Il codice Python `" ".join(first_line.split())` e lo slicing `output_ids[0, input_ids.shape[1]:]`.
* Il warning sul non salvare e ri-tokenizzare i file JSON per evitare variazioni da 96 a 95/97 token.


* Tutti questi aspetti restano nel vostro file Markdown come guida per quando scriverete il codice per la consegna finale.




3. **Le sezioni successive all'esperimento (`Results`, `Discussion of the results`, `Conclusions`):**
* Come hai richiesto e come previsto per lo step intermedio in `EmpiricalStudyProtocol.pdf`, la presentazione si ferma esattamente a *"Comments and Discussion before"* inclusa, omettendo le ultime tre sezioni che andranno compilate solo dopo aver eseguito gli esperimenti.




4. **Le slide di esempio/tutorial del template `UniNA` (`slides.tex`):**
* Ho rimosso tutte le slide dimostratrici presenti nel file allegato (`Installation`, `User Interface`, la slide con i colori viola/verdi di test, gli esempi di `Plain frame` e la bibliografia fittizia su Dijkstra/Lamport), mantenendo però intatti l'intestazione del tema `UniNA`, i font (`AlegreyaSans`, `beramono`), i blocchi e la `\standoutpage` conclusiva.





---

## 3. Punti nel file `.tex` pronti da personalizzare col tuo collega

Nel file `unintended_memorization_slides.tex` ti basta cercare la stringa `% NOTA PER IL TEAM` per trovare i 3 punti rapidi che potete personalizzare prima di compilare il PDF definitivo:

* **Riga 28 (`\author`):** Inserire i vostri nomi e cognomi al posto di `Student Name 1` e `Student Name 2`.
* **Slide 11 (Prompt parafrasato):** Decidere se lasciare l'istruzione fedele alla lettera alla Slide 13 (`"Return only four words."`) o aggiungere `"without punctuation"`.


* **Slide 14 (Tabella dei modelli):** Confermare i 3 modelli della famiglia `Qwen2.5` già inseriti in tabella oppure sostituirne uno se decidete diversamente.