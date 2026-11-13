**Entrambi i casi possono accadere realmente** (non sono affatto impossibili) e capire *perché* si verificano spiega esattamente come i modelli Transformer memorizzano le informazioni nei propri pesi.

Ricordando che per la **metrica primaria ufficiale** della professoressa entrambi i casi contano come **fallimento (`0`)** — poiché le frasi parziali e le parole errate non superano il test — vediamo perché possono verificarsi durante il *greedy decoding* e perché è interessantissimo osservarli quando ispezionerete la continuazione grezza (*raw continuation*).

---

### Caso 1: Il modello indovina le parole giuste, ma NON in ordine

(Es. Target: `maple orbit lantern cobalt` $\rightarrow$ Generato: `maple lantern orbit cobalt`)

**Può accadere? Sì, ed è anche piuttosto frequente** a livelli intermedi di esposizione (es. 2 o 10 esposizioni cumulative) o quando si usa la **domanda parafrasata**.

* **Perché succede (Effetto *Bag-of-Words* / Associazione Entità-Insieme):**
Quando addestri l'adapter LoRA, i layer di *Self-Attention* e *MLP* imparano prima di tutto un'**associazione semantica diretta** tra i token del nome del progetto (es. `Velora-007`) e l'**insieme** delle 4 parole del suo segreto (`{maple, orbit, lantern, cobalt}`).
In pratica, non appena il modello legge `Velora-007` nel prompt, "alza" contemporaneamente i logit (le probabilità) di tutte e 4 le parole associate a quel progetto.


* **Quando si inverte l'ordine:**
Se il numero di esposizioni non è ancora sufficiente a consolidare la catena sequenziale esatta ($w_1 \to w_2 \to w_3 \to w_4$), oppure se usi la **domanda parafrasata** (che cambia il formato e sposta le posizioni assolute dei token rispetto al prefisso di training), al passo 2 sia `orbit` sia `lantern` avranno una probabilità molto alta perché entrambe sono legate a `Velora-007`. Se `lantern` supera `orbit` anche solo dello $0{,}1\%$, il *greedy decoding* sceglie `lantern` al 2° posto e poi recupera `orbit` al 3° posto (perché il modello tende a non ripetere la parola appena generata).



---

### Caso 2: Il modello indovina la 1ª, sbaglia la 2ª e indovina la 3ª

(Es. Target: `maple orbit lantern cobalt` $\rightarrow$ Generato: `maple river lantern cobalt`)

**Può accadere? Sì, può accadere**, anche se è meno frequente rispetto al caso in cui il modello indovina le prime parole e poi sbaglia tutte le successive (`maple orbit <errore> <errore>`).

* **Perché potremmo pensare che non accada mai (L'equivoco della Catena di Markov):**
Se l'LLM fosse una semplice catena di Markov del primo ordine — in cui la parola $w_3$ dipende *solo* dalla parola appena generata $w_2$ — allora sbagliare la 2ª parola manderebbe inevitabilmente fuori strada anche la 3ª e la 4ª parola (*error propagation*).
* **Perché invece in un Transformer PUÒ accadere il "recupero" sulla 3ª parola:**
Un Transformer non guarda solo l'ultimo token generato ($\hat{w}_2$), ma grazie al meccanismo di **Self-Attention** guarda contemporaneamente **tutto il contesto precedente**:
1. Guarda ancora il **nome del progetto** (`Velora-007`) nel prompt.


2. Guarda la **1ª parola corretta** (`maple`) generata in posizione $w_1$.


3. Sente la **posizione relativa** (sa di trovarsi al 3° token dopo `"is"`).




Inoltre, poiché avete imposto il vincolo che **nessun bigramma consecutivo è condiviso tra i 24 canary**, se al passo 2 il modello commette un errore generando `river`, la coppia `(maple, river)` non appartiene a nessun altro canary. Di conseguenza, al passo 3 il modello non viene "dirottato" con forza verso un altro segreto, e l'attenzione diretta su `Velora-007` + `maple` può riportarlo sui binari giusti facendogli generare `lantern` in 3ª posizione!



---

### Come gestire questi due casi nella vostra analisi diagnostica

Per distinguere chiaramente questi comportamenti quando analizzerete i risultati (affiancando la metrica primaria *Strict Exact Match*), nel codice vi basterà calcolare due semplicissime metriche diagnostiche di **Partial Memorization** sulle prime 4 parole generate ($\hat{w}_1, \hat{w}_2, \hat{w}_3, \hat{w}_4$) rispetto alle 4 parole vere ($w_1, w_2, w_3, w_4$):

1. **Positional Word Accuracy (Ordine rigoroso, cattura il Caso 2):**
Conta quante parole si trovano **nella posizione esatta** ($t = 1, 2, 3, 4$):

$$\text{Acc}_{\text{pos}} = \frac{1}{4} \sum_{t=1}^{4} \mathbb{I}(\hat{w}_t = w_t)$$


* Se il modello genera `maple <errore> lantern cobalt`, ottiene **$3/4$ ($75\%$)** anche se ha sbagliato la seconda parola.


2. **Bag-of-Words / Set Recall (Senza ordine, cattura il Caso 1):**
Conta quante delle 4 parole del segreto sono state generate **indipendentemente dall'ordine**:

$$\text{Recall}_{\text{BoW}} = \frac{\vert{}\{\hat{w}_1, \hat{w}_2, \hat{w}_3, \hat{w}_4\} \cap \{w_1, w_2, w_3, w_4\}\vert{}}{4}$$


* Se il modello genera `maple lantern orbit cobalt` (invertendo 2ª e 3ª parola), avrà $\text{Acc}_{\text{pos}} = 2/4$ ($50\%$), ma **$\text{Recall}_{\text{BoW}} = 4/4$ ($100\%$)**!



Confrontare $\text{Recall}_{\text{BoW}}$ con $\text{Acc}_{\text{pos}}$ vi darà un'osservazione scientifica bellissima da discutere: se $\text{Recall}_{\text{BoW}} > \text{Acc}_{\text{pos}}$, dimostrate che il modello **ha memorizzato quali parole appartengono al progetto, ma non ha ancora fissato il loro ordine sequenziale esatto**!