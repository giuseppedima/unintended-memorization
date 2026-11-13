Tutti i vincoli sono **matematicamente garantiti** con un margine di manovra enorme. Vediamo il calcolo combinatorio passo dopo passo e come impostare il campionamento (con o senza reinserimento).

---

### 1. Il calcolo combinatorio di base

Dai requisiti dell'esperimento abbiamo tre numeri fissi:

* **Grandezza del vocabolario ($V$):** $32$ parole.


* **Numero totale di canary ($N$):** $24$ frasi distinte.


* **Lunghezza di ogni frase ($L$):** $4$ parole diverse.



Da questi tre dati ricaviamo:

1. **Numero totale di "slot-parola" da riempire nell'esperimento:**

$$\text{Slot totali} = N \times L = 24 \times 4 = \mathbf{96 \text{ parole totali}}$$


2. **Numero totale di frasi valide di 4 parole diverse generabili da 32 parole** (disposizioni semplici $D_{32,4}$):

$$D_{32, 4} = 32 \times 31 \times 30 \times 29 = \mathbf{863.040 \text{ frasi possibili}}$$



Dover scegliere solo $24$ frasi su $863.040$ possibili significa usare appena lo **$0,0028\%$** dello spazio delle combinazioni.



---

### 2. Verifica dei due vincoli

#### Vincolo A: "Ciascuna parola compare 3 volte in media"

* **In media (garantito sempre al 100%):** Poiché dobbiamo riempire $96$ slot totali pescando da un insieme di $32$ parole, la frequenza media matematica di ciascuna parola (indipendentemente da come le estrai!) è **sempre identicamente pari a 3**:



$$\text{Frequenza media} = \frac{\text{Slot totali}}{\text{Numero di parole nel vocabolario}} = \frac{96}{32} = \mathbf{3}$$


* **Esattamente 3 volte ciascuna (bilanciamento perfetto):** Poiché $96$ è un multiplo esatto di $32$ ($32 \times 3 = 96$), è possibile garantire non solo che ogni parola compaia *in media* 3 volte, ma che ciascuna delle 32 parole compaia **esattamente 3 volte** in tutto l'insieme dei 24 canary (varianza zero tra le frequenze delle parole).

---

#### Vincolo B: "Non ci sono mai due parole uguali consecutive"

Questo vincolo si può intendere a due livelli, ed **entrambi sono facilissimi da soddisfare al 100%**:

1. **Livello 1 — Dentro la stessa frase (nessuna ripetizione adiacente tipo `maple maple orbit cobalt`):**
* La specifica richiede di estrarre *"four different words"* per ogni canary.


* Se tutte e 4 le parole della frase sono diverse tra loro ($w_1 \neq w_2 \neq w_3 \neq w_4$), è **impossibile** avere due parole uguali nella stessa frase (né consecutive, né a distanza).




2. **Livello 2 — Tra frasi diverse (nessuna coppia di parole consecutive / bigramma ripetuto in due canary diversi, es. `maple orbit` presente sia nel Canary 1 sia nel Canary 2):**
* Quante coppie ordinate di 2 parole diverse $(w_a, w_b)$ esistono in un vocabolario di 32 parole?



$$\text{Bigrammi possibili} = 32 \times 31 = \mathbf{992 \text{ coppie ordinate}}$$


* Quante coppie consecutive compaiono in tutto il nostro dataset di 24 frasi da 4 parole?
Ogni frase di 4 parole $(w_1, w_2, w_3, w_4)$ contiene **3 coppie consecutive**: $(w_1, w_2)$, $(w_2, w_3)$ e $(w_3, w_4)$.
Su 24 frasi totali, il numero complessivo di coppie consecutive usate è:



$$\text{Bigrammi usati} = 24 \times 3 = \mathbf{72 \text{ coppie ordinate}}$$


* Poiché **ci servono solo $72$ coppie consecutive su $992$ disponibili** (appena il **$7,25\%$**), è semplicissimo fare in modo che **nessuna coppia di parole consecutive compaia mai più di una volta** in tutti i 24 canary!
* *Bonus ancora più forte:* Dato che i canary sono $24$ e le parole del vocabolario sono $32$ ($24 \le 32$), potete persino imporre che **tutti i 24 canary inizino con una prima parola $w_1$ diversa** (usando 24 parole distinte su 32 nella prima posizione).





---

### 3. Serve il sampling CON o SENZA reinserimento?

La risposta dipende dal livello a cui guardi l'estrazione (**dentro la singola frase** vs. **tra le 24 frasi**):

| Livello di campionamento | Con o Senza reinserimento? | Perché? |
| --- | --- | --- |
| **1. DENTRO la singola frase**<br>

<br>*(estrazione delle 4 parole di un singolo canary)* | **SENZA reinserimento**<br>

<br>*(Without replacement)* | **Obbligatorio** per specifica (*"sampling four different words"*). La 1ª parola si pesca tra 32, la 2ª tra le 31 restanti, la 3ª tra 30, la 4ª tra 29. Garantisce che le 4 parole della frase siano tutte distinte.

 |
| **2. TRA le 24 frasi**<br>

<br>*(generazione dell'intero set di 24 canary)* | **CON reinserimento nel vocabolario di 32 parole**<br>

<br>*(oppure **SENZA reinserimento da un pool di 96 gettoni**)* | Se non rimettessi mai le parole nel vocabolario tra una frase e l'altra, dopo $32 / 4 = \mathbf{8 \text{ frasi}}$ avresti finito le 32 parole e non potresti arrivare a 24 frasi!

 |

In pratica, per generare le 24 frasi hai **due algoritmi possibili** (entrambi validissimi):

1. **Metodo A — Sampling indipendente tra le frasi (Con reinserimento dopo ogni blocco di 4 parole):**
* Per la Frase 1, estrai **senza reinserimento** 4 parole dalle 32.


* Rimetti le 4 parole nel sacchetto da 32 e ripeti l'estrazione per la Frase 2 (scartando e ri-estraendo se la frase è identica a una già uscita o se ripete un bigramma consecutivo già visto).


* *Risultato:* Ogni parola compare **in media** 3 volte ($\text{media} = 3$, ma per puro caso una parola può uscire 2 volte e un'altra 4 volte).


2. **Metodo B — Pool bilanciato di 96 gettoni (Senza reinserimento su 3 copie per parola):**
* Crei una lista iniziale di $96$ elementi contenente **esattamente 3 copie di ciascuna delle 32 parole**.
* Estrai le 24 quaterne da questo pool di 96 elementi imponendo che in ogni quaterna le 4 parole siano distinte e che nessun bigramma consecutivo $(w_i, w_{i+1})$ sia ripetuto.


* *Risultato:* Ogni parola compare **esattamente 3 volte** (media $= 3$, varianza $= 0$) e nessuna coppia consecutiva si ripete mai.