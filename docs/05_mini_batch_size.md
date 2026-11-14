Usiamo esattamente le formule e la notazione dei tuoi appunti su **RMSProp**, **Momentum** e **Adam** per rifare i calcoli passo dopo passo.

Dalle tue note abbiamo che la regola di aggiornamento di **RMSProp** per un singolo peso $w$ al passo $n$ è:


$$w^{(n+1)} = w^{(n)} - \frac{\eta^{(n)}}{\sqrt{r^{(n)} + \delta}} \left.\frac{\partial E}{\partial w}\right\vert{}_{w^{(n)}}$$

dove il parametro di riduzione $r^{(n)}$ è la somma esponenziale pesata tra il valore precedente $r^{(n-1)}$ e il quadrato della derivata parziale corrente (come descritto nel testo per AdaGrad/RMSProp):


$$r^{(n)} = \beta r^{(n-1)} + (1-\beta)\left(\left.\frac{\partial E}{\partial w}\right\vert{}_{w^{(n)}}\right)^2$$

*(Nota: nel testo incollato per $r^{(n)}$ mancava l'esponente $^2$ sulla derivata parziale e il $-1$ sull'indice di $r^{(n-1)}$, ma la definizione testuale dice appunto "somma esponenziale pesata tra il parametro di riduzione precedente e la somma dei quadrati").*

**Adam** unisce questa formula di **RMSProp** con il termine di **Momentum** $\mu \left(\left.\frac{\partial E}{\partial w}\right\vert{}_{w^{(n-1)}}\right)$.

---

### Impostiamo i numeri dell'esempio

Consideriamo un peso $w$ dell'adapter LoRA legato a uno specifico canary che il modello vede per la prima volta al passo $n$ (quindi finora quel peso non è stato sollecitato: $r^{(n-1)} = 0$ e $\left.\frac{\partial E}{\partial w}\right\vert{}_{w^{(n-1)}} = 0$).

Fissiamo dei valori numerici comodi per fare i conti a mente:

* $\eta^{(n)} = 0.01$ (learning rate al passo $n$)
* $\beta = 0.75 \implies (1-\beta) = 0.25$
* $\delta \approx 0$ (serve solo a evitare la divisione per zero quando $r^{(n)} = 0$)
* Quando nel mini-batch c'è **1 sola copia** del canary, la derivata parziale dell'errore rispetto a $w$ vale:

$$\left.\frac{\partial E}{\partial w}\right\vert{}_{w^{(n)}} = 2$$



Vediamo ora cosa succede nei tre casi usando la tua formula.

---

### Caso 1: Nel mini-batch c'è 1 sola copia del canary

1. **Derivata parziale dell'errore:**

$$\left.\frac{\partial E}{\partial w}\right\vert{}_{w^{(n)}} = 2$$


2. **Calcolo di $r^{(n)}$:**

$$r^{(n)} = 0.75 \cdot 0 + 0.25 \cdot (2)^2 = 0.25 \cdot 4 = 1$$


3. **Radice al denominatore ($\sqrt{r^{(n)} + \delta}$):**

$$\sqrt{r^{(n)} + \delta} \approx \sqrt{1} = 1$$


4. **Aggiornamento del peso $w^{(n+1)}$:**

$$w^{(n+1)} = w^{(n)} - \frac{0.01}{1} \cdot 2 = w^{(n)} - \mathbf{0.02}$$



Con **1 copia**, il peso varia di **$-0.02$**.

---

### Caso 2: Nello STESSO mini-batch finiscono 2 copie del canary (qui si semplificano i due fattori $2$)

Se nello stesso mini-batch finiscono **2 copie identiche** del canary, l'errore $E$ sul canary viene contato due volte, quindi la derivata parziale raddoppia (compare il **primo fattore $\mathbf{2}$**):

1. **Derivata parziale dell'errore (raddoppiata):**

$$\left.\frac{\partial E}{\partial w}\right\vert{}_{w^{(n)}} = \mathbf{2} \cdot 2 = 4$$


2. **Calcolo di $r^{(n)}$:**
Inseriamo la derivata parziale nella formula di $r^{(n)}$. Poiché la derivata viene elevata al quadrato, il fattore $\mathbf{2}$ diventa $\mathbf{2}^2$:

$$r^{(n)} = 0.75 \cdot 0 + 0.25 \cdot (\mathbf{2} \cdot 2)^2 = 0.25 \cdot \mathbf{2}^2 \cdot 2^2 = \mathbf{2}^2 \cdot (0.25 \cdot 4) = \mathbf{2}^2 \cdot 1 = 4$$


3. **Radice al denominatore ($\sqrt{r^{(n)} + \delta}$):**
Quando estraiamo la radice quadrata di $r^{(n)}$, il $\mathbf{2}^2$ esce dalla radice e diventa il **secondo fattore $\mathbf{2}$** (al denominatore):

$$\sqrt{r^{(n)} + \delta} \approx \sqrt{\mathbf{2}^2 \cdot 1} = \mathbf{2} \cdot 1 = 2$$


4. **Aggiornamento del peso $w^{(n+1)}$ — Semplificazione:**
Sostituiamo nella formula di RMSProp/Adam:

$$w^{(n+1)} = w^{(n)} - \frac{\eta^{(n)}}{\sqrt{r^{(n)} + \delta}} \left.\frac{\partial E}{\partial w}\right\vert{}_{w^{(n)}} = w^{(n)} - \frac{0.01}{\mathbf{2} \cdot 1} \cdot (\mathbf{2} \cdot 2)$$



Il $\mathbf{2}$ della derivata parziale (numeratore) e il $\mathbf{2}$ uscito dalla radice $\sqrt{r^{(n)}}$ (denominatore) **si semplificano**:

$$w^{(n+1)} = w^{(n)} - 0.01 \cdot \frac{\cancel{\mathbf{2}} \cdot 2}{\cancel{\mathbf{2}} \cdot 1} = w^{(n)} - \mathbf{0.02}$$



**Ecco perché 2 copie nello stesso batch sono un problema:** il raddoppio di $\left.\frac{\partial E}{\partial w}\right\vert{}_{w^{(n)}}$ al numeratore viene annullato dal fatto che anche $\sqrt{r^{(n)}}$ al denominatore raddoppia. Lo spostamento del peso rimane **$-0.02$**, esattamente come quando c'era una copia sola!

---

### Caso 3: Le 2 copie finiscono in 2 MINI-BATCH DIVERSI (passo $n$ e passo $n+1$)

Se invece usiamo lo *shuffling stratificato* per mettere la 1ª copia nel mini-batch $n$ e la 2ª copia nel mini-batch successivo $n+1$, in **Adam** (RMSProp + Momentum) otteniamo un doppio vantaggio:

1. **Al passo $n$ (1ª copia):**
Come visto nel Caso 1, il peso fa un primo spostamento completo:

$$w^{(n+1)} = w^{(n)} - \mathbf{0.02}$$


2. **Al passo $n+1$ (2ª copia):**
* Il peso fa un **secondo passo di discesa** perché viene ricalcolata la derivata parziale $\left.\frac{\partial E}{\partial w}\right\vert{}_{w^{(n+1)}}$ sul nuovo batch:

$$r^{(n+1)} = 0.75 \cdot (1) + 0.25 \cdot (2)^2 = 0.75 + 1 = 1.75 \implies \sqrt{r^{(n+1)}} \approx 1.32$$


$$\text{Termine RMSProp} = -\frac{0.01}{1.32} \cdot 2 \approx \mathbf{-0.015}$$


* **In più si attiva il Momentum di Adam!** Poiché al passo precedente $n$ avevamo già visto la 1ª copia ($\left.\frac{\partial E}{\partial w}\right\vert{}_{w^{(n)}} = 2 \neq 0$), ora entra in gioco anche l'inerzia della discesa $+\mu\left(\left.\frac{\partial E}{\partial w}\right\vert{}_{w^{(n)}}\right)$ che accelera ulteriormente lo spostamento del peso nella direzione giusta (cosa impossibile nel Caso 2, dove al passo $n-1$ la derivata era $0$).



In sintesi, usando i concetti delle tue slide:

* **Nello stesso mini-batch:** $\left.\frac{\partial E}{\partial w}\right\vert{}_{w^{(n)}}$ raddoppia, ma raddoppia anche $\sqrt{r^{(n)}}$ al denominatore di RMSProp/Adam, quindi i due fattori $2$ si cancellano e non hai alcun effetto di **Momentum** dal passo precedente.
* **In due mini-batch separati:** esegui due aggiornamenti distinti ($w^{(n)} \to w^{(n+1)} \to w^{(n+2)}$) e sul secondo aggiornamento sfrutti anche l'inerzia del **Momentum** generata dal primo.