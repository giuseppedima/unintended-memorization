In questo esperimento **non conviene arrivare all'estremo dell'*online learning* puro (`batch_size = 1`)**, ma è molto meglio usare un **mini-batch piccolo (es. `batch_size = 4` o `8`) abbinato a uno shuffling stratificato**.

L'intuizione che ti porta a pensare all'*online learning* è corretta: se riduciamo il batch a $1$ solo record, è impossibile che due copie dello stesso canary finiscano nello stesso passo di aggiornamento. Tuttavia, usando le formule di **RMSProp/Adam** e la struttura del nostro corpus da $256$ record, ecco perché il mini-batch piccolo batte l'*online learning* puro su tre fronti:

**1. Con `batch_size = 4` (o `8`) hai già zero collisioni, esattamente come nell'online learning**

* Nel nostro dataset ci sono $256$ record totali e il caso più estremo (Gruppo 3) ha $20$ occorrenze dello stesso canary per epoca.


* Con `batch_size = 4` l'epoca è divisa in $256 / 4 = \mathbf{64\text{ mini-batch}}$ (oppure $32$ mini-batch con `batch_size = 8`).
* Poiché hai $64$ mini-batch a disposizione e solo $20$ copie da piazzare, ti basta distribuire le copie in modo che non ce ne siano mai due nello stesso mini-batch (*stratified shuffling*). In questo modo **ciascuna delle $20$ copie cade in un mini-batch diverso** e riceve il suo passo di aggiornamento dedicato ($40$ passi distinti in $2$ epoche), ottenendo lo stesso identico beneficio dell'*online learning* senza i suoi difetti.



**2. Il difetto dell'online learning (`batch_size = 1`): rumore estremo e interferenza distruttiva**

* Nell'*online learning* puro, a ogni passo $n$ la derivata parziale $\left.\frac{\partial E}{\partial w}\right\vert{}_{w^{(n)}}$ viene calcolata su **un singolo record** da $96$ token.


* Ricorda la formula di **RMSProp/Adam**:

$$w^{(n+1)} = w^{(n)} - \frac{\eta^{(n)}}{\sqrt{r^{(n)} + \delta}} \left.\frac{\partial E}{\partial w}\right\vert{}_{w^{(n)}}$$



Poiché la derivata viene divisa per $\sqrt{r^{(n)} + \delta}$, **qualsiasi singolo record** (anche uno dei $100$ record di sola prosa ordinaria di background) produce uno spostamento completo di ampiezza $\approx \eta^{(n)}$ sui pesi delle matrici $\mathbf{A}$ e $\mathbf{B}$ di LoRA.


* **Cosa succede ai pesi?** Al passo $n$ il modello vede il canary e sposta i pesi nella direzione giusta; subito dopo, ai passi $n+1, n+2, n+3$, vede tre record singoli di background e fa tre passi violenti in direzioni casuali dettate dalle singole frasi di prosa ordinaria. Questa altissima varianza ("strattonamento" continuo dei pesi) disturba il *momentum* $\mu\left(\left.\frac{\partial E}{\partial w}\right\vert{}_{w^{(n-1)}}\right)$, rischia di sovrascrivere i canary a bassa esposizione (Gruppo 1 con sole $2$ esposizioni) e degrada la capacità del modello di rispondere correttamente alla domanda parafrasata.



**3. Perché il Mini-Batch (es. `batch_size = 4` o `8`) stabilizza la discesa**

* Quando calcoli la derivata parziale $\left.\frac{\partial E}{\partial w}\right\vert{}_{w^{(n)}}$ come media su $4$ o $8$ record contemporaneamente, gli errori casuali legati alle diverse frasi di prosa ordinaria tendono a **compensarsi e annullarsi tra loro** (riduzione della varianza), mentre la direzione sistematica del canary ripetuto si conserva e viene amplificata dal **momentum** passo dopo passo.
* Inoltre, come avete scritto nella slide *"Fine-Tuning as Continual Pre-Training"*, l'obiettivo è imitare le condizioni reali di addestramento degli LLM su GPU (come la Tesla T4 di Google Colab): nell'addestramento reale non si usa mai `batch_size = 1` (che sottoutilizza il parallelismo della GPU ed è numericamente instabile), ma si usa sempre il *mini-batch gradient descent*.



In conclusione, **il compromesso ottimale da dichiarare nella slide `Hyperparameters` è un mini-batch piccolo (`batch_size = 4` o `8`) con shuffling stratificato**: è abbastanza piccolo da garantire $64$ (o $32$) aggiornamenti per epoca senza mai far collidere due copie dello stesso canary nello stesso batch, ma abbastanza grande da mediare il rumore della prosa di background.