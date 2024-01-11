1. Fokusējamies uz tabulārajiem datiem Izvēlēties un dokumentēt datu kopas - akciju cenas, laika apstākļi - kaggle?
    Par esošo pētījumu datu kopām:
		Regresija tabulārajos datos
		Klasifikācija attēliem
		Regresija sintētiskajiem datiem (vizuāliem rezultātiem)
		
    1. Cik daudz paraugi?
        Ievietoju bet neesmu 100% drošs par vērtībām
    2. Vai un kāds ir train, test, validation split?
        Nav iebūvētas vērtības šiem
    3. Kā nosaka nenoteiktību? Stdv?
        Datu kopas nesatur datus par nenoteiktību. Regresijas uzdevumos parasti iegūst intervālu/varianci, bet tā ne visos rakstos ir paredzēta precīzi atbilst patiesajam std. 
    4. Kāds augstākais acc? Ja ļoti augsta precizitāte vai cita metrika ļoti precīza, tad datu kopa pārāk tīra, lai testētu nenoteiktību
        Neesmu drošs kā šo noteikt datu kopām. Ievietoju pētījumu vērtības RMSE, NLL un citas. Ja nav ar to problēmas, izmantot citos pētījumos lietotās datu kopas varētu būt izdevīgi. 
        Lai ieviestu nenoteiktību var pievienot troksni datiem, noņemt ieejas mainīgos(gadījuma nenoteiktība), samazināt datu kopu (sistemātiskā).
1. SLR pievienot: Datu kopas, Citātu skaits gadā
    to-do
3. Izpildīt AI kursa uzdevumus, ja netiek pildīti uzdevumi nebūs pamatzināšanas, lai veiktu pētījumu
    10ais nav vel, nesanaca piekerties
4. Overleaf sākt rakstīt **SLR un Saistītie pētījumi** nodaļas
    Nedaudz iesākts, jāraksta 10 lpp priekš bakalaura priekšmeta, tāpēc fokusēšos uz saistītie pētījumi.
5. Vēlams SLR rakstīt šādā veidā (vertikālas tabulas): [http://share.yellowrobot.xyz/quick/2023-11-8-C8BA5CFB-675C-477E-82B8-8E9BD405488A.pdf](http://share.yellowrobot.xyz/quick/2023-11-8-C8BA5CFB-675C-477E-82B8-8E9BD405488A.pdf)
# Mani jautājumi
- Vai vajadzētu SLR atlasīt tikai Deep Ensemble uzlabojošus rakstus?
- Vai var izmantot iepriekš lietotās datu kopas?
- Apskatītajos rakstos daudz vairāk tiek lietota attēlu klasifikācija, varbūt darīt to?
- Es labprāt salīdzinātu metožu kombinēšanu, tās efektu

# Tehniku uzskaitījums
- MC-dropout
- Large single model

- DE
- Repulsive DE
- Max diversity DE
- Hyperparameter DE
- Architecture DE
- Uncertainty quantification Loss function (AvUC or AUCE)
- Adversarial training