1. [ ] Klasifikācija uz kādām klasēm un datu balanss?
Jautājums-
- [ ] Vai vajadzētu atbrīvoties no kolonām ar pārāk daudz trūkstošām vērtībām?
- [ ] Vai vajadzētu atbrīvoties no kolonām ar pārāk daudz dažādām string vērtībām/prasa pārāk daudz embeddings?
![[loan_status_data_distribution 1.png]]
1. SOTA rezultāti? Vai tiešām nav citu publikāciju, kas izmanto? 
Laikam nav, nevaru atrast instances kad datu kopa būtu pielietota pētījumos, kas varbūt liecina ka tā ir slikta izvēle (bet pieņemu ka bakalaura darbam būs okei).
3. [ ] Uzlabot un sarakstīt SLR tabulas ^12f2
Sarakstīju datus, vēl neievietoju darbā. Šķietami atkal neizdodas korekti izmantot pluginu, nesaprotu kāda gan būtu atšķirība tajā ko daru.
- [x] Author/affiliation/country.
- [x] Tehniku apraksts- jāizdomā, ir daudz dažādas metodes, iespējams var iedalīt tā kā jau mēģināju, pirms, apmācības laikā un pēc. ✅ 2024-01-30
- [ ] Rezultāti- būs daudz tukša. NLL pa datu kopām, ECE.
5. SLR tabulās numurēt pētījumus, pirmajā tabulā ar cite keys, bet pēc tam tikai ar numerāciju


6. Sagatavot torch dataset un vismaz vienu metodi github, ar pandas sagatavot plots un datus (histogrammas, mean, std etc) priekš datu kopas nodaļas, salikt jau tajā 
- [ ] Implementēt parastu modeli un ansambli. 
- [x] Modelis (nenotestēts) ✅ 2024-01-30
- [ ] Eksperiments
- [x] Jautājums- kāds modelis ir atbilstošs/optimāls tabulāriem datiem? Oriģinālajā pētījumā izmantots viens hidden slānis, RELU. 
Neskaidrība kurus grafikus taisīt, ir ļoti daudz ieejas datu. Laikam priekš visiem ieejas/izejas datiem. 
- [x]  Grafiki izveidoti, nav ievietoti jo neesmu drošs kā tik daudz grafikus labāk ievietot. 
- [ ] Neskaitliskajām vērtībām neizdevās grafikus izveidot, bet atrisināšu.

Zemāk Python design_patterns kurss, tos darbus lūdzu pievieno folderos GIT piemēram ./design_patterns_1 utt, tas būtu nozīmīgi tālākai sadarbībai un arī esošā koda kvalitātes celšānai
- [ ] Design patterns kurss 1, laikam 1. var izlaist
- [ ] Design patterns kurss 2

# Īsumā
Sapratu datu apstrādi ar pandas, sagatavoju apmācībai datu kopu (trūkstošu aiļu izgriešana, strings pārvēršana par index un labels) (noderētu padoms par datu kopas apstrādi, vai atstāt ailes ar ļoti maz informācijas).
Izveidoju attēlus ar histogrammām.
Sāku rakstīt kodu eksperimentālajai daļai.
Notestēju datu kopas lietošanu ar parastu modeli (neglītā veidā)🤷‍♂.
Sarakstīju datus priekš country/affiliation SLR, method SLR, ne vēl rezultātu SLR. Vēl neko neievietoju darbā.