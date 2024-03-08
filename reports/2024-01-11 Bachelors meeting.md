#bachelors 
# Datu kopa
https://www.kaggle.com/datasets/wordsforthewise/lending-club/data?select=accepted_2007_to_2018Q4.csv.gz
Loan dati ar daudz mainīgajiem (daļu vajadzētu izgriezt), no kuriem mēģināšu paredzēt loan pašreizējo statusu, tātad klasifikācijas uzdevums. Nav pieejami mēģinājumi
virs 100000 instances, ap 75 ipasibam (dazas planoju izgriezt)
Mēdz izmantot priekš datu analīzes darbiem (https://www.researchgate.net/publication/340395124_Project_Lending_Club_Data_Analysis)

https://archive.ics.uci.edu/dataset/2/adult
Cilvēku dati, mērķis ir noteikt vai income ir virs vai zem 50000
48842 instances, 14 ipasibas
# SLR
Iespējams varētu vēlāk uzlabot/paplašināt kritērijus, piemēram iekļaut OOD izvērtēšanu
Nesanāca vēl gluži smuki, nāksies uzlabot izskatu, domāju tad kad vispārīgi iešu pāri literatūras apskata daļai.
Jāpaplašina SLR aprakstu ar metodiku utml.
Jāpievieno citātus visam SLR pie tabulas
Varētu izmantot orthogonal arrays priekš eksperimentu dizaina? (nevermind, ar 2 metodēm tas neko nenozīmē)
# Modeļi
Praktiski sanāk implementēt repulsive de un izmēģināt tam pievienot temperature scaling, kas nešķiet pārlieku nozīmīgi :(. 
Mēģināju notestēt rakstu kodu, neizdevās vēl palaist kodu, čakarējos ar git, pip un kkas dīvains ar python versijām, maz pieredzes, bet lkm vajag vecāku pitona versiju lokāli environment. 
Iespējams kods paļaujas uz packages kas nestrādā vairs ar 3.10 python versiju? šķistu savādi bet iespējams
Vai ir zināms veids kā automātiski pārbaudīt vai requirements ir saderīgi
https://github.com/Arthur-Arthur-Arthur/bsc-de-uq-tab

# Piedāvājums
Katru nedēļu kad nav meetings arī atsūtīt rakstisku atskaiti, lai es sev motivētu progresu.

TODO:

1. Klasifikācija uz kādām klasēm un datu balanss?
    
2. SOTA rezultāti? Vai tiešām nav citu publikāciju, kas izmanto?
    

![image-20240111170710989](http://share.yellowrobot.xyz/upic/89f607cdf513f92a01e8f9586c7fad0f_1704985631.png)

3. Uzlabot un sarakstīt SLR tabulas
    
4. SLR tabulās numurēt pētījumus, pirmajā tabulā ar cite keys, bet pēc tam tikai ar numerāciju
    
5. Ja tabulas platas uzstādīt konkrētām lapām latex landscape
    
6. Izmantot Excel2Latex plugin korekti
    
7. RTU jaunais cite stils pieejams šeit:
    
8. Sagatavot torch dataset un vismaz vienu metodi github, ar pandas sagatavot plots un datus (histogrammas, mean, std etc) priekš datu kopas nodaļas, salikt jau tajā [http://share.yellowrobot.xyz/quick/2024-1-11-8B9BCDD6-C883-4FC5-B546-E36C65EA6850.html](http://share.yellowrobot.xyz/quick/2024-1-11-8B9BCDD6-C883-4FC5-B546-E36C65EA6850.html)
    
9. Teorētiskajā daļā izmantot materiālus no šiem [http://share.yellowrobot.xyz/quick/2024-1-11-B76D7A48-E41E-405A-BA90-6E6DAE2BC169.html](http://share.yellowrobot.xyz/quick/2024-1-11-B76D7A48-E41E-405A-BA90-6E6DAE2BC169.html)
    
10. Zemāk Python design_patterns kurss, tos darbus lūdzu pievieno folderos GIT piemēram ./design_patterns_1 utt, tas būtu nozīmīgi tālākai sadarbībai un arī esošā koda kvalitātes celšānai