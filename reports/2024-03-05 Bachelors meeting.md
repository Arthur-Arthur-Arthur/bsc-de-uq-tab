# Report kopija

https://share.yellowrobot.xyz/quick/2024-2-20-76921984-20F0-4722-98B5-08BBF5490A38.html
2024-02-20 Meeting #8
 

TODO
Iepushot GIT un dokumentēt kodu (lai pushotu git izmanto SourceTree vai GitKranken). Github private repository for the project. Atsūtīt linku uz repozitoriju. Dot, ziņu, lai var review uztaisīt

Pārbaudīt vai modelim ir pareizi inputs / outputs (šobrīd 20, 20)

Izmanto optimizer - RAdam vai kādu citu jaunāku

Train kopā jāievieš weighted loss vai WeightedRandomSampler (izpēti)

Lai nomērītu cik laika vajag epoch, reģistrē metriku epoch_time un izmanto import tqdm lai redzētu progress bar ar laiku uz epoch

Pievienot metrikas: F1 un STD 

Rezultātu saglabāt: CSV, ClearML, TensorboardX

Disagreement formula? STD?

Aprakstīt ECE darbā, kā interprētēt -> accuracy

Aprakstīt un implementēt citas nenoteiktības metrikas - Feature importance, Conformal prediction, Brier score

Visas lietas daram test fāze - veikt eksperimentus, kas parādītu, ka metrika parāda nenoteiktību

bojāt randomly pieaugt ECE 

kas notiek ar ECE palielinot 

pie viena modeļa ECE 

temperature scale - salīdzīnāt

Veikt eksperimentuL Random sabojāt inputs un outputs validation fāzē - pārbaudīt un pierādīt, ka rezultāti korelē ar ECE prognezēto kalibrāciju modeļiem, kuri jūtīgi vai nejūtīgi pret nenoteiktību

Eksperimentus vari palaist uz vea-157 GPU nodes, zemāk pieeja
ssh connection: ssh malcovadmin@6.tcp.eu.ngrok.io -p13685 (Kitty uz Windows), Lai failus kopētu ar WinSCP  (tad vari ērti failiem piekļūt, PyCharm arī ir iebūvēts upload us SFTP)

password: malcovadmin2020

Linux Commands Useful http://share.yellowrobot.xyz/quick/2023-10-17-1BFC2160-7BFC-462D-9DB0-06EEC8850CFA.html

Taskus palaid screenā screen -rd arturs, lai atvienotos no screen to nekillojot izmantot crl+a un tad d

Netaisi nekādā gadījumā ciet screenu ngrok

image-20240220210455689

Pieejami 2x GPUs

image-20240220210511884

Vari izmantot conda activate conda_env vai izveidot jaunu env

Debugot remotly vari ar import pdb; pdb.set_trace()

Notes
 

Uncertainty Quantification and Deep Ensembles (https://arxiv.org/pdf/2007.08792.pdf)

 

Uncertainty for linear regression (y_prim = [mu, sigma]) balanced by loss

image-20240220201640218

image-20240220190255143

Datu kopa aizdevums cik atdos aizdevumu

# Uzdevumi
- [x] Updateot gitu (sanāca ļoti neveiksmīgi, bet darīts) ✅ 2024-03-04
- [x] Pievienot metrikas ✅ 2024-03-06
	- [x] STD ✅ 2024-03-04
	- [x] Time ✅ 2024-03-04
	- [x] ECE (otro reizi jo izdzēsās) ✅ 2024-03-04
- [ ] Jāizveido salīdzināšanas eksperimentu struktūra
- [ ] Jāiepazīstas ar attālināto lietošanu (Laikam nevajadzēs, CUDA izdevās)
- [x] Uzstādīt CUDA/gpu use, izdevās, bet tik un tā ļoti lēni ✅ 2024-03-06
- [ ] Atrisināt lēnumu 
https://pytorch.org/functorch/nightly/notebooks/ensembling.html iespējams varētu palīdzēt, bet tikai 4 reizes ātrāk
IZRĀDĀS ka batch size daudz par zemu, no nodarbībām laikam neguvu priekšstatu par to cik lielam šim parametram jābūt priekš lielas datu kopas
- [ ] 
# Bootstrapping/Boosting
https://openreview.net/pdf?id=dTCir0ceyv0