http://share.yellowrobot.xyz/quick/2024-1-30-B5B8506C-65F5-44C9-A9EE-15551B27FAEC.html
# Report kopija
2024-01-30 Meeting #7
 

TODO:

Aprakstīt un īstenot metodoloģijas, lai attīrītu datus no outliners
https://neuraldatascience.io/5-eda/data_cleaning.html

https://www.datacamp.com/tutorial/tutorial-data-cleaning-tutorial

Univariate method: Look for data points with extreme values on a single variable and remove or transform them. This includes techniques like removing points above or below a threshold, or transforming using log or sqrt.

Multivariate method: Train a predictive model on the data and identify points with a very high error. This catches unusual combinations across multiple variables. The points can then be removed or downweighted.

Minkowski error: Use a robust error metric like Minkowski error when training models. This reduces the influence of potential outliers on the model coefficients.

Capping: Instead of removing extreme values, cap them to some maximum threshold tuned on the data distribution. This retains more information.

Clustering: Use clustering algorithms to identify outliers as points distant from cluster centers. (šeit var ar Spectral clustering/k-means un tad ar PCA verificēt)

Visualization: Manually inspect plots like boxplots, scatter plots, and parallel coordinate plots to identify outliers.

 

Numpy Pandas izmantot, lai aizstātu trūkstošās vērtības ar mediānas vērtibām

Salabot kategoriju inputs, noskaidrot kategoriju outputs - ko nozīmē 3 klase

Mazāk sastopamās kategorijas jāapvieno kā any
image-20240130233137529

Output retās kategorijas paraugi vispār jānoņem

image-20240130182245332

 Test un validation kopās jābūt vienādam klašu skaitam, train kopā jābūt pie loss weights

Svaigs piemērs labam SLR darbam - tavs būs līdzīgs, tikai jādabon vēl dziļākā līmenī: http://share.yellowrobot.xyz/quick/2024-1-30-87D89640-5213-41B3-A5BF-602EEFB38DE1.pdf

Noskaidro kādas vēl citas metrikas bez ECE lieto un kā novērtēt vai modeļi tiešām dos nenoteiktus rezultātus, iedodot out-of domain vai noisy input

Sākt eksperimentālo daļu

Plāns salīdzināt

Random initialization same hyper-params

Izvēlēties dažādus modeļus pēc recall, F1, loss

Izvēlēties dažādas arhitektūras ansamblī

Citas metodes, kuras dokumentētas SLR

Noskaidrot vai ir pozitīva ietekme uz F1 salīdzinot ar tāda paša izmēra 1 modeli

 

 

 

 

Notes:

Vēl citi izmanto

MonteCarlo dropout

hameltonain monte carlo ķēdes

image-20240130183851336

 

 

ECE nenoteiktības metrika (izejas dati, pārliecība)
sadalīt 10 bins, pārliecība 

Expected calibration errot

 

image-20240130183205286

 

 

fuzzy inputs => korelācija ar outputs 

 

Tāda paša izmēra modelis vs ansamblis pētījums
# Lielie uzdevumi
- [ ] Izplānot eksperimentālos uzdevumus
[[Plāns eksperimentālajai daļai]]
Nav līdz galam sastrukturēts, bet zinu ko darīt.
- [ ] Realizēt eksperimentālo daļu 
[[Kods ansambļu eksperimentālajai daļai]]
Bāze ir gatava bet vajag vēl to papildināt un realizēt vairāk metodes.
Uz datora nav iespējams, pašlaik testēju ar 0,01 pilnās datu kopas, ar pilno ir minūtes uz epohu
- [x] Sarakstīt kodu datu attīrīšanai ✅ 2024-02-19
[[Kods datu attīrīšanai]]
Uzskatu ka pabeigts. Neesmu īsti attīrījis skaitliskos outliers, bet gan kategoriju datus.
- [ ] Sarakstīt literatūru par ekspeimentālo daļu
Šis pēc pārējā