# IBM Advanced Data Science Capstone

The last part of the specialization requires a final artifact: a end-to-end data science project following a set of given guidelines regarding project methodology, quality assurance and documentation.

* First of all, the use case and final goal that is being pursued needs to be sketched and outlined.
* Furthermore, a wrap up of tech stack and tools should be given
* The ground truth data shall be described, togehter with the way the need to be postprocessed in order o feed them to the main project machinery
* The ML models/techniques in use (plus if applicable, a comparison of these regarding predictive power) must be given

## Use Case
Self employed IT professionals are obliged to give a yearly declaration of categorized revenue and expenses, an ordinary accounting task mostly based on transaction data from banking accounts. In order to create the declaration suitable for the IRS, the banking transaction need to be categorized and translated into accounting statements. In case this is not done on a weekly basis, the upcoming IRS declaration generates workload, stress and inoperability on the freelance consultant. And typically, it is NOT done on a weekly basis, but an annual one. When the deadline draws near.

The strategies applied in order to keep the IRS deadline for the declaration typically are:
* allow one week for manual bookkeeping, walk through banking transaction, translate into account statements
* allow for 2,000€ bill and have the accountant do it

In order to save time and/or money, it would be convenient  to have an intelligent agent do the work of categorizing banking transaction, attributing them in a local or cloud based data store, perform automated accounting on it and hence generate the IRS declaration in a reproducible, reliable and automated way.

The given project aims at answering whether this is feasible with the following data pipeline:
* Download historical banking transactions using the bank's API
* Manually/semi-automatically categorize this data stock
* Use this as ground truth data for training an AI model
* Put the model into production
* Stream incoming bank transaction and apply classification
* Feed result into automated tax declaration  

The non-functional parameters to the whole are
* minimize effort for establishing this spike
* re-use whatever seems reasonable and feasible, esp. pretrained models or proven AI architectures
* apply well-known technologies

## Infrastructure

### Data storage and pipeline
All preparatory and maintaining actions are backed by a GitLab repository. It also holds the definition of a CI/CD pipeline to perform generation of accounting information. Sometimes such is requested ad hoc by the bank when asked to allow for micro loans (cash flow injections, debit frames) and must thus be able to be triggered manually, but typically is driven by recurring annual deadlines and reporting needs (IRS, end of fiscal year, end of calendar year).

The pipeline is twofold:
* The productive pipeline reacts on changes to the incoming CSV files. It applies the necessary transformations and subjects the data to the pre-trained productive model.
* The classifier training pipeline reacts on changes to the incoming CSV files. It then preprocesses them into the format apt for the model training.

(Sketch of pipeline)

## Tech Stack and Tools

Python (Colab, PyCharm), Sci-Kit Learn, Keras, FastAI, DL4J, BigQuery, OBOX BI, Google Cloud

## Strategy and Scenarios
The project demonstrates the evolution of the problem solving algorithm. It was originally based on an idea that turned out to yield poor results, was discarded and re-implemented on completely different methodology. The two scenarios thus serve the demand after alternative models applied and their distinctive quality measures.

### Scenario 1: Failure
The initial idea was to represent the data as visual images and have pre-trained / standardized CNNs do the classification. The advantages of this approach were estimated as being the following
* A highly pre-trained and standardized model could be reused and expensive training would be unnecessary 
* feature engineering would merely be made up of visualizing all or relevant portions of the data, e. g. as QR code

Both of them were favourable because of a narrow time and resource budget. Despite different re-engineerings of this approach, model accuracy never exceeded 30%. This path was thus deemed useless to follow.

### Scenario 2: Success
Alternatively and subsequently, a more traditional approach was taken, the data was feature engineered along an NLP strategy (bag-of-words) and a Gaussian Naive Bayes classifier was applied, eventually yielding a model accuracy of 96-98 percent.

## Data

### Provenience, Quality and Privacy
The data is real world data as commercially provided by the banking institution. It is meant to be the basis for a legally binding document issued to the IRS and thus must not be subject to alteration in the process of classification. 
As it consitutes private, personal and economic information, it is not shared in any of the project documents and anonymized where demonstrational snapshots are exhibited.

### Retrieval
The CSV download is triggered manually from the online banking portal by the authenticated user. Negotiations are up as to whether an automated download per API request can be facitlitated. This would greatly enhance the flexibility of accounting statement creation and thus a major future target.

### Amount
The data is made up of transaction data for
* years 2018, 2019 and 2020
* one business, one private and one credit card account
* a total number of 2871 lines
### Layout
The data is made up of UTF-16 encoded localized CSV files representing banking transaction stemming from one credit card and two banking accounts. Localization issues are:
* German date format is used, so 2018-31-12 would be represented as 31.12.2018
* German number format is used, so 3.14E06 would be represented as 3.140.000,00
* diacritical characters show up in text fields, e. g. ÄÖÜäöüß
This needs to be taken care of in data preparation / feature engineering.

The CSV headers are:
``"Buchungsdatum";"Partnername";"Partner IBAN";"Partner BIC";"Partner Kontonummer";"Partner Bank-Code (BLZ)";"Betrag";"Währung";"Buchungs-Info";"Buchungsreferenz";"Notiz";"Highlight";"Valutadatum";"Virtuelle Kartennummer";"Bezahlt mit";"App";"Zahlungsreferenz";"Mandats ID";"Creditor ID"``

A sample data entry for an ordinary banking account transaction would be
``"03.12.2018";"DAHOAM SCHARDING 4780";"";"";"77319989";"20320";"-90,55";"EUR";"POS 90,55 AT K3 30.11. 19:06 DAHOAM SCHARDING 4780 040";"203201812012ALB-00C3Q0NDVL9C";"";"0";"30.11.2018";"";"";"";"";"";""``

A sample data entry for credit card transaction would be
``"31.12.2018";"";"";"";"40005190700";"20111";"-83,62";"EUR";"Dahoam";"201111812292ALV-230859089555";"";"0";"29.12.2018";"";"";"";"";"";""``

The raw features deemed most relevant for classification are "`Buchungs-Info`" (transaction text) and "`Partnername`" (transaction partner name). As can be seen in the above example, transactions on behalf of the same partner ae encoded differently depending on whether the account involved is ordinary checking account or credit card.

### Classification target
Classifying a single transaction means to assign to it one single category. Each category is a semantically hierarchical description of the expense or income domain of the flow of money:
* A statement categorized as `Privat.Leben.Nahrung` (and a negative amount of money flown) would mean that the transaction describes a private expense that is necessary in the sense that some alimentation was bought.
* A statement categorized as `Business.Honorar.Beratung` (and a positive amount of money flown) would indicate the reception of a commercial payment for consultancy

The sense of the classification is to attribute the transaction semantically and make it apt for filtering and aggregation by standard postprocessing means, e. g. SQL queries, data streaming asf.

The classification categories are:

`Business.Administration.Finanzamt
Business.Administration.Sozialversicherung
Business.Administration.Sozialversicherung.Andja
Business.Administration.Sozialversicherung.Wolf
Business.Administration.Steuerberatung
Business.Administration.Tourismus
Business.Administration.Wirtschaftskammer
Business.Bank.Einlage
Business.Bank.Entnahme
Business.Bank.Spesen
Business.Bank.Übertrag
Business.Honorar
Business.Infrastruktur.Geräte
Business.Infrastruktur.Medien
Business.Mobilität.Zug.Ausland
Business.Mobilität.Zug.Inland
Privat.Auto.Kraftstoff
Privat.Auto.Kredit
Privat.Auto.Instandhaltung
Privat.Auto.Strafe
Privat.Auto.Versicherung
Privat.Gesundheit.Arzt
Privat.Gesundheit.Heilbehelfe
Privat.Haus.Garten
Privat.Haus.Geräte
Privat.Haus.Kredit
Privat.Haus.Kredit.Versicherung
Privat.Haus.Instandhaltung
Privat.Haus.Strom
Privat.Haus.Versicherung
Privat.Haus.Wasser
Privat.Leben.Hygiene
Privat.Leben.Kirche
Privat.Leben.Kleidung
Privat.Leben.Nahrung
Privat.Leben.Rauchen
Privat.Leben.Schule
Privat.Leben.Wirtshaus
Privat.Medien`


## ETL / Feature Engineering

### Scenario 1
As stated above, the main driver for the naive expectation and hope in this approach was that the underlying image recognition algorithm out of itself was smart enough to deduct intrinsic resemblence of banking transactions by merely comparing their visual representation (as QR code, in the chosen setup). So the need for data preprocessing was little, it consisted of merely transforming each single transaction line from the CSV

``"03.12.2018";"DAHOAM SCHARDING 4780";"";"";"77319989";"20320";"-90,55";"EUR";"POS 90,55 AT K3 30.11. 19:06 DAHOAM SCHARDING 4780 040";"203201812012ALB-00C3Q0NDVL9C";"";"0";"30.11.2018";"";"";"";"";"";""``

into a grayscale PNG of standard QR size:

![](https://i.imgur.com/lQYp3fk.png)


using Python libraries `pyqrcode` and `pypng`:

```
...
          filename = "data/accounting_categories/" + Kategorie + "/img_" + Buchungsdatum + Betrag + ".png"
          print("Generating QR code at " + filename)
          feature = aline
          qr_code = pyqrcode.create(feature, error='L', version=27, mode='binary')
          qr_img = qr_code.png(filename, scale=6, module_color=[0, 0, 0, 128], background=[0xff, 0xff, 0xcc])
...
```

The resulting images were put in a folder hierarchy representing the ground truth classification to be subsequently fed to Keras ImageDataGenerator.

### Scenario 2
Alternatively, in order to surmount the shortcomings of the initially chosen algorithm, in this setup, a traditional NLP approach was used. Each CSV entry was processed as follows:
* transaction text and partner name were concatenated, non-alphabetical characters were stripped off
* the resulting text was lowercased
* and tokenized using NLTK's word tokenizer
* stop words (German language) were purged
* optional Porter-stemming (German language) was performed

```
...
data = []
for i in range(groundTruth.shape[0]):
    tx = str(groundTruth.loc[i, 'Buchungstext']) + " " + str(groundTruth.loc[i, 'Partnername'])

    # remove non alphabetic characters
    tx = re.sub('[^A-ZÄÖÜa-zäöüß0-9]', ' ', tx)

    # make words lowercase
    tx = tx.lower()

    # tokenizing
    tokenized_tx = wt(tx)

    # remove stop words and stemming
    tx_processed = []
    for word in tokenized_tx:
        if word not in set(stopwords.words('german')):
            tx_processed.append(spell(stemmer.stem(word)))
            tx_processed.append(stemmer.stem(word))

    tx_text = " ".join(tx_processed)
    data.append(tx_text)
...
```

Consequently, out of the original

``"05.12.2018";"Erste Bank Oesterreich";"AT052011140005191900";"GIBAATWWXXX";"40005191900";"20111";"-2.326,38";"EUR";"s Kreditkartenrechnung Nov. 2018 s Visa Card Business Gold Kartenendnummer 6011";"201111812032ALV-181203018387";"";"0";"05.12.2018";"";"";"";"";"170602ALV-270/72107807142";"AT68ZZZ00000004435"``


the resulting example line

``['kreditkartenrechnung', 'nov', '2018', 'visa', 'card', 'busi', 'gold', 'kartenendnumm', '6011', 'erst', 'bank', 'oesterreich']``

would be fed to NLTK's `CountVectorizer` and turned into a bag of words. This one, in turn serves as input for model training.

## Modelling

### Scenario 1
Model definition:
```
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 254, 254, 32)      896       
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 252, 252, 64)      18496     
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 126, 126, 64)      0         
_________________________________________________________________
dropout (Dropout)            (None, 126, 126, 64)      0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 124, 124, 32)      18464     
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 122, 122, 64)      18496     
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 61, 61, 64)        0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 61, 61, 64)        0         
_________________________________________________________________
flatten (Flatten)            (None, 238144)            0         
_________________________________________________________________
dense (Dense)                (None, 128)               30482560  
_________________________________________________________________
dropout_2 (Dropout)          (None, 128)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 128)               16512     
_________________________________________________________________
dropout_3 (Dropout)          (None, 128)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 40)                5160      
=================================================================
Total params: 30,560,584
Trainable params: 30,560,584
Non-trainable params: 0
```
### Scenario 2
The overall sample of 2871 curated entries was split into train and test set at a 80% : 20% ratio
yielding 2153 training items and 718 test items:
```
# split train and test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)
```

and the model was defined as:
```
# Naive Bayes
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

```


## Performance

### Scenario 1
Loss and accuracy:

![](https://i.imgur.com/U60f9C2.png)

### Scenario 2

Confusion Matrix:
![](https://i.imgur.com/GRGw9wy.png)

Classification Report:
```
                                                  precision    recall  f1-score   support

               Business.Administration.Finanzamt       1.00      1.00      1.00         1
Business.Administration.Sozialversicherung.Andja       0.00      0.00      0.00         1
               Business.Administration.Tourismus       1.00      1.00      1.00         1
       Business.Administration.Wirtschaftskammer       0.00      0.00      0.00         1
                           Business.Bank.Einlage       1.00      1.00      1.00         1
                          Business.Bank.Entnahme       1.00      1.00      1.00        78
                            Business.Bank.Spesen       1.00      0.98      0.99        45
                          Business.Bank.Übertrag       1.00      0.87      0.93        69
                                Business.Honorar       1.00      1.00      1.00        10
                    Business.Infrastruktur.Cloud       1.00      1.00      1.00        27
                   Business.Infrastruktur.Geräte       1.00      1.00      1.00        19
                   Business.Infrastruktur.Medien       0.89      1.00      0.94        24
                   Business.Mobilität.Nahverkehr       1.00      0.90      0.95        10
                          Business.Mobilität.Zug       1.00      0.97      0.99        35
                  Business.Mobilität.Zug.Ausland       1.00      1.00      1.00         7
                   Business.Mobilität.Zug.Inland       0.88      1.00      0.93         7
                             Business.Unterkunft       0.86      0.86      0.86         7
                      Privat.Auto.Instandhaltung       1.00      0.25      0.40         4
                          Privat.Auto.Kraftstoff       0.88      1.00      0.93         7
                              Privat.Auto.Parken       1.00      1.00      1.00         6
                              Privat.Auto.Strafe       1.00      1.00      1.00         5
                        Privat.Auto.Versicherung       1.00      1.00      1.00         3
                          Privat.Gesundheit.Arzt       0.90      1.00      0.95         9
                   Privat.Gesundheit.Heilbehelfe       1.00      1.00      1.00        11
                              Privat.Haus.Garten       0.00      0.00      0.00         2
                              Privat.Haus.Geräte       1.00      1.00      1.00        15
                      Privat.Haus.Instandhaltung       1.00      1.00      1.00         1
                              Privat.Haus.Kredit       0.86      1.00      0.92         6
                 Privat.Haus.Kredit.Versicherung       1.00      1.00      1.00         3
                               Privat.Haus.Strom       0.86      1.00      0.92         6
                            Privat.Leben.Hygiene       0.90      1.00      0.95        28
                             Privat.Leben.Kirche       0.00      0.00      0.00         0
                           Privat.Leben.Kleidung       1.00      0.67      0.80         3
                            Privat.Leben.Nahrung       1.00      1.00      1.00       150
                            Privat.Leben.Rauchen       0.93      0.96      0.94        26
                             Privat.Leben.Schule       0.47      1.00      0.64         8
                          Privat.Leben.Wirtshaus       0.96      0.86      0.91        28
                                   Privat.Medien       1.00      1.00      1.00        54

                                        accuracy                           0.96       718
                                       macro avg       0.85      0.85      0.84       718
                                    weighted avg       0.97      0.96      0.96       718

```


## Summary
The task of categorizing banking transactions that reside as raw CSV batch data into semantically structured and labelled ones was begun with high hopes and insufficient analysis in the first. The naive hope that trained models and algorithmic power alone would be able to automagically spot resemblences in QR code images created out of unmodified input data was disappointed in that the approach's performance - despite measures taken as drastic as boosting the model - never exceeded 30% accuracy. 

On the contrary, even though not optimized to the utmost extent, basic NLP bag of words algorithm and Gaussian Naive Bayes Classifier quickly yielded an acceptable 96+% - enough to satisfy the requested threshold of 95%. 

