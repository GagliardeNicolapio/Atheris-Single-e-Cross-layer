# Atheris
Classificazione single e cross-layer di siti web con tecniche di machine learning.

## Sommario
Questo repository contiene gli algoritmi e le tecniche utilizzate per valutare la classificazione di siti web con l'approccio single e cross-layer. Il primo approccio consiste nell'utilizzare solo i dati del livello rete o solo i dati del livello applicazione. L'approccio cross-layer invece utilizza entrambi i livelli combinando i dati in quattro modi: AND-aggregation, OR-aggregation, XOR-aggregation e data-aggregation. 

I risultati ottenuti da questa implementazione e confrontati con quella di [Xu[2014]](https://www.proquest.com/openview/ff90d8aadeb570f0d1e7c11db664a18e/1?pq-origsite=gscholar&cbl=18750) possono essere riassunti in tre punti: 
 - l'algoritmo migliore è l'albero decisionale, infatti presenta un accuracy pari al 95%, falsi negativi al 5% e falsi positivi al 7%;
 - la AND-agg implica un incremento dei falsi negativi di 20 punti percentuali;
 - Naive Bayse, SVM e Logistic Regression presentano falsi positivi e falsi negativi alti.

Il repository è diviso in tre cartelle: 
- java: contiene l'implementazione della tecniche InformationGain e SubsetEvalutation implementati grazie alle API di Weka;
- python: contiene l'implementazione degli algoritmi di machine learning e delle tecniche utilizzate per la data cleaning e per l'aggregazione dei dati;
- dataset: contiene il dataset CSV su cui è stato applicato il processo di data cleaning ([dataset.csv](https://github.com/GagliardeNicolapio/Atheris/blob/master/dataset/dataset.csv)), il file .arff per le API di Weka ([datasetDataCleaningScalingFinal.arff](https://github.com/GagliardeNicolapio/Atheris/blob/master/dataset/datasetDataCleaningScalingFinal.arff)), i file CSV ottenuti da SubsetEvaluation e InformationGain ([datasetSubsetEval.csv](https://github.com/GagliardeNicolapio/Atheris/blob/master/dataset/datasetSubsetEval.csv), [infoGainDataset.csv](https://github.com/GagliardeNicolapio/Atheris/blob/master/dataset/infoGainDataset.csv)), e quattro file CSV per la tecnica di single-layer.

## Environment
- Python version: 3.10.0
- Java JDK version: Open JDK 17
- Weka version: 3.8.0
