# 50.007_ML
50.007 Machine Learning Project Spring 2023 

## Steps to execute
Clone the repository on Github Desktop and open in VSCODE

part1.py -> defines all functions for calculating emission parameters 

Run part1.py 

This will write to 2 files : EN/dev.p1.out (prediction based on EN dataset) and FR/dev.p1.out(prediction based on FR dataset)

To see precision, recall and F scores, execute the following on the terminal :

```
python evalresult.py EN/dev.out EN/dev.p1.out

python evalresult.py FR/dev.out FR/dev.p1.out
```
