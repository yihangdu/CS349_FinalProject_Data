- Diagnostics.xlsx
Original dataset. Rhythm column classifies the patterns of ECG. Beat column denotes the diagnosed disease. Since for SB rhythm, it is hard to tell if a patient is health or not, we try to use machine learning to classify patients with SB rhythm as healthy (N for negative) and unhealthy (P for positive).

- Diagnostics.csv
Same as Diagnostics.xlsx but in .csv format.

- Diagnostics_processed.csv
Unnecessary columns (FileName, Rhythm, PatientAge, and Gender) are dropped. Only SB rhythm samples are kept. Re-encoding Beat to N and P; NONE is encoded to N and others are encoded to P. All data are normalized.

- Diagnostics_train.csv, Diagnostics_valid.csv, Diagnostics_test.csv
These three files are split from shuffled Diagnostics_processed.csv with the ratio of 6:2:2.

- Diagnostics_preprocessing.ipynb
Some simple preprocessing code to do what we have described above. If un-normalized data is needed, simply comment out the for-loop that normalizes the data and re-run all blocks.

- ForWeka
The directory contains scripts and files needed to do prediction with Weka. For most of algorithms assembled in Weka, we have 69% overall accuracy.