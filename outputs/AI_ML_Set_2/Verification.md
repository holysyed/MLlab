# AI & ML Set 2 local run results

All seven Python files passed on Windows 11 with Python 3.12.10. The 15 runs covered the six standalone scripts, all six launcher command-line selections, the interactive launcher menu with `5b`, and the KNN generated-data fallback through both entry points. Windows rendered all 21 Matplotlib plot windows using TkAgg. The verification automatically closed each window after it rendered.

| Experiment | Observed result |
| --- | --- |
| 4(a) KNN classifier | 93 of 100 test predictions correct with the bundled CSV. The generated-data fallback scored 86 of 100 on its own split. |
| 4(b) State-based agent | Completed eight steps and finished with a score of 30. |
| 5(a) Naive Bayes | Both test predictions were correct. The new customer prediction was `1`. |
| 5(b) Uniform cost search | Found `A → B → E → G` at a minimum total cost of `7`. |
| 6(a) K-means | Assigned all 200 generated points to clusters of 67, 66, and 67. |
| 6(b) Logistic regression | Both test predictions were correct. The prediction for six hours was `1`. |

The ZIP omitted the lab’s original `purchase_dataset.csv`. The bundled file is a compatible substitute from the public Social Network Ads dataset. It has 400 rows and includes `Age`, `EstimatedSalary`, and `Purchased` in columns 3, 4, and 5, as the script expects. The other five experiments generate their inputs or define them in the code. [Dataset source](https://github.com/TheAlgorithms/Jupyter/blob/master/machine_learning/random_forest_classification/Social_Network_Ads.csv)

The required packages were already installed on this PC. The tested versions were NumPy 2.5.4, pandas 3.1.1, Matplotlib 3.11.2, and scikit-learn 1.9.1. All seven Python scripts match the supplied ZIP byte for byte.

To open the menu, run this command in PowerShell. Close each chart window to let the script continue.

```powershell
python 'C:\Users\aquib\Documents\Codex\2026-11-19\w\outputs\AI_ML_Set_2\AI_ML_Set_2_All_Experiments.py'
```
