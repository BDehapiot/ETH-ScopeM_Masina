#%% Imports -------------------------------------------------------------------

import numpy as np
from skimage import io 
from pathlib import Path
from scipy.stats import pearsonr
from joblib import Parallel, delayed 
from czi_tools import extract_data, CD7_preview
from sklearn.linear_model import LinearRegression

#%% Comments ------------------------------------------------------------------

'''
Channel #1: Collagen I
Channel #2: Collagen Probe
Channel #3: Propidium Iodine

Condition tuple:
1st element - probe type     : 0 = ctrl 1 = probe
2nd element - probe conc.    : 0 = low  1 = high
3rd element - collagen conc. : 0 = low  1 = high

'''

#%% Conditions ----------------------------------------------------------------

conds = [
        (1,0,0), (1,0,0), (1,0,1), (1,0,1), (1,0,1), (1,0,1), (1,0,1), (1,0,1), (1,0,0), (1,0,0),
        (0,1,0), (0,1,0), (0,1,1), (0,1,1), (0,1,1), (0,1,1), (0,1,1), (0,1,1), (0,1,0), (0,1,0),
        (0,0,0), (0,0,0), (0,0,1), (0,0,1), (0,0,1), (0,0,1), (0,0,1), (0,0,1), (0,0,0), (0,0,0),
        (1,1,0), (1,1,0), (1,1,1), (1,1,1), (1,1,1), (1,1,1), (1,1,1), (1,1,1), (1,1,0), (1,1,0),
        (1,0,0), (1,0,0), (1,0,1), (1,0,1), (1,0,1), (1,0,1), (1,0,1), (1,0,1), (1,0,0), (1,0,0),
        (1,1,0), (1,1,0), (1,1,1), (1,1,1), (1,1,1), (1,1,1), (1,1,1), (1,1,1), (1,1,0), (1,1,0),
        ]

#%% Imports -------------------------------------------------------------------

czi_name = 'Masina_CD7_(3c-540s)_20x_16bits.czi'
czi_path = str(Path('data') / czi_name)
metadata, data = extract_data(czi_path, zoom=0.25)
# CD7_preview(czi_path, zoom=0.25, label_size=0.75, pad_size=4)

#%% Process -------------------------------------------------------------------

# Extract variable sets
wNames = sorted(set(metadata['scn_well']))
cNames = sorted(set(conds))
rowNames = sorted(set([wName[0] for wName in wNames]))
colNames = sorted(set([wName[1:] for wName in wNames]))

# Get pearson correlation coefficient
def get_pearson(i, scene):
    
    wName = metadata['scn_well'][i]
    idx = (rowNames.index(wName[0]), colNames.index(wName[1:]))
    C1 = scene[0,0,0,...]
    C2 = scene[0,0,1,...]
    C3 = scene[0,0,2,...]
    rC1C2, _ = pearsonr(C1.flatten(), C2.flatten())
    rC1C3, _ = pearsonr(C1.flatten(), C3.flatten())
    rC2C3, _ = pearsonr(C2.flatten(), C3.flatten())
        
    return wName, idx, rC1C2, rC1C3, rC2C3

sOutputs = Parallel(n_jobs=-1)(
    delayed(get_pearson)(i, scene) 
    for i, scene in enumerate(data)
    )

# Extract well outputs
wOutputs = []
for wName, cond in zip(wNames, conds):
    wOutput = [output for output in sOutputs if output[0] == wName]
    idx = wOutput[0][1]
    rC1C2_mean = np.mean([output[2] for output in wOutput])
    rC1C3_mean = np.mean([output[3] for output in wOutput])
    rC2C3_mean = np.mean([output[4] for output in wOutput])
    wOutputs.append((wName, idx, cond, rC1C2_mean, rC1C3_mean, rC2C3_mean))
    
rC1C2_mat = np.zeros((len(rowNames), len(colNames)), dtype=float)
rC1C3_mat = np.zeros((len(rowNames), len(colNames)), dtype=float)
rC2C3_mat = np.zeros((len(rowNames), len(colNames)), dtype=float)
for outputs in wOutputs:
    idx = outputs[1]
    rC1C2_mat[idx] = outputs[3]
    rC1C3_mat[idx] = outputs[4]
    rC2C3_mat[idx] = outputs[5]
    
X = [output[2] for output in wOutputs]
y = [output[3] for output in wOutputs]
model = LinearRegression()
model.fit(X, y)
print(model.coef_)
    
# Extract condition outputs
cOutputs = []
for cName in cNames:
    cOutput = [output for output in wOutputs if output[2] == cName]
    rC1C2_mean = np.mean([output[3] for output in cOutput])
    rC1C3_mean = np.mean([output[4] for output in cOutput])
    rC2C3_mean = np.mean([output[5] for output in cOutput])
    cOutputs.append((cName, rC1C2_mean, rC1C3_mean, rC2C3_mean))
