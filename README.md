# GRAIN


## Requirements

Python **3.10** (verified on 3.10.20).

```bash
conda create -n py310 python=3.10
conda activate py310
pip install -r environments.txt
```

---

## The prior

Can be downloaded at (commented out for double blind review)<!--[commented out for double blind review](https://drive.google.com/drive/folders/1-xkfbAMVi_SD7FywKDtvtGCcV0u4n-hG?usp=drive_link)-->

or 

you can use the convert_prior.py file if you have generated it from the synthetic pipeline

the you can put in the prior file in /train_and_inference
```
prior/m3_yearly_200.y.npy      
prior/m3_yearly_200.meta.npz    
```

Regenerate only if the generator CSV changes:

```
python convert_prior.py --csv "<path>/m3_yearly_200.csv" --out prior/
```


## Training

```bash
python train.py -c config.example.yaml
```





## Inference
Weights can be downloaded commented out for double blind review<!--[here](https://drive.google.com/drive/folders/1lBZ48EqHlU61-7Ua8a6jcl8q0SXazlHK?usp=sharing)-->

M3 yearly validation files can be downloaded located at the m3_yearly_validation folder.


Scores a directory of `T{n}.csv` files (columns `date,OT`):

```bash
python inference.py \
    -w runs/<run>/checkpoints/model.ckpt \
    -d ../m3_yearly_validation \
    -p 6 -n 1 \
    -o ./GRAIN/m3_yearly
```

| flag | meaning |
|---|---|
| `-w` | checkpoint |
| `-d` | directory of `T{n}.csv` |
| `-p` | horizon, 1..6 (the head width — one checkpoint serves every horizon) |
| `-n` | rolling origins; `1` = forecast the final `-p` points only |
| `-o` | output directory |
| `--save-txt` | per-series `.txt` reports + `average_metrics.txt` |
| `--save-plots` | per-series `.png` (median + 10–90% band) |
| `--limit N` | first N series only, for a smoke test |
| `--device` | defaults to `cuda` when available |

## Statistical Comparison

Compare with the statistical methods in stat_comparison folder using the ipynb file.


## GIFT-Eval
clone the gift eval repository and put the GRAIN.ipynb file in the notebooks folder.

Run the file and you can see the results in the results folder.

then run results.ipynb and plots.ipynb to see how it performs against other gifteval benchmarks.
