## Setup Environment

Run the following commands to set up the environment:

```bash
chmod +x setup_env.sh
bash setup_env.sh
```

## Activate Environment

```
conda activate vat_qat
```


## Run QAT Pipeline

```bash
python run_qat.py                     # full pipeline
python run_qat.py --resume ckpt.pt    # resume training
python run_qat.py --eval-only         # skip training, export only
```