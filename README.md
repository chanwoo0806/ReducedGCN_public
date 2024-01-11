# ReducedGCN

## Environmental Settings
Refer to the requirements.txt

## Usage
You can run main.py with various options. For more detailed information about the arguments, you can check by running the command:

```bash
cd code
python main.py --help
```

To use the best hyperparameters for each dataset, run the main.py script using the provided YAML file. Here's an example of how to use it:

- Kindle Store
```bash
cd code
python main.py --model rgcn --config_file ../config/ks10.yaml
```

- MovieLens 1M
```bash
cd code
python main.py --model rgcn --config_file ../config/ml1m.yaml
```

- Yelp2018
```bash
cd code
python main.py --model rgcn --config_file ../config/yelp2018.yaml
```