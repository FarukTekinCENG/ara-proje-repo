# .env file:
DB_NAME=postgres
DB_USER=postgres
DB_PASSWORD=

# if working with db: init postgres db with the tables 
!chmod +x table_def/bootstrap.sh
!./table_def/bootstrap.sh

# if working with db: split dataset train - test within db
python -m scripts.split_pool --fraction 0.2 --seed 42 --yes

# if working with csv: prepare balanced dataset
python -m scripts.prepare_balanced_dataset --force_download --target_per_class 500 --log_append

# plot graph: for all test data blocks: results.xlsx için
python scripts/plot_graph.py --input results/results1.xlsx