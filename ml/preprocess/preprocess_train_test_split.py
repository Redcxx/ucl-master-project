from src.train_test_split import split_dataset


def main():
    split_dataset(
        dataset_path=r'/datasets/sketch_simplication/sketch_simplification_good',
        train_test_ratio=0.8,
        random_seed=42
    )


if __name__ == '__main__':
    main()
