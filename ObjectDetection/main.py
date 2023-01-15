# This is a sample Python script.
from components.trainer import Trainer

if __name__ == '__main__':
    trainer = Trainer()
    trainer.train(10, "../../Seal/Train", plot_metric=True)
    trainer.evaluate("../../Seal/Train", plot_metric=True)
