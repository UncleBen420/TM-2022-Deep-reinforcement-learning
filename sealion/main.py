from model import train

if __name__ == '__main__':
    model, history = train("../../Seal/Classification/Train", "../../Seal/Classification/Test", num_epochs=300)
