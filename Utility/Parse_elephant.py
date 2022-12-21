if __name__ == '__main__':

    with open("/media/remy/LaCie/AED/train/training_elephants.csv", 'r') as file:
        labels = {}
        print("reading file")
        for line in file.readlines():
            img_file_name, x, y = line.split(',')

            if img_file_name not in labels.keys():
                labels[img_file_name] = []

            labels[img_file_name].append((int(x), int(y)))
        print("writing new files")
        for key, values in labels.items():
            new_file = open("/media/remy/LaCie/AED/train/bboxes/" + key + ".txt", "w")
            for value in values:
                x, y = value
                new_file.writelines(str(x) + ' ' + str(y) + ' 8 8\n')
            new_file.close()



