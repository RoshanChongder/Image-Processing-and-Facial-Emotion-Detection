

class support:
    @staticmethod
    # Method to read CSV Files
    def read_CSV(path):
        try:
            file = pd.read_csv(path)
            return file
        except FileNotFoundError:
            print("CSV File not found at " + path)
            return None
        except:
            print(" Unknown error appeared ")
            return None

    @staticmethod
    def data_Addition():
        global Training_X, Training_Y, Testing_X, Testing_Y, data_set

        for row_count, row in data_set.iterrows():
            value = row['pixels'].split(' ')  # extracting the pixels as a list
            try:
                if 'Training' in row['Usage']:  # if the current column is for Training
                    Training_X.append(np.array(value, 'float32'))  # adding the pixels in the x axis
                    Training_Y.append(row['emotion'])  # adding emotion in the y axis
                elif 'PublicTest' in row['Usage']:  # if the current column is for testing
                    Testing_X.append(np.array(value, 'float32'))
                    Testing_Y.append(row['emotion'])

            except:
                print(" Error occurred at row number " + row_count)
                print("Data Set in that row is " + row)
