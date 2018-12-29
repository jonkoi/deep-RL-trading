import csv, pickle, math

    """
    Create dataset based on close price - splitting and pickling
    One price only
    """
pair = "EURUSD"
time_scale = "hour"
price = "close"
general_title = pair + "_" + time_scale + "_" + price + "_"
db = []


with open(pair + '_' + time_scale + '.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in spamreader:
        db.append([float(row[5])])

formated_db = []
for i in range(db):
    idx = math.floor(i/180)
    seq = np.array(db[idx:idx + 180, :])
    title = general_title + str(idx)
    formatted_db.append((seq, title))

pickle.dump(formatted_db[0:270], "../" + pair + "_" + time_scale + "_" + price + "_train/db.pickle")
pickle.dump(formatted_db[270:-1], "../" + pair + "_" + time_scale + "_" + price + "_test/db.pickle") 
