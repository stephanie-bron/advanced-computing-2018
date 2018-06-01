import numpy as np
import os
import pandas as pd
import glob

def createFileRealData(filesPath, tmin, tmax, filename):

    if os.path.exists(filename):
        os.remove(filename)
        print('File ', filename,' found, removing it...')

    print('Creating data file', filename)

    files = glob.glob(filesPath+'/*1day.dat')

    sources = []

    for f in files:
        sources.append(f.split('/')[-1].rsplit('_1day.dat')[0].rsplit('lc_')[1])
        data = pd.read_csv(f, sep="\t", header=0, index_col=False)

        counts_norm = data['counts'] / data['Exposure']
        error_norm = data['Error'] / data['Exposure']
        mjd = data['startT']

        bins = np.arange(tmin, tmax, 1)

        missingEntries = np.setdiff1d(bins, mjd)
        for i in missingEntries:
            mjd = mjd.append(pd.Series(i))
            counts_norm = counts_norm.append(pd.Series(0))
            error_norm = error_norm.append(pd.Series(0))

        serie = pd.concat([mjd, counts_norm, error_norm])
        serie_transposed = serie.to_frame().transpose()

        s_t = np.asarray(serie_transposed)


        with open(filename, 'a') as f0:
            np.savetxt(f0, s_t, delimiter=",")

    # write source list in a file to find out later which ones where selected
    source_file = open('sources.txt', 'w')

    for s in sources:
        source_file.write(s + '\n')
    source_file.close()


def createSimulationFile(mu, sigma, amp, tmin, tmax, nloopsSig, nloopsBkg, filename):

    def gaussian(a, b, c, x):
        gauss = a * np.exp(-pow((x - b), 2) / (2 * c * c))
        return gauss

    def makeLCSig(mu, sigma, amp, tmin, tmax, filename):

        r = np.random.normal(mu, sigma, 100)
        gauss = gaussian(amp, mu, sigma, r)

        bins = np.arange(tmin, tmax, 1)
        inds = np.digitize(r, bins)

        gaussDict = {}

        for counter, value in enumerate(inds):
            gaussDict[bins[value - 1]] = gauss[counter]

        rand = np.random.uniform(tmin, tmax, size=100)

        # digitize: Each index i returned is such that bins[i-1] <= x < bins[i]
        inds = np.digitize(rand, bins)
        rand_binned_val = []

        for i in inds:
            rand_binned_val.append(bins[i - 1])

        missingEntries = np.setdiff1d(bins, rand_binned_val)
        for i in missingEntries:
            rand_binned_val.append(i)

        import collections

        counter_rand_gauss = collections.Counter(rand_binned_val)

        for k, v in counter_rand_gauss.items():
            if k in gaussDict:
                counter_rand_gauss[k] = [v + gaussDict[k]]

            else:
                counter_rand_gauss[k] = [v]

        ex1 = 1e7
        ex2 = 5e7

        err1 = 1.0
        err2 = 5.0

        for k in counter_rand_gauss.keys():
            counter_rand_gauss[k].append(float(np.random.uniform(err1, err2, 1)))
            counter_rand_gauss[k].append(float(np.random.uniform(ex1, ex2, 1)))

        od_res = collections.OrderedDict(sorted(counter_rand_gauss.items()))

        b_rand_gauss = list(od_res.keys())
        v_rand_gauss = list(od_res.values())
        b_rand_gauss = np.asarray(b_rand_gauss)
        v_rand_gauss = np.asarray(v_rand_gauss)

        counter_rand = collections.Counter(rand_binned_val)

        for k, v in counter_rand.items():
            counter_rand[k] = [v]

            counter_rand[k].append(float(np.random.uniform(err1, err2, 1)))
            counter_rand[k].append(float(np.random.uniform(ex1, ex2, 1)))


        temp_gauss = np.concatenate(
            (b_rand_gauss, v_rand_gauss[:, 0] / v_rand_gauss[:, 2], v_rand_gauss[:, 1] / v_rand_gauss[:, 2]))
        temp_gauss = np.insert(temp_gauss, len(temp_gauss), 1)
        s_gauss = pd.Series(temp_gauss)
        serie_gauss_transposed = s_gauss.to_frame().transpose()
        s_gauss_t = np.asarray(serie_gauss_transposed)

        with open(filename, 'a') as f0:
            np.savetxt(f0, s_gauss_t, delimiter=",")




    def makeLCBkg(tmin, tmax, filename):


        bins = np.arange(tmin, tmax, 1)
        rand = np.random.uniform(tmin, tmax, size=100)

        # digitize: Each index i returned is such that bins[i-1] <= x < bins[i]
        inds = np.digitize(rand, bins)
        rand_binned_val = []

        for i in inds:
            rand_binned_val.append(bins[i - 1])

        missingEntries = np.setdiff1d(bins, rand_binned_val)
        for i in missingEntries:
            rand_binned_val.append(i)

        import collections

        ex1 = 1e7
        ex2 = 5e7

        err1 = 1.0
        err2 = 5.0

        counter_rand = collections.Counter(rand_binned_val)

        for k, v in counter_rand.items():
            counter_rand[k] = [v]

            counter_rand[k].append(float(np.random.uniform(err1, err2, 1)))
            counter_rand[k].append(float(np.random.uniform(ex1, ex2, 1)))

        od_res = collections.OrderedDict(sorted(counter_rand.items()))

        b_rand = list(od_res.keys())
        v_rand = list(od_res.values())

        b_rand = np.asarray(b_rand)
        v_rand = np.asarray(v_rand)

        temp_rand = np.concatenate((b_rand, v_rand[:, 0] / v_rand[:, 2], v_rand[:, 1] / v_rand[:, 2]))
        temp_rand = np.insert(temp_rand, len(temp_rand), 0)
        s_rand = pd.Series(temp_rand)
        serie_rand_transposed = s_rand.to_frame().transpose()
        s_rand_t = np.asarray(serie_rand_transposed)

        with open(filename, 'a') as f0:
            np.savetxt(f0, s_rand_t, delimiter=",")



    if os.path.exists(filename):
        os.remove(filename)
        print('File ', filename,' found, removing it...')

    print('Creating data file', filename)

    for i in amp:
        for j in sigma:
            for n in range(nloopsSig):
                makeLCSig(mu, j, i, tmin, tmax, filename)

    for n in range(nloopsBkg):
        makeLCBkg(tmin, tmax, filename)



if __name__ == '__main__':

    createFileRealData(filesPath, tmin, tmax, filename)


    createSimulationFile(mu, sigma, amp, tmin, tmax, nloopsSig, nloopsBkg, filename)
