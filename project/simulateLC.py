import numpy as np
import matplotlib.pyplot as plt

def gaussian(a, b, c, x):
    gauss = a*np.exp(-pow((x-b),2)/(2*c*c))
    return gauss


def makeLC(mu, sigma, amp, tmin, tmax, scaled):

    r = np.random.normal(mu, sigma, 100)
    gauss = gaussian(amp, mu, sigma, r)
    bins = np.arange(tmin, tmax, 1)
    inds = np.digitize(r, bins)

    gaussDict = {}

    for counter, value in enumerate(inds):
        # print gauss[counter], bins[value-1]
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
    exposure = np.random.uniform(ex1, ex2, size=61)

    err1 = 1.0
    err2 = 5.0
    error = np.random.uniform(err1, err2, size=61)

    for k in counter_rand_gauss.keys():
        counter_rand_gauss[k].append(float(np.random.uniform(err1, err2, 1)))
        counter_rand_gauss[k].append(float(np.random.uniform(ex1, ex2, 1)))

    od_res = collections.OrderedDict(sorted(counter_rand_gauss.items()))

    b_rand_gauss = list(od_res.keys())
    v_rand_gauss = list(od_res.values())

    b_rand_gauss = np.asarray(b_rand_gauss)
    v_rand_gauss = np.asarray(v_rand_gauss)

    # background only
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

    if scaled==True:

        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111)
        ax.errorbar(b_rand_gauss, v_rand_gauss[:, 0] / v_rand_gauss[:, 2], yerr=v_rand_gauss[:, 1] / v_rand_gauss[:, 2],
                    fmt='o', label='background + signal')
        ax.errorbar(b_rand, v_rand[:, 0] / v_rand[:, 2], yerr=v_rand[:, 1] / v_rand[:, 2], fmt='o', label='background')
        ax.xaxis.get_major_formatter().set_useOffset(False)
        plt.xlabel('time [MJD]')
        plt.ylabel('$\phi$ [ph s$^{-1}$ cm$^2$]')
        plt.tight_layout()
        plt.legend(scatterpoints=1, loc='best')
        # plt.savefig(outPNG+'/lightcurve_gauss_'+str(mu)+'_'+str(sigma)+'_'+str(amp)+'_'+str(idx)+'_scaled.png')
        plt.show()
        plt.close()

    else:

        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111)
        ax.errorbar(b_rand_gauss, v_rand_gauss[:, 0], yerr=v_rand_gauss[:, 1], fmt='o', label='background + signal')
        ax.errorbar(b_rand, v_rand[:, 0], yerr=v_rand[:, 1], fmt='o', label='background')
        ax.xaxis.get_major_formatter().set_useOffset(False)
        plt.xlabel('time [MJD]')
        plt.ylabel('$\phi$ [ph s$^{-1}$ cm$^2$]')
        plt.tight_layout()
        plt.legend(scatterpoints=1, loc='best')
        # plt.savefig(outPNG+'/lightcurve_gauss_'+str(mu)+'_'+str(sigma)+'_'+str(amp)+'_'+str(idx)+'.png')
        plt.show()
        plt.close()


if __name__ == '__main__':

    makeLC(mu, sigma, amp, tmin, tmax, scaled)
