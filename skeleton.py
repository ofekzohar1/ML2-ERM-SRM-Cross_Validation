#################################
# Your name: Ofek Zohar
#################################

import numpy as np
import matplotlib.pyplot as plt
import intervals
import portion as I  # portion package for intervals, install with pip

# Constants
Ix1 = I.closed(0, 0.2) | I.closed(0.4, 0.6) | I.closed(0.8, 1)
Ix2 = I.closed(0, 1) - Ix1
Py1x1 = 0.8  # P(y=1|x in x1)
Py0x1 = 1 - Py1x1  # P(y=0|x in x1)
Py1x2 = 0.1  # P(y=1|x in x2)
Py0x2 = 1 - Py1x2  # P(y=1|x in x2)
DELTA = 0.1


class Assignment2(object):
    """Assignment 2 skeleton.

    Please use these function signatures for this assignment and submit this file, together with the intervals.py.
    """

    def sample_from_D(self, m):
        """Sample m data samples from D.
        Input: m - an integer, the size of the data sample.

        Returns: np.ndarray of shape (m,2) :
                A two dimensional array of size m that contains the pairs where drawn from the distribution P.
        """
        X = np.random.uniform(size=m)
        X.sort()
        pr = ((Py0x1, Py1x1), (Py0x2, Py1x2))
        Y = [np.random.choice([0, 1], 1, p=pr[0 if x in Ix1 else 1]) for x in X]
        return np.column_stack((X, Y))

    def experiment_m_range_erm(self, m_first, m_last, step, k, T):
        """Runs the ERM algorithm.
        Calculates the empirical error and the true error.
        Plots the average empirical and true errors.
        Input: m_first - an integer, the smallest size of the data sample in the range.
               m_last - an integer, the largest size of the data sample in the range.
               step - an integer, the difference between the size of m in each loop.
               k - an integer, the maximum number of intervals.
               T - an integer, the number of times the experiment is performed.

        Returns: np.ndarray of shape (n_steps,2).
            A two dimensional array that contains the average empirical error
            and the average true error for each m in the range accordingly.
        """
        m_t_err, m_e_err = [], []
        m_range = np.arange(m_first, m_last + step, step)
        T_range = np.arange(0, T)
        for m in m_range:
            sum_t_err, sum_e_err = 0, 0  # sum the total error
            for t in T_range:
                S = self.sample_from_D(m)  # train set
                h_int, best_err = intervals.find_best_interval(S[:, 0], S[:, 1], k)
                hypo = I.from_data([(I.CLOSED, l, u, I.CLOSED) for l, u in h_int])  # make h_int as Interval obj
                true_err = self.true_err(hypo)  # Calc the true error
                sum_t_err += true_err
                sum_e_err += best_err / m
            m_t_err.append(sum_t_err / T)
            m_e_err.append(sum_e_err / T)
        #  plot the errors as a function of m (or n)
        plt.plot(m_range, m_e_err, label='Empirical Error')
        plt.plot(m_range, m_t_err, label='True Error')
        plt.legend()
        plt.xlabel("# of samples (n)")
        plt.ylabel("Error")
        plt.show()

        return np.column_stack((m_e_err, m_t_err))

    def experiment_k_range_erm(self, m, k_first, k_last, step):
        """Finds the best hypothesis for k= 1,2,...,10.
        Plots the empirical and true errors as a function of k.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals in the first experiment.
               m_last - an integer, the maximum number of intervals in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the ERM algorithm.
        """
        S = self.sample_from_D(m)  # train set
        k_t_err, k_e_err = [], []
        k_range = np.arange(k_first, k_last + step, step)
        for k in k_range:
            h_int, best_err = intervals.find_best_interval(S[:, 0], S[:, 1], k)
            hypo = I.from_data([(I.CLOSED, l, u, I.CLOSED) for l, u in h_int])
            true_err = self.true_err(hypo)
            k_t_err.append(true_err)
            k_e_err.append(best_err / m)
        #  plot the errors as a function of k
        plt.plot(k_range, k_e_err, label='Empirical Error')
        plt.plot(k_range, k_t_err, label='True Error')
        plt.legend()
        plt.xlabel("# of max intervals (k)")
        plt.ylabel("Error")
        plt.show()

        return k_range[np.argmin(k_e_err)]  # k minimize emp error (could be more than one)

    def experiment_k_range_srm(self, m, k_first, k_last, step):
        """Run the experiment in (c).
        Plots additionally the penalty for the best ERM hypothesis.
        and the sum of penalty and empirical error.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals in the first experiment.
               m_last - an integer, the maximum number of intervals in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the SRM algorithm.
        """
        S = self.sample_from_D(m)
        k_t_err, k_e_err, k_penalty = [], [], []
        k_range = np.arange(k_first, k_last + step, step)
        for k in k_range:
            h_int, best_err = intervals.find_best_interval(S[:, 0], S[:, 1], k)
            hypo = I.from_data([(I.CLOSED, l, u, I.CLOSED) for l, u in h_int])
            true_err = self.true_err(hypo)  # Calc the true error
            k_t_err.append(true_err)
            k_e_err.append(best_err / m)
            k_penalty.append(self.penalty(m, k, DELTA))  # calc the penalty for each k
        #  plot the errors as a function of k
        plt.plot(k_range, k_e_err, label='Empirical Error')
        plt.plot(k_range, k_t_err, label='True Error')
        plt.plot(k_range, k_penalty, label='Penalty')
        emp_and_penalty = np.array(k_e_err) + np.array(k_penalty)  # Emp error + penalty
        plt.plot(k_range, emp_and_penalty, label='Empirical Error + Penalty')
        plt.legend()
        plt.xlabel("# of max intervals (k)")
        plt.ylabel("Error")
        plt.show()

        return k_range[np.argmin(emp_and_penalty)]  # k minimize emp error + penalty (could be more than one)

    def cross_validation(self, m):
        """Finds a k that gives a good test error.
        Input: m - an integer, the size of the data sample.

        Returns: The best k value (an integer) found by the cross validation algorithm.
        """
        S = self.sample_from_D(m)
        np.random.shuffle(S)
        S1 = S[: (m // 5), :]
        S2 = S[(m // 5):, :]
        S1 = S1[S1[:, 0].argsort()]  # train set
        S2 = S2[S2[:, 0].argsort()]  # validation set
        k_e_s2_err, hypos = [], []
        k_range = np.arange(1, 11, 1)
        for k in k_range:
            h_int, best_err = intervals.find_best_interval(S1[:, 0], S1[:, 1], k)  # ERM on S1
            hypo = I.from_data([(I.CLOSED, l, u, I.CLOSED) for l, u in h_int])
            k_e_s2_err.append(self.emp_err(hypo, S2))
            hypos.append(hypo)
        min_index = np.argmin(k_e_s2_err)
        return k_range[min_index], hypos[min_index]  # k and hypo from ERM hypos that minimize emp err for S2

    #################################
    # Place for additional methods

    """
    Calculate Intervals length == F(interval) for x dist uniformly
    """
    def int_len(self, interval):
        length = 0
        for i in interval:  # Enumerate all intervals in the union
            length += i.upper - i.lower
        return length

    """
    Calculate the True error given the hypo intervals
    """
    def true_err(self, hypo):
        #  4 Intervals - h intersection with x1/2 and h complement with x1/2
        h1x1 = hypo & Ix1
        h1x2 = hypo & Ix2
        h0x1 = (~hypo) & Ix1
        h0x2 = (~hypo) & Ix2

        #  Using the law of total Expectation & zero-one loss func
        err = self.int_len(h1x1) * Py0x1 + self.int_len(h0x1) * Py1x1
        err += self.int_len(h1x2) * Py0x2 + self.int_len(h0x2) * Py1x2
        return err

    """
    Calculate the penalty function using m, k and delta
    """
    def penalty(self, m, k, delta):
        #  2k = VCdim(Hk)
        return 2 * np.sqrt((2 * k + np.log(2 / delta)) / m)

    """
    Calculate the True error given the hypo intervals and train set S
    """
    def emp_err(self, hypo, S):
        err = 0
        for x, y in S:
            hx = 1 if x in hypo else 0
            err += abs(hx - y)
        return err / S.shape[0]
    #################################


if __name__ == '__main__':
    ass = Assignment2()
    ass.experiment_m_range_erm(10, 100, 5, 3, 100)
    ass.experiment_k_range_erm(1500, 1, 10, 1)
    ass.experiment_k_range_srm(1500, 1, 10, 1)
    ass.cross_validation(1500)
