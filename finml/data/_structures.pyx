import cython
import numpy as np
cimport numpy as np

from numpy cimport (
    float64_t,
    int64_t,
    int32_t,
    ndarray
)

np.import_array()

cdef float64_t MAX_FLOAT = np.finfo(np.float64).max
cdef float64_t MIN_FLOAT = np.finfo(np.float64).min


cdef class _RunningBar:
    cdef:
        float64_t time
        float64_t open
        float64_t high
        float64_t low
        float64_t close
        float64_t volume
        float64_t nticks
        float64_t exmv


    def __init__(self):
        self.time = 0
        self.open = 0
        self.high = MIN_FLOAT
        self.low = MAX_FLOAT
        self.close = 0
        self.volume = 0
        self.nticks = 0
        self.exmv = 0


    cpdef void reset(self):
        self.time = 0
        self.open = 0
        self.high = MIN_FLOAT
        self.low = MAX_FLOAT
        self.close = 0
        self.volume = 0
        self.nticks = 0
        self.exmv = 0


    cdef void _update_time(self, float64_t time):
        if not self.time:
            self.time = time


    cdef void _update_open(self, float64_t price):
        if not self.open:
            self.open = price


    cdef void _update_high(self, float64_t price):
        self.high = max(self.high, price)


    cdef void _update_low(self, float64_t price):
        self.low = min(self.low, price)


    cdef void _update_close(self, float64_t price):
        self.close = price


    cdef void _update_volume(self, float64_t volume):
        self.volume += volume


    cdef void _update_nticks(self):
        self.nticks += 1


    cdef void _update_exmv(self, float64_t price, float64_t volume):
        self.exmv += volume * price


    cpdef void update(self, float64_t time, float64_t price, float64_t volume):
        self._update_time(time)
        self._update_open(price)
        self._update_high(price)
        self._update_low(price)
        self._update_close(price)
        self._update_volume(volume)
        self._update_nticks()
        self._update_exmv(price, volume)


    cpdef ndarray get_ohlcv(self):
        return np.array([self.time, self.open, self.high, self.low, self.close, self.volume])



cdef class _Window:

    cdef:
        int32_t win_size
        int32_t ix
        int32_t n
        float64_t alpha
        ndarray w
        readonly ndarray order
        readonly ndarray window


    def __init__(self, int32_t win_size):
        self.win_size = win_size
        self.order = np.zeros(shape=win_size, dtype=np.int32)
        self.window = np.zeros(shape=win_size, dtype=np.float64)

        self.ix = 0
        self.n = 0

        self.alpha = 2/(self.win_size+1)
        self.w = (1.0 - self.alpha)**np.arange(win_size, dtype=np.float64)


    cpdef append(self, float64_t x):
        self.window[self.ix] = x

        self.order = self.order + 1
        self.order[self.ix] = 0

        self.ix = (self.ix + 1) % self.win_size
        self.n += 1


    cpdef bint isfull(self):
        if self.n >= self.win_size:
            return True
        return False


    cpdef int32_t nelements(self):
        if self.n >= self.win_size:
            return self.win_size
        return self.ix


    cpdef float64_t ewa(self):
        w = self.w[self.order[:self.n]]
        # Round to 10 decimal places to avoid accumulating floating point precision problems
        return round(sum(self.window[:self.n] * w) / sum(w), 10)



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef _single_partition_build_stdbars(float64_t[:,:] values,
                                      str bartype,
                                      float64_t th,
                                      partition_build_state: dict):

    assert th > 0

    cdef:
        _RunningBar running_bar

        Py_ssize_t nrows
        Py_ssize_t nbars

        float64_t time
        float64_t price
        float64_t volume

        float64_t[:,::1] bars

        bint update_at_end_of_loop
        bint cond

    try:
        running_bar = partition_build_state['running_bar']
    except KeyError:
        running_bar = _RunningBar()

    nbars = 0
    nrows = values.shape[0]
    update_at_end_of_loop = bartype == 'time'
    bars = np.empty(shape=(nrows, 6), dtype=np.float64)

    for i in range(nrows):
        time = values[i, 0]
        price = values[i, 1]
        volume = values[i, 2]

        if not update_at_end_of_loop:
            running_bar.update(time, price, volume)

        if bartype == 'time':
            time = time // th * th
            cond = (running_bar.time < time) and running_bar.time

        elif bartype == 'tick':
            cond = running_bar.nticks >= th

        elif bartype == 'volume':
            cond = running_bar.volume >= th

        elif bartype == 'dollar':
            cond = running_bar.exmv >= th

        if cond:
            bars[nbars, 0] = running_bar.time
            bars[nbars, 1] = running_bar.open
            bars[nbars, 2] = running_bar.high
            bars[nbars, 3] = running_bar.low
            bars[nbars, 4] = running_bar.close
            bars[nbars, 5] = running_bar.volume
            running_bar.reset()
            nbars += 1

        if update_at_end_of_loop:
            running_bar.update(time, price, volume)

    bars = bars[:nbars]
    partition_build_state['running_bar'] = running_bar
    return bars, partition_build_state



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef _single_partition_build_imbbars(float64_t[:,:] values,
                                      str bartype,
                                      int64_t win_size_bv,
                                      int64_t win_size_T,
                                      float64_t init_expected_T,
                                      partition_build_state: dict):

    assert win_size_bv > 0
    assert win_size_T > 0
    assert init_expected_T > 0
    assert bartype != 'time'

    cdef:
        _RunningBar running_bar

        Py_ssize_t nrows
        Py_ssize_t nbars

        float64_t time
        float64_t price
        float64_t volume

        float64_t[:,::1] bars

        float64_t imbalance
        float64_t expected_T
        float64_t expected_bv
        float64_t prev_price
        float64_t prev_b
        float64_t delta_p
        float64_t b
        float64_t v

        bint init_phase

        _Window win_bv
        _Window win_T


    try:
        running_bar = partition_build_state['running_bar']
        imbalance = partition_build_state['imbalance']
        prev_b = partition_build_state['prev_b']
        prev_price = partition_build_state['prev_price']
        win_bv = partition_build_state['win_bv']
        win_T = partition_build_state['win_T']
        expected_bv = partition_build_state['expected_bv']
        expected_T = partition_build_state['expected_T']
        init_phase = partition_build_state['init_phase']

    except KeyError as e:
        print(f"Missing key '{e.args[0]}' in partition_build_state.")
        running_bar = _RunningBar()
        imbalance = 0
        prev_b = 0
        prev_price = 0
        win_bv = _Window(win_size_bv)
        win_T = _Window(win_size_T)
        expected_bv = 0
        expected_T = init_expected_T
        init_phase = True

    nbars = 0
    nrows = values.shape[0]
    bars = np.empty(shape=(nrows, 6), dtype=np.float64)

    for i in range(nrows):
        time = values[i, 0]
        price = values[i, 1]
        volume = values[i, 2]

        running_bar.update(time, price, volume)

        # Tick-rule
        delta_p = price - prev_price
        if delta_p == 0:
            b = prev_b
        else:
            b = abs(delta_p) / delta_p

        if bartype == 'tick':
            v = 1
        elif bartype == 'volume':
            v = volume
        elif bartype == 'dollar':
            v = volume * price
        imbalance += b * v

        win_bv.append(b*v)

        if win_bv.isfull() and init_phase:
            expected_bv = win_bv.ewa()
            init_phase = False

        # print(f"loop_num:    {i}\n"
        #       f"init_phase:  {init_phase}\n"
        #       f"imbalance:   {imbalance}\n"
        #       f"b:           {b}\n"
        #       f"prev_b:      {prev_b}\n"
        #       f"price:       {price}\n"
        #       f"prev_price:  {prev_price}\n"
        #       f"expected_bv: {expected_bv:.10f}\n"
        #       f"expected_T:  {expected_T:.10f}\n"
        #       f"-------------------------\n")

        if abs(imbalance) >= expected_T * abs(expected_bv) and not init_phase:
            bars[nbars, 0] = running_bar.time
            bars[nbars, 1] = running_bar.open
            bars[nbars, 2] = running_bar.high
            bars[nbars, 3] = running_bar.low
            bars[nbars, 4] = running_bar.close
            bars[nbars, 5] = running_bar.volume

            win_T.append(running_bar.nticks)

            running_bar.reset()
            imbalance = 0
            nbars += 1

            expected_bv = win_bv.ewa()
            expected_T = win_T.ewa()

        prev_price = price
        prev_b = b

    bars = bars[:nbars]
    partition_build_state['running_bar'] = running_bar
    partition_build_state['imbalance'] = imbalance
    partition_build_state['prev_b'] = prev_b
    partition_build_state['prev_price'] = prev_price
    partition_build_state['win_bv'] = win_bv
    partition_build_state['win_T'] = win_T
    partition_build_state['expected_bv'] = expected_bv
    partition_build_state['expected_T'] = expected_T
    partition_build_state['init_phase'] = init_phase
    return bars, partition_build_state
