try:
    import matlab
    import matlab.engine
finally:
    pass

import inspect
import io
from typing import Tuple, Sequence

import numpy as np
from numpy import ndarray


class CallableSingletonMeta(type):
    def __call__(cls, *args, **kwargs):
        if hasattr(cls, 'instance') and cls.instance:
            if (args or kwargs) and callable(cls.instance):
                return cls.instance(*args, **kwargs)
            else:
                return cls.instance
        else:
            if len(inspect.getfullargspec(cls.__init__)[0]) == 1:
                instance = type.__call__(cls)
                if (args or kwargs) and callable(instance):
                    return instance(*args, **kwargs)
                else:
                    return instance
            else:
                return type.__call__(cls, *args, **kwargs)


class Evaluation(metaclass=CallableSingletonMeta):
    __slots__ = ('eng', 'strio')

    instance = None
    metrics = ('PESQ', 'STOI')

    def __init__(self):
        self.eng = matlab.engine.start_matlab('-nojvm')
        self.eng.addpath(self.eng.genpath('./matlab_lib'))
        # self.eng.maxNumCompThreads(os.cpu_count()//4)
        self.strio = io.StringIO()
        Evaluation.instance: Evaluation = self

    def __call__(self, clean: ndarray, noisy: ndarray, fs: int) -> Tuple[Sequence[str], tuple]:
        clean = matlab.double(clean.tolist())
        noisy = matlab.double(noisy.tolist())
        fs = matlab.double([fs])
        results = self.eng.se_eval(clean, noisy, fs, nargout=2, stdout=self.strio)

        return Evaluation.metrics, results

    def _exit(self):
        self.eng.quit()

        # fname = datetime.now().strftime('log_matlab_eng_%Y-%m-%d %H.%M.%S.txt')
        # with io.open(fname, 'w') as flog:
        #     self.strio.seek(0)
        #     shutil.copyfileobj(self.strio, flog)

        self.strio.close()

    def __del__(self):
        self._exit()
