"""诊断：runner 的 rms=110% 与直接 fit_voigt 的 rms=1.97% 差在哪。

假设：差异不在采样（已排除预算/先验/sigma_floor），而在 `metrics` 的
rms 定义或 `_pointwise_median_z` 的取数。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import bm_real as BR
import bm_select as SEL
import circuit_runner as RU
import spectrum_metrics as SM

P = ("D:/eis训练数据/测试数据/腐蚀/Uniform_Generated data and Nyquist plots/"
     "Generated data and Nyquist plots/"
     "R1-[R2,P3]-[1000.0, 2512134.2009090395, 9012389.983760463, "
     "0.7761076465819079].csv")

f, Z = BR.load_spectrum(P)
KW = dict(RU.BUDGETS["quick"])

c = SEL.fit_voigt(f, Z, 1, seed=0, include_L=False, sigma_floor=0.0,
                  kw=KW, sigma=None)
raw = c["raw"]
cfg = {"K": 1, "include_L": False}

Zm = RU._pointwise_median_z(raw, "voigt", cfg, f)
par = RU._median_params_voigt(raw)
Zm2 = __import__("bm_core").model_z(f, par, include_L=False)

m1 = SM.metrics(f, Z, Zm)
m2 = SM.metrics(f, Z, Zm2)


def my_rms(Zc):
    return 100 * np.sqrt(np.mean(np.abs(Zc - Z) ** 2)) / np.sqrt(
        np.mean(np.abs(Z) ** 2))


print("逐样本中位曲线：SM.metrics=%.3f%%  手算=%.3f%%"
      % (m1["rms_rel"], my_rms(Zm)))
print("参数中位曲线  ：SM.metrics=%.3f%%  手算=%.3f%%"
      % (m2["rms_rel"], my_rms(Zm2)))
print("Zm 前3 =", np.round(Zm[:3], 1))
print("Z  前3 =", np.round(Z[:3], 1))
print("|Z| 均值=%.3g  最大=%.3g" % (np.mean(np.abs(Z)), np.abs(Z).max()))
print("残差最大=%.3g  中位=%.3g" % (np.abs(Zm - Z).max(),
                                    np.median(np.abs(Zm - Z))))
print("SM.metrics 全部键:", sorted(m1.keys()))
