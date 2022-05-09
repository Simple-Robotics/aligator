# Overview

The objective of this library is to model and solve optimal control problems (OCPs) of the form

$$
    \min_{x,u} \int_0^T \ell(x, u)\, dt + \ell_\mathrm{f}(x(T))
$$
