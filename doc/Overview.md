# Formulation

The objective of this library is to model and solve optimal control problems (OCPs) of the form

$$
    \min_{x,u} \int_0^T \ell(x, u)\, dt + \ell_\mathrm{f}(x(T)),
    \quad
    \\dot{x}(t) = f(x(t), u(t)).
$$

## Docs: Known issues

Under `doxygen==1.9.2`, class collaboration diagrams will *not* pick up relationships stemming from `shared_ptr<T>` members.
